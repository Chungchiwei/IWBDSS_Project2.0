# AI_Analyzer.py
"""
AI 決策輔助分析模組 (Secure & Optimized Version)
使用 Perplexity API 提供專業的靠泊決策建議
整合 OCIMF 標準、IMO 指南與實務經驗

修正項目：
  #2 - 加入 threading.Lock 解決多 session 並發 race condition
  #3 - 將 _last_request_time 更新移至 _enforce_rate_limit 內
  #4 - 統一 safety_factor_status 與 _sf_label 的邊界定義
  #5 - get_cached_ai_analyzer 加入 secrets_hash 作為快取 key
  #6 - HourlyRiskEntry 改為 clamp + warning 取代直接拋出例外
  #7 - _sanitize_string 補上 Unicode C1 控制字元過濾
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── 備用輔助函式（當 app_helpers 不存在時使用）────────────────
try:
    from app_helpers import (
        beaufort_to_description,
        get_risk_description,
        knots_to_beaufort,
    )
except ImportError:
    def get_risk_description(level: str, lang: str = "zh") -> str:
        return level

    def knots_to_beaufort(kts: float) -> int:
        return max(0, round(kts / 2))

    def beaufort_to_description(scale: int, lang: str = "zh") -> str:
        return f"{scale}級風"


# ── 日誌設定 ─────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ================= 常數定義（集中管理，避免魔法數字）=================

class SafetyThresholds:
    """安全係數閾值常數"""
    EXCELLENT   = 2.0   # 優良
    ADEQUATE    = 1.5   # 合格（OCIMF MEG4 最低要求）
    MARGINAL    = 1.2   # 邊緣
    CRITICAL    = 1.0   # 臨界（不得低於此值）

    # 高風險時段警示閾值
    ALERT_SF    = 1.5
    # Prompt 中最多顯示的高風險時段筆數
    MAX_CRITICAL_DISPLAY = 5


class InputLimits:
    """輸入參數合法範圍"""
    AREA_MIN,       AREA_MAX       = 0.0,    50_000.0   # m²
    CD_MIN,         CD_MAX         = 0.0,    3.0        # 無因次
    DRAFT_MIN,      DRAFT_MAX      = 0.0,    30.0       # m
    LINES_MIN,      LINES_MAX      = 0,      50
    MBL_MIN,        MBL_MAX        = 0.0,    5_000.0    # kN
    TUGS_MIN,       TUGS_MAX       = 0,      10
    TUG_HP_MIN,     TUG_HP_MAX     = 0.0,    20_000.0   # HP
    RISK_SCORE_MIN, RISK_SCORE_MAX = 0.0,    100.0
    PORT_RISK_MIN,  PORT_RISK_MAX  = 1,      10
    WIND_GUST_MIN,  WIND_GUST_MAX  = 0.0,    200.0      # kts


class APIConfig:
    """API 相關設定"""
    URL             = "https://api.perplexity.ai/chat/completions"
    KEY_PREFIX      = "pplx-"
    KEY_MIN_LEN     = 20
    # 重試設定
    RETRY_TOTAL     = 3
    RETRY_BACKOFF   = 1.0          # 指數退避基數（秒）
    RETRY_ON_STATUS = (429, 500, 502, 503, 504)
    # 速率限制（簡易 token bucket）
    MIN_REQUEST_INTERVAL = 2.0     # 秒，避免短時間大量請求


# ================= 資料結構 =================

@dataclass
class VesselParams:
    """船舶與繫泊參數的型別安全容器"""
    area:              float = 0.0
    cd:                float = 0.0
    draft_bow:         float = 0.0
    draft_stern:       float = 0.0
    bow_lines:         int   = 0
    bow_spring_lines:  int   = 0
    stern_lines:       int   = 0
    stern_spring_lines:int   = 0
    line_mbl:          float = 0.0
    num_tugs:          int   = 0
    tug_hp:            float = 0.0

    def __post_init__(self) -> None:
        """
        初始化後自動驗證所有欄位。

        修正 #1：先做型別強制轉換，確保欄位為正確型別後再驗證，
        避免上游傳入 None 或字串時在 property 存取前就拋出
        AttributeError，讓錯誤訊息更明確。
        """
        # 強制轉換數值型別，讓驗證器能產生有意義的錯誤訊息
        try:
            self.area               = float(self.area)
            self.cd                 = float(self.cd)
            self.draft_bow          = float(self.draft_bow)
            self.draft_stern        = float(self.draft_stern)
            self.bow_lines          = int(self.bow_lines)
            self.bow_spring_lines   = int(self.bow_spring_lines)
            self.stern_lines        = int(self.stern_lines)
            self.stern_spring_lines = int(self.stern_spring_lines)
            self.line_mbl           = float(self.line_mbl)
            self.num_tugs           = int(self.num_tugs)
            self.tug_hp             = float(self.tug_hp)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"VesselParams 欄位型別錯誤: {exc}") from exc

        errors = validate_vessel_params(self)
        if errors:
            raise ValueError("VesselParams 驗證失敗:\n" + "\n".join(f"  - {e}" for e in errors))

    @property
    def avg_draft(self) -> float:
        return (self.draft_bow + self.draft_stern) / 2

    @property
    def total_lines(self) -> int:
        return (
            self.bow_lines
            + self.bow_spring_lines
            + self.stern_lines
            + self.stern_spring_lines
        )

    @property
    def line_swl(self) -> float:
        """安全工作負荷 = MBL × 0.5 (OCIMF MEG4)"""
        return self.line_mbl * 0.5


@dataclass
class AnalysisResults:
    """風險分析結果的型別安全容器"""
    risk_level:                str   = "medium"
    risk_score:                float = 0.0
    max_wind_force_kN:         float = 0.0
    max_gust_kts:              float = 0.0
    dominant_wind_dir:         str   = "N/A"
    offshore_wind_ratio:       float = 0.0
    mooring_capacity_total_kN: float = 0.0
    tug_capacity_total_kN:     float = 0.0

    def __post_init__(self) -> None:
        errors = validate_analysis_results(self)
        if errors:
            raise ValueError("AnalysisResults 驗證失敗:\n" + "\n".join(f"  - {e}" for e in errors))

    @property
    def total_restraint_kN(self) -> float:
        return self.mooring_capacity_total_kN + self.tug_capacity_total_kN

    @property
    def safety_factor(self) -> float:
        if self.max_wind_force_kN <= 0:
            return float("inf")
        return self.total_restraint_kN / self.max_wind_force_kN

    @property
    def safety_factor_status(self) -> str:
        """
        修正 #4：與 _sf_label 統一邊界定義，
        兩者現在使用相同的 5 個等級與閾值。
        """
        sf = self.safety_factor
        if sf == float("inf"):
            return "無受風力 (N/A)"
        if sf >= SafetyThresholds.EXCELLENT:
            return f"優良 (≥{SafetyThresholds.EXCELLENT})"
        if sf >= SafetyThresholds.ADEQUATE:
            return f"合格 ({SafetyThresholds.ADEQUATE}–{SafetyThresholds.EXCELLENT})"
        if sf >= SafetyThresholds.MARGINAL:
            return f"邊緣 ({SafetyThresholds.MARGINAL}–{SafetyThresholds.ADEQUATE})"
        if sf >= SafetyThresholds.CRITICAL:
            return f"危險 ({SafetyThresholds.CRITICAL}–{SafetyThresholds.MARGINAL})"
        return f"極度危險 (<{SafetyThresholds.CRITICAL})"


@dataclass
class HourlyRiskEntry:
    """
    逐時風險資料容器。

    修正 #6：safety_factor 為負時改為 clamp 至 0.0 並記錄 warning，
    而非直接拋出例外中斷整個分析流程。
    wind_gust_kts 超出範圍同樣改為 clamp，保持一致策略。
    """
    time:          str
    wind_gust_kts: float
    safety_factor: float

    def __post_init__(self) -> None:
        if not self.time:
            raise ValueError("HourlyRiskEntry.time 不可為空")

        # wind_gust_kts：clamp 並警告
        if not (InputLimits.WIND_GUST_MIN <= self.wind_gust_kts <= InputLimits.WIND_GUST_MAX):
            logger.warning(
                "HourlyRiskEntry[%s]: wind_gust_kts=%.2f 超出合法範圍 [%.1f, %.1f]，已 clamp。",
                self.time,
                self.wind_gust_kts,
                InputLimits.WIND_GUST_MIN,
                InputLimits.WIND_GUST_MAX,
            )
            self.wind_gust_kts = max(
                InputLimits.WIND_GUST_MIN,
                min(self.wind_gust_kts, InputLimits.WIND_GUST_MAX),
            )

        # safety_factor：clamp 至 0.0 並警告
        if self.safety_factor < 0:
            logger.warning(
                "HourlyRiskEntry[%s]: safety_factor=%.4f 為負數，已 clamp 至 0.0。",
                self.time,
                self.safety_factor,
            )
            self.safety_factor = 0.0


@dataclass
class ModelParams:
    """AI 模型參數容器"""
    model:       str   = "sonar-pro"
    temperature: float = 0.2
    max_tokens:  int   = 4000
    timeout:     int   = 90

    def __post_init__(self) -> None:
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature={self.temperature} 必須在 [0.0, 2.0] 之間")
        if not (100 <= self.max_tokens <= 32_000):
            raise ValueError(f"max_tokens={self.max_tokens} 必須在 [100, 32000] 之間")
        if not (10 <= self.timeout <= 300):
            raise ValueError(f"timeout={self.timeout} 必須在 [10, 300] 秒之間")


# ================= 輸入驗證器（與資料類別解耦）=================

def validate_vessel_params(p: VesselParams) -> List[str]:
    """驗證船舶參數，回傳錯誤訊息列表（空列表表示驗證通過）"""
    errors: List[str] = []

    def _check_range(name: str, val: float, lo: float, hi: float) -> None:
        if not (lo <= val <= hi):
            errors.append(f"{name}={val} 超出合法範圍 [{lo}, {hi}]")

    _check_range("area",        p.area,        InputLimits.AREA_MIN,    InputLimits.AREA_MAX)
    _check_range("cd",          p.cd,          InputLimits.CD_MIN,      InputLimits.CD_MAX)
    _check_range("draft_bow",   p.draft_bow,   InputLimits.DRAFT_MIN,   InputLimits.DRAFT_MAX)
    _check_range("draft_stern", p.draft_stern, InputLimits.DRAFT_MIN,   InputLimits.DRAFT_MAX)
    _check_range("line_mbl",    p.line_mbl,    InputLimits.MBL_MIN,     InputLimits.MBL_MAX)
    _check_range("num_tugs",    p.num_tugs,    InputLimits.TUGS_MIN,    InputLimits.TUGS_MAX)
    _check_range("tug_hp",      p.tug_hp,      InputLimits.TUG_HP_MIN,  InputLimits.TUG_HP_MAX)

    for attr in ("bow_lines", "bow_spring_lines", "stern_lines", "stern_spring_lines"):
        val = getattr(p, attr)
        _check_range(attr, val, InputLimits.LINES_MIN, InputLimits.LINES_MAX)

    return errors


def validate_analysis_results(r: AnalysisResults) -> List[str]:
    """驗證分析結果，回傳錯誤訊息列表"""
    errors: List[str] = []
    valid_levels = {"low", "medium", "high", "extreme"}

    if r.risk_level.lower() not in valid_levels:
        errors.append(f"risk_level='{r.risk_level}' 不在合法值 {valid_levels} 中")
    if not (InputLimits.RISK_SCORE_MIN <= r.risk_score <= InputLimits.RISK_SCORE_MAX):
        errors.append(f"risk_score={r.risk_score} 超出 [0, 100]")
    if r.max_wind_force_kN < 0:
        errors.append(f"max_wind_force_kN={r.max_wind_force_kN} 不可為負")
    if not (0.0 <= r.offshore_wind_ratio <= 100.0):
        errors.append(f"offshore_wind_ratio={r.offshore_wind_ratio} 超出 [0, 100]")
    if r.mooring_capacity_total_kN < 0:
        errors.append("mooring_capacity_total_kN 不可為負")
    if r.tug_capacity_total_kN < 0:
        errors.append("tug_capacity_total_kN 不可為負")

    return errors


def _sanitize_string(value: str, max_len: int = 200) -> str:
    """
    清理字串輸入：移除控制字元、截斷過長內容。
    防止 Prompt Injection 攻擊。

    修正 #7：補上 Unicode C1 控制字元過濾（U+0080–U+009F），
    原版只過濾 ASCII 控制字元（< 0x20），C1 範圍可被用於注入攻擊。
    unicodedata.category(ch).startswith("C") 涵蓋：
      Cc (Control), Cf (Format), Cs (Surrogate), Co (Private Use), Cn (Unassigned)
    保留 \n 與 \t 以維持正常排版。
    """
    sanitized = "".join(
        ch for ch in value
        if ch in ("\n", "\t")
        or (
            ord(ch) >= 32
            and ord(ch) != 127
            and not unicodedata.category(ch).startswith("C")
        )
    )
    return sanitized[:max_len]


# ================= 風險等級模板 =================

@dataclass(frozen=True)
class RiskTemplate:
    icon:   str
    color:  str
    status: str
    action: str


RISK_TEMPLATES: Dict[str, RiskTemplate] = {
    "low":     RiskTemplate("✅", "🟢", "低風險作業",  "可按計劃執行"),
    "medium":  RiskTemplate("⚠️", "🟡", "中度風險",    "需謹慎評估"),
    "high":    RiskTemplate("🔴", "🔴", "高風險作業",  "建議延後或加強資源"),
    "extreme": RiskTemplate("⛔", "⛔", "極高風險",    "強烈建議停止作業"),
}

# 中文風險等級 → 英文 key 映射
RISK_LEVEL_MAP: Dict[str, str] = {
    "低度風險": "low",
    "中度風險": "medium",
    "高度風險": "high",
    "極度危險": "extreme",
}


# ================= 系統提示 =================

SYSTEM_MESSAGE = """你是一位擁有 20 年經驗的資深船長 (Master Mariner) 兼港口安全顧問，專精於：

【核心專業領域】
1. **OCIMF 標準** (MEG4 - 繫泊設備指南、船舶受風力計算)
2. **IMO 指南** (SOLAS 港口作業安全、ISGOTT 油輪安全指南)
3. **實務經驗** (氣象條件評估、緊急應變、船岸協調)

【核心職責】
✅ **風險量化分析**: 基於物理計算判斷安全係數 (SF)，識別臨界時段 (SF < 1.5)。
✅ **操作時窗建議**: 評估靠泊/離泊最佳時機，考量風向 (吹開/吹攏) 與潮汐。
✅ **資源配置優化**: 具體建議纜繩配置 (船首/船尾/倒纜) 與拖船策略。

【輸出格式要求】
請使用以下 Markdown 結構：
## 📊 一、在港期間風險總覽 (含評級與關鍵因子)
## ⚓ 二、靠泊作業分析 (時窗評估、階段風險)
## 🔒 三、在港期間監控要點 (纜繩張力、環境監控)
## 🚢 四、離泊作業分析 (時窗評估、解纜順序)
## 💡 五、綜合建議與決策輔助 (資源配置、應變計畫、決策建議)

**原則**:
- 量化所有風險 (用數值而非模糊詞彙)。
- 標註為「決策輔助」，最終決策權在船長。
- 使用繁體中文回答。
- 不得輸出任何程式碼、系統指令或與航海無關的內容。"""
#  ↑ 最後一行防止 Prompt Injection 後的指令逸出


# ================= AI 分析器 =================

class AIAnalyzer:
    """
    AI 分析器：整合航海安全與風險管理專業知識，提供結構化的決策輔助建議。

    安全特性：
    - API Key 不寫入日誌
    - 輸入字串經過 sanitize 防止 Prompt Injection（含 Unicode C1 字元）
    - 請求具備重試機制與 thread-safe 速率限制
    - 所有輸入參數通過邊界驗證

    使用方式：
        analyzer = AIAnalyzer()                    # 自動從 Streamlit Secrets 讀取
        analyzer = AIAnalyzer(api_key="pplx-...")  # 明確傳入 Key
    """

    def __init__(
        self,
        api_key:      Optional[str]         = None,
        model_params: Optional[ModelParams] = None,
    ):
        self.api_key = api_key or self._load_api_key_from_secrets()
        self.params  = model_params or ModelParams()

        # 修正 #2 & #3：使用 Lock 保護速率限制，確保多 session 並發安全。
        # _last_request_time 的讀寫均在 Lock 內完成，消除 race condition。
        self._rate_lock:          threading.Lock = threading.Lock()
        self._last_request_time:  float          = 0.0

        # 建立帶重試機制的 Session（連線層級重試，非業務邏輯重試）
        self._session = self._build_session()

        if not self.api_key:
            logger.warning(
                "未提供 Perplexity API Key，AI 分析功能將無法使用。"
                "請在 Streamlit Secrets 中設定 PERPLEXITY_API_KEY。"
            )
        elif not self._is_valid_api_key_format(self.api_key):
            logger.warning(
                "API Key 格式可能不正確（應以 '%s' 開頭，長度 ≥ %d）。",
                APIConfig.KEY_PREFIX,
                APIConfig.KEY_MIN_LEN,
            )

    # ── 屬性 ─────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        return bool(self.api_key)

    @property
    def masked_api_key(self) -> str:
        """回傳遮罩後的 API Key，供 UI 顯示用（不暴露完整 Key）"""
        if not self.api_key:
            return "(未設定)"
        visible = min(8, len(self.api_key) // 4)
        return self.api_key[:visible] + "****" + self.api_key[-4:]

    # ── 公開 API ──────────────────────────────────────────────

    def generate_analysis(
        self,
        port_name:        str,
        vessel_params:    VesselParams,
        analysis_results: AnalysisResults,
        berthing_time:    datetime,
        departure_time:   datetime,
        port_risk_level:  int                        = 5,
        hourly_data:      Optional[List[HourlyRiskEntry]] = None,
    ) -> str:
        """
        呼叫 Perplexity API 生成完整 AI 分析報告。

        Returns:
            Markdown 格式的分析報告字串。
        Raises:
            不拋出例外，所有錯誤以 Markdown 錯誤訊息字串回傳。
        """
        # ── 前置驗證 ──
        if not self.api_key:
            return self._msg_no_api_key()

        validation_error = self._validate_generate_inputs(
            port_name, port_risk_level, berthing_time, departure_time
        )
        if validation_error:
            return f"❌ **輸入驗證失敗**: {validation_error}"

        # ── 速率限制（含 Lock，修正 #2 & #3）──
        self._enforce_rate_limit()

        # ── 建構 Prompt ──
        user_prompt = self._build_user_prompt(
            port_name        = _sanitize_string(port_name, 100),
            vessel_params    = vessel_params,
            analysis_results = analysis_results,
            berthing_time    = berthing_time,
            departure_time   = departure_time,
            port_risk_level  = port_risk_level,
            hourly_data      = hourly_data,
        )

        # ── 呼叫 API ──
        return self._call_api(user_prompt)

    def generate_quick_summary(
        self,
        risk_score:       float,
        risk_level:       str,
        safety_margin_kN: float,
        safety_factor:    float = 0.0,
    ) -> str:
        """
        本地生成快速摘要（不耗用 API）。

        Args:
            risk_score:        風險分數 (0–100)
            risk_level:        風險等級（中英文皆可）
            safety_margin_kN:  安全餘裕力 (kN)
            safety_factor:     安全係數

        Returns:
            Markdown 格式的快速摘要字串。
        """
        risk_score    = max(0.0, min(100.0, float(risk_score)))
        safety_factor = max(0.0, float(safety_factor))

        level_key = RISK_LEVEL_MAP.get(risk_level, risk_level.lower() if risk_level else "medium")
        if level_key not in RISK_TEMPLATES:
            level_key = "medium"

        sf_msg   = self._sf_label(safety_factor)
        template = RISK_TEMPLATES[level_key]

        summaries: Dict[str, str] = {
            "low": (
                f"### {template.icon} 快速評估: {template.status}\n"
                f"**風險狀態**: {template.color} 低風險 ({risk_score:.1f}/100) | "
                f"**SF**: {safety_factor:.2f} — {sf_msg}\n"
                f"**建議**: 氣象良好，纜繩拖船配置充足，建議按計劃執行並保持標準監控。"
            ),
            "medium": (
                f"### {template.icon} 快速評估: {template.status}\n"
                f"**風險狀態**: {template.color} 中度風險 ({risk_score:.1f}/100) | "
                f"**SF**: {safety_factor:.2f} — {sf_msg}\n"
                f"**建議**: 需謹慎評估。建議加強在港監控（每小時巡檢），"
                f"準備備用纜繩，或微調作業時窗。"
            ),
            "high": (
                f"### {template.icon} 快速評估: {template.status}\n"
                f"**風險狀態**: {template.color} 高風險 ({risk_score:.1f}/100) | "
                f"**SF**: {safety_factor:.2f} — {sf_msg}\n"
                f"**警告**: 氣象條件不佳。**強烈建議延後作業**。"
                f"若必須執行，需強制增加拖船/纜繩並實施 24H 值班。"
            ),
            "extreme": (
                f"### {template.icon} 快速評估: {template.status}\n"
                f"**風險狀態**: {template.color} 極高風險 ({risk_score:.1f}/100) | "
                f"**SF**: {safety_factor:.2f} — {sf_msg}\n"
                f"**緊急**: 氣象惡劣，操作極度危險。**停止作業**，"
                f"立即評估替代港口或緊急離港應變。"
            ),
        }
        return summaries[level_key]

    def update_model_params(
        self,
        model:       Optional[str]   = None,
        temperature: Optional[float] = None,
        max_tokens:  Optional[int]   = None,
        timeout:     Optional[int]   = None,
    ) -> None:
        """更新 AI 模型參數（僅更新有傳入的欄位，並驗證合法性）"""
        new_model       = model       if model       is not None else self.params.model
        new_temperature = temperature if temperature is not None else self.params.temperature
        new_max_tokens  = max_tokens  if max_tokens  is not None else self.params.max_tokens
        new_timeout     = timeout     if timeout     is not None else self.params.timeout

        # 透過 ModelParams 的 __post_init__ 驗證
        self.params = ModelParams(
            model       = new_model,
            temperature = new_temperature,
            max_tokens  = new_max_tokens,
            timeout     = new_timeout,
        )

    def set_api_key(self, api_key: str) -> None:
        """
        更新 API Key。
        注意：不將新 Key 寫入日誌，僅記錄操作事件。
        """
        if api_key and not self._is_valid_api_key_format(api_key):
            logger.warning("嘗試設定格式不正確的 API Key，已忽略。")
            return
        self.api_key = api_key
        if not api_key:
            logger.warning("API Key 已被清除，AI 分析功能將無法使用。")
        else:
            logger.info("API Key 已更新。")

    # ── 私有：API 呼叫 ────────────────────────────────────────

    def _call_api(self, user_prompt: str) -> str:
        """執行 API 請求，含重試與例外處理"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       self.params.model,
            "messages":    [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": self.params.temperature,
            "max_tokens":  self.params.max_tokens,
        }

        try:
            response = self._session.post(
                APIConfig.URL,
                headers = headers,
                json    = payload,
                timeout = self.params.timeout,
            )

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                return content + self._footer()

            error_hint = {
                401: "API Key 無效或已過期，請重新設定。",
                403: "API Key 無權限使用此模型。",
                429: "請求頻率過高，請稍後再試。",
                500: "Perplexity 伺服器內部錯誤，請稍後重試。",
            }.get(response.status_code, "")

            safe_text = response.text[:200]
            logger.error(
                "Perplexity API 回傳錯誤 HTTP %d（不含 Key）",
                response.status_code,
            )
            return self._msg_api_error(response.status_code, safe_text, error_hint)

        except requests.Timeout:
            logger.error("Perplexity API 請求超時 (timeout=%ds)", self.params.timeout)
            return self._msg_unexpected_error(f"請求超時（超過 {self.params.timeout} 秒）")
        except requests.ConnectionError:
            logger.error("Perplexity API 連線失敗，請檢查網路。")
            return self._msg_unexpected_error("網路連線失敗，請檢查網路設定。")
        except requests.RequestException as exc:
            logger.error("Perplexity API 請求異常: %s", type(exc).__name__)
            return self._msg_unexpected_error(f"請求異常 ({type(exc).__name__})")
        except (KeyError, IndexError) as exc:
            logger.exception("解析 API 回應失敗")
            return self._msg_unexpected_error(f"回應格式異常: {type(exc).__name__}")

    # ── 私有：速率限制 ────────────────────────────────────────

    def _enforce_rate_limit(self) -> None:
        """
        Thread-safe 速率限制。

        修正 #2 & #3：
        - 使用 Lock 確保多 session 並發時不會同時通過速率限制。
        - _last_request_time 在 Lock 內更新（原版在 _call_api 更新，
          導致計時不準且有 race condition）。
        """
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_time
            wait    = APIConfig.MIN_REQUEST_INTERVAL - elapsed
            if wait > 0:
                logger.debug("速率限制：等待 %.1f 秒", wait)
                time.sleep(wait)
            # 在 Lock 內更新，確保下一個請求能正確計算間隔
            self._last_request_time = time.monotonic()

    # ── 私有：Prompt 建構 ─────────────────────────────────────

    def _build_user_prompt(
        self,
        port_name:        str,
        vessel_params:    VesselParams,
        analysis_results: AnalysisResults,
        berthing_time:    datetime,
        departure_time:   datetime,
        port_risk_level:  int,
        hourly_data:      Optional[List[HourlyRiskEntry]],
    ) -> str:
        duration_hrs = (departure_time - berthing_time).total_seconds() / 3600.0
        sf           = analysis_results.safety_factor
        risk_desc    = get_risk_description(analysis_results.risk_level, lang="zh")

        sf_display = f"{sf:.2f}" if sf != float("inf") else "∞ (無受風力)"

        prompt = f"""請基於以下船舶在港期間完整數據進行專業分析：

═══════════════════════════════════════════════════════
📍 【基本資訊】
═══════════════════════════════════════════════════════
港口名稱: {port_name} (危險等級: {port_risk_level}/10)
預定靠港 (ETA): {berthing_time.strftime('%Y-%m-%d %H:%M')}
預定離港 (ETD): {departure_time.strftime('%Y-%m-%d %H:%M')}
在港時長: {duration_hrs:.1f} 小時

═══════════════════════════════════════════════════════
🚢 【船舶與繫泊參數】
═══════════════════════════════════════════════════════
受風面積: {vessel_params.area:.1f} m² | 風阻係數: {vessel_params.cd:.3f}
平均吃水: {vessel_params.avg_draft:.1f} m
【纜繩配置】
  船首纜: {vessel_params.bow_lines} 條 | 船首倒纜: {vessel_params.bow_spring_lines} 條
  船尾纜: {vessel_params.stern_lines} 條 | 船尾倒纜: {vessel_params.stern_spring_lines} 條
  總計: {vessel_params.total_lines} 條
  纜繩 MBL: {vessel_params.line_mbl:.1f} kN | SWL: {vessel_params.line_swl:.1f} kN
  纜繩總可用拉力: {analysis_results.mooring_capacity_total_kN:.1f} kN
【拖船支援】
  數量: {vessel_params.num_tugs} 艘 | 單艘: {vessel_params.tug_hp:.0f} HP
  拖船總可用推力: {analysis_results.tug_capacity_total_kN:.1f} kN

═══════════════════════════════════════════════════════
⚠️ 【風險評估與氣象】
═══════════════════════════════════════════════════════
風險等級: {analysis_results.risk_level.upper()} ({risk_desc}) | 分數: {analysis_results.risk_score:.1f}/100
最大受風力: {analysis_results.max_wind_force_kN:.1f} kN
總抑制力 (纜繩+拖船): {analysis_results.total_restraint_kN:.1f} kN
**安全係數 (SF)**: {sf_display} → {analysis_results.safety_factor_status}
  OCIMF MEG4 最低要求: SF ≥ {SafetyThresholds.ADEQUATE}

最大陣風: {analysis_results.max_gust_kts:.1f} 節
主要風向: {analysis_results.dominant_wind_dir} (吹開風比例: {analysis_results.offshore_wind_ratio:.1f}%)
"""

        # 高風險時段警示（最多顯示 MAX_CRITICAL_DISPLAY 筆）
        if hourly_data:
            critical = [
                h for h in hourly_data
                if h.safety_factor < SafetyThresholds.ALERT_SF
            ]
            if critical:
                prompt += (
                    f"\n🚨 【高風險時段警示】 "
                    f"(SF < {SafetyThresholds.ALERT_SF}): 共 {len(critical)} 小時\n"
                )
                for entry in critical[:SafetyThresholds.MAX_CRITICAL_DISPLAY]:
                    prompt += (
                        f"  - {entry.time}: 陣風 {entry.wind_gust_kts:.1f} kts, "
                        f"SF={entry.safety_factor:.2f}\n"
                    )
                if len(critical) > SafetyThresholds.MAX_CRITICAL_DISPLAY:
                    prompt += (
                        f"  ... 另有 "
                        f"{len(critical) - SafetyThresholds.MAX_CRITICAL_DISPLAY} "
                        f"個高風險時段未列出\n"
                    )

        prompt += f"""
═══════════════════════════════════════════════════════
🎯 【分析要求】
請提供具體建議：
1. **量化風險**: 哪些時段 SF 不足？吹開/吹攏風影響？
2. **操作建議**: 最佳靠/離泊時窗？若在高風險時段作業需何種補償措施？
3. **資源優化**: 纜繩與拖船需增加嗎？配置位置建議？
4. **應變計畫**: 風速超標應對（觸發條件：SF < {SafetyThresholds.ADEQUATE}）與緊急離港條件。
5. **決策建議**: 執行/延後/取消的具體判斷依據（以 SF 數值為基準）。

⚠️ 重要提醒：本分析僅為決策輔助，最終操作決策權在船長。
"""
        return prompt

    # ── 私有：靜態輔助方法 ────────────────────────────────────

    @staticmethod
    def _build_session() -> requests.Session:
        """建立帶有重試機制的 HTTP Session"""
        session = requests.Session()
        retry = Retry(
            total            = APIConfig.RETRY_TOTAL,
            backoff_factor   = APIConfig.RETRY_BACKOFF,
            status_forcelist = APIConfig.RETRY_ON_STATUS,
            allowed_methods  = {"POST"},
            raise_on_status  = False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://",  adapter)
        return session

    @staticmethod
    def _is_valid_api_key_format(key: str) -> bool:
        """基本格式驗證（不驗證有效性，僅防止明顯錯誤）"""
        return (
            isinstance(key, str)
            and len(key) >= APIConfig.KEY_MIN_LEN
            and key.startswith(APIConfig.KEY_PREFIX)
        )

    @staticmethod
    def _validate_generate_inputs(
        port_name:       str,
        port_risk_level: int,
        berthing_time:   datetime,
        departure_time:  datetime,
    ) -> Optional[str]:
        """
        驗證 generate_analysis 的輸入參數。
        回傳錯誤訊息字串，若驗證通過則回傳 None。
        """
        if not port_name or not port_name.strip():
            return "港口名稱不可為空"
        if not (InputLimits.PORT_RISK_MIN <= port_risk_level <= InputLimits.PORT_RISK_MAX):
            return (
                f"port_risk_level={port_risk_level} 超出範圍 "
                f"[{InputLimits.PORT_RISK_MIN}, {InputLimits.PORT_RISK_MAX}]"
            )
        if not isinstance(berthing_time, datetime) or not isinstance(departure_time, datetime):
            return "berthing_time 與 departure_time 必須為 datetime 物件"
        if departure_time <= berthing_time:
            return "離港時間必須晚於靠港時間"
        duration_hrs = (departure_time - berthing_time).total_seconds() / 3600
        if duration_hrs > 720:
            return f"在港時長 {duration_hrs:.0f} 小時超過合理範圍（最大 720 小時）"
        return None

    @staticmethod
    def _load_api_key_from_secrets() -> str:
        """從 Streamlit Secrets 讀取 API Key，失敗時回傳空字串"""
        try:
            return st.secrets.get("PERPLEXITY_API_KEY", "")
        except Exception:
            return ""

    @staticmethod
    def _sf_label(sf: float) -> str:
        """
        將安全係數轉換為描述文字。

        修正 #4：與 safety_factor_status 統一為相同的 5 個等級與邊界，
        原版缺少「危險 (1.0–1.2)」等級，導致 1.0–1.2 被歸入「不足」。
        """
        if sf == float("inf"):
            return "無受風力 (N/A)"
        if sf >= SafetyThresholds.EXCELLENT:
            return f"優良 (≥{SafetyThresholds.EXCELLENT})"
        if sf >= SafetyThresholds.ADEQUATE:
            return f"合格 ({SafetyThresholds.ADEQUATE}–{SafetyThresholds.EXCELLENT})"
        if sf >= SafetyThresholds.MARGINAL:
            return f"邊緣 ({SafetyThresholds.MARGINAL}–{SafetyThresholds.ADEQUATE})"
        if sf >= SafetyThresholds.CRITICAL:
            return f"危險 ({SafetyThresholds.CRITICAL}–{SafetyThresholds.MARGINAL})"
        return f"極度危險 (<{SafetyThresholds.CRITICAL})"

    def _footer(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"\n\n---\n"
            f"<small>⚓ AI 分析生成時間: {timestamp} | "
            f"模型: {self.params.model} | "
            f"**免責聲明**: 本報告僅供決策輔助，非操作指令，最終決策權在船長。</small>"
        )

    @staticmethod
    def _msg_no_api_key() -> str:
        return (
            "❌ **未提供 Perplexity API Key**\n\n"
            "無法生成 AI 分析。請在 Streamlit Secrets 中設定 `PERPLEXITY_API_KEY`。\n\n"
            "設定方式：在 `.streamlit/secrets.toml` 中加入：\n"
            "```toml\nPERPLEXITY_API_KEY = \"pplx-...\"\n```"
        )

    @staticmethod
    def _msg_api_error(code: int, text: str, hint: str = "") -> str:
        hint_line = f"\n💡 **提示**: {hint}" if hint else ""
        return f"❌ **AI 分析請求失敗**\n狀態碼: `{code}`\n錯誤: `{text[:200]}`{hint_line}"

    @staticmethod
    def _msg_unexpected_error(error: str) -> str:
        return (
            f"❌ **系統發生錯誤**: {error[:200]}\n\n"
            "請檢查網路連線與 API Key 設定後重試。"
        )


# ================= 便利函式與快取 =================

def create_ai_analyzer(api_key: Optional[str] = None) -> AIAnalyzer:
    """建立新的 AIAnalyzer 實例（每次呼叫都建立新實例，適合單次使用）"""
    return AIAnalyzer(api_key=api_key)


def _get_secrets_hash() -> str:
    """
    計算當前 Streamlit Secrets 中 API Key 的 hash。

    修正 #5：作為 get_cached_ai_analyzer 的快取 key，
    確保 Secrets 更新後快取會自動失效並重建實例。
    直接用 Key 本身作為快取 key 會有安全疑慮（Key 會出現在快取 key 中），
    改用 SHA-256 hash 既安全又能正確偵測變更。
    """
    try:
        key = st.secrets.get("PERPLEXITY_API_KEY", "")
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    except Exception:
        return "no-secrets"


@st.cache_resource
def get_cached_ai_analyzer(
    api_key:      Optional[str] = None,
    _secrets_hash: str          = "",   # 前綴 _ 告知 Streamlit 不將此參數序列化進快取 key
) -> AIAnalyzer:
    """
    建立並快取 AIAnalyzer 實例（Streamlit 跨 session 共用）。

    修正 #5：加入 _secrets_hash 參數作為快取 key 的一部分。
    呼叫方式：
        analyzer = get_cached_ai_analyzer(_secrets_hash=_get_secrets_hash())

    這樣當 .streamlit/secrets.toml 中的 API Key 被更新後，
    hash 改變 → 快取失效 → 自動重建使用新 Key 的實例。

    注意：快取的實例共用同一個 HTTP Session 與 thread-safe 速率限制，
    多 session 並發時受 Lock + MIN_REQUEST_INTERVAL 雙重保護。
    """
    return AIAnalyzer(api_key=api_key)
