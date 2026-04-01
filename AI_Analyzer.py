# AI_Analyzer.py
"""
AI 決策輔助分析模組 (Optimized Version)
使用 Perplexity API 提供專業的靠泊決策建議
整合 OCIMF 標準、IMO 指南與實務經驗
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

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
        return round(kts / 2)

    def beaufort_to_description(scale: int, lang: str = "zh") -> str:
        return f"{scale}級風"


# ── 日誌設定 ─────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ================= 資料結構 =================

@dataclass
class VesselParams:
    """船舶與繫泊參數的型別安全容器"""
    area: float = 0.0
    cd: float = 0.0
    draft_bow: float = 0.0
    draft_stern: float = 0.0
    bow_lines: int = 0
    bow_spring_lines: int = 0
    stern_lines: int = 0
    stern_spring_lines: int = 0
    line_mbl: float = 0.0
    num_tugs: int = 0
    tug_hp: float = 0.0

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
        """安全工作負荷 = MBL × 0.5"""
        return self.line_mbl * 0.5


@dataclass
class AnalysisResults:
    """風險分析結果的型別安全容器"""
    risk_level: str = "medium"
    risk_score: float = 0.0
    max_wind_force_kN: float = 0.0
    max_gust_kts: float = 0.0
    dominant_wind_dir: str = "N/A"
    offshore_wind_ratio: float = 0.0
    mooring_capacity_total_kN: float = 0.0
    tug_capacity_total_kN: float = 0.0

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
        sf = self.safety_factor
        if sf >= 2.0:
            return "優良 (≥2.0)"
        if sf >= 1.5:
            return "合格 (≥1.5)"
        if sf >= 1.2:
            return "邊緣 (1.2-1.5)"
        return "不足 (<1.2)"


@dataclass
class HourlyRiskEntry:
    """逐時風險資料容器"""
    time: str
    wind_gust_kts: float
    safety_factor: float


@dataclass
class ModelParams:
    """AI 模型參數容器"""
    model: str = "sonar-pro"
    temperature: float = 0.2
    max_tokens: int = 4000
    timeout: int = 90


# ================= 風險等級模板 =================

@dataclass(frozen=True)
class RiskTemplate:
    icon: str
    color: str
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
- 使用繁體中文回答。"""


# ================= AI 分析器 =================

class AIAnalyzer:
    """
    AI 分析器
    整合航海安全與風險管理專業知識，提供結構化的決策輔助建議。

    使用方式：
        analyzer = AIAnalyzer()                    # 自動從 Streamlit Secrets 讀取
        analyzer = AIAnalyzer(api_key="pplx-...")  # 明確傳入 Key
    """

    API_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_params: Optional[ModelParams] = None,
    ):
        """
        初始化 AI 分析器。

        Args:
            api_key:      Perplexity API Key。
                          優先順序：傳入參數 → Streamlit Secrets → 空字串（不使用硬寫 Key）
            model_params: AI 模型參數，預設使用 ModelParams 預設值。
        """
        self.api_key = api_key or self._load_api_key_from_secrets()
        self.params = model_params or ModelParams()

        if not self.api_key:
            logger.warning(
                "未提供 Perplexity API Key，AI 分析功能將無法使用。"
                "請在 Streamlit Secrets 中設定 PERPLEXITY_API_KEY。"
            )

    # ── 屬性 ─────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        return bool(self.api_key)

    # ── 公開 API ──────────────────────────────────────────────

    def generate_analysis(
        self,
        port_name: str,
        vessel_params: VesselParams,
        analysis_results: AnalysisResults,
        berthing_time: datetime,
        departure_time: datetime,
        port_risk_level: int = 5,
        hourly_data: Optional[List[HourlyRiskEntry]] = None,
    ) -> str:
        """
        呼叫 Perplexity API 生成完整 AI 分析報告。

        Returns:
            Markdown 格式的分析報告字串。
        """
        if not self.api_key:
            return self._msg_no_api_key()

        user_prompt = self._build_user_prompt(
            port_name=port_name,
            vessel_params=vessel_params,
            analysis_results=analysis_results,
            berthing_time=berthing_time,
            departure_time=departure_time,
            port_risk_level=port_risk_level,
            hourly_data=hourly_data,
        )

        try:
            response = requests.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.params.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "temperature": self.params.temperature,
                    "max_tokens":  self.params.max_tokens,
                },
                timeout=self.params.timeout,
            )

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                return content + self._footer()

            logger.error("Perplexity API 回傳錯誤 HTTP %d: %s", response.status_code, response.text[:300])
            return self._msg_api_error(response.status_code, response.text)

        except requests.Timeout:
            logger.error("Perplexity API 請求超時 (timeout=%ds)", self.params.timeout)
            return self._msg_unexpected_error(f"請求超時（超過 {self.params.timeout} 秒）")
        except requests.RequestException as exc:
            logger.exception("Perplexity API 連線錯誤")
            return self._msg_unexpected_error(str(exc))
        except (KeyError, IndexError) as exc:
            logger.exception("解析 API 回應失敗")
            return self._msg_unexpected_error(f"回應格式異常: {exc}")

    def generate_quick_summary(
        self,
        risk_score: float,
        risk_level: str,
        safety_margin_kN: float,
        safety_factor: float = 0.0,
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
        level_key = RISK_LEVEL_MAP.get(risk_level, risk_level.lower() if risk_level else "medium")
        sf_msg = self._sf_label(safety_factor)
        template = RISK_TEMPLATES.get(level_key, RISK_TEMPLATES["medium"])

        summaries = {
            "low": (
                f"### {template.icon} 快速評估: {template.status}\n"
                f"**風險狀態**: {template.color} 低風險 ({risk_score:.1f}/100) | "
                f"**SF**: {safety_factor:.2f} - {sf_msg}\n"
                f"**建議**: 氣象良好，纜繩拖船配置充足，建議按計劃執行並保持標準監控。"
            ),
            "medium": (
                f"### {template.icon} 快速評估: {template.status}\n"
                f"**風險狀態**: {template.color} 中度風險 ({risk_score:.1f}/100) | "
                f"**SF**: {safety_factor:.2f} - {sf_msg}\n"
                f"**建議**: 需謹慎評估。建議加強在港監控（每小時巡檢），"
                f"準備備用纜繩，或微調作業時窗。"
            ),
            "high": (
                f"### {template.icon} 快速評估: {template.status}\n"
                f"**風險狀態**: {template.color} 高風險 ({risk_score:.1f}/100) | "
                f"**SF**: {safety_factor:.2f} - {sf_msg}\n"
                f"**警告**: 氣象條件不佳。**強烈建議延後作業**。"
                f"若必須執行，需強制增加拖船/纜繩並實施 24H 值班。"
            ),
            "extreme": (
                f"### {template.icon} 快速評估: {template.status}\n"
                f"**風險狀態**: {template.color} 極高風險 ({risk_score:.1f}/100) | "
                f"**SF**: {safety_factor:.2f} - {sf_msg}\n"
                f"**緊急**: 氣象惡劣，操作極度危險。**停止作業**，"
                f"立即評估替代港口或緊急離港應變。"
            ),
        }
        return summaries.get(level_key, summaries["medium"])

    def update_model_params(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """更新 AI 模型參數（僅更新有傳入的欄位）"""
        if model is not None:
            self.params.model = model
        if temperature is not None:
            self.params.temperature = temperature
        if max_tokens is not None:
            self.params.max_tokens = max_tokens
        if timeout is not None:
            self.params.timeout = timeout

    def set_api_key(self, api_key: str) -> None:
        """更新 API Key"""
        self.api_key = api_key
        if not api_key:
            logger.warning("API Key 已被清除，AI 分析功能將無法使用。")

    # ── Prompt 建構 ───────────────────────────────────────────

    def _build_user_prompt(
        self,
        port_name: str,
        vessel_params: VesselParams,
        analysis_results: AnalysisResults,
        berthing_time: datetime,
        departure_time: datetime,
        port_risk_level: int,
        hourly_data: Optional[List[HourlyRiskEntry]],
    ) -> str:
        duration_hrs = (departure_time - berthing_time).total_seconds() / 3600.0
        sf = analysis_results.safety_factor
        risk_desc = get_risk_description(analysis_results.risk_level, lang="zh")

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
受風面積: {vessel_params.area} m² | 風阻係數: {vessel_params.cd}
平均吃水: {vessel_params.avg_draft:.1f} m
【纜繩配置】
總纜繩數: {vessel_params.total_lines} 條
纜繩 MBL: {vessel_params.line_mbl} kN (SWL: {vessel_params.line_swl:.1f} kN)
總可用拉力: {analysis_results.mooring_capacity_total_kN:.1f} kN
【拖船支援】
數量: {vessel_params.num_tugs} 艘 ({vessel_params.tug_hp} HP)
總可用推力: {analysis_results.tug_capacity_total_kN:.1f} kN

═══════════════════════════════════════════════════════
⚠️ 【風險評估與氣象】
═══════════════════════════════════════════════════════
風險等級: {analysis_results.risk_level.upper()} ({risk_desc}) \
| 分數: {analysis_results.risk_score:.1f}/100
最大受風力: {analysis_results.max_wind_force_kN:.1f} kN
總抑制力 (纜繩+拖船): {analysis_results.total_restraint_kN:.1f} kN
**安全係數**: {sf:.2f} ({analysis_results.safety_factor_status})

最大陣風: {analysis_results.max_gust_kts:.1f} 節
主要風向: {analysis_results.dominant_wind_dir} \
(吹開風: {analysis_results.offshore_wind_ratio:.1f}%)
"""

        # 高風險時段警示（最多顯示 5 筆）
        if hourly_data:
            critical = [h for h in hourly_data if h.safety_factor < 1.5]
            if critical:
                prompt += f"\n🚨 【高風險時段警示】 (SF < 1.5): 共 {len(critical)} 小時\n"
                for entry in critical[:5]:
                    prompt += (
                        f"- {entry.time}: 風速 {entry.wind_gust_kts:.1f} kts, "
                        f"SF={entry.safety_factor:.2f}\n"
                    )

        prompt += """
═══════════════════════════════════════════════════════
🎯 【分析要求】
請提供具體建議：
1. **量化風險**: 哪些時段 SF 不足？吹開/吹攏風影響？
2. **操作建議**: 最佳靠/離泊時窗？若在高風險時段作業需何種補償措施？
3. **資源優化**: 纜繩與拖船需增加嗎？配置位置建議？
4. **應變計畫**: 風速超標應對與緊急離港觸發條件。
5. **決策建議**: 執行/延後/取消的具體判斷依據。
"""
        return prompt

    # ── 靜態輔助方法 ──────────────────────────────────────────

    @staticmethod
    def _load_api_key_from_secrets() -> str:
        """從 Streamlit Secrets 讀取 API Key，失敗時回傳空字串"""
        try:
            return st.secrets.get("PERPLEXITY_API_KEY", "")
        except Exception:
            return ""

    @staticmethod
    def _sf_label(sf: float) -> str:
        """將安全係數轉換為描述文字"""
        if sf >= 1.5:
            return "合格 (≥1.5)"
        if sf >= 1.2:
            return "邊緣 (1.2-1.5)"
        return "不足 (<1.2)"

    def _footer(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"\n\n---\n"
            f"<small>AI 分析生成時間: {timestamp} | "
            f"模型: {self.params.model} | "
            f"免責聲明: 僅供決策輔助，非操作指令。</small>"
        )

    @staticmethod
    def _msg_no_api_key() -> str:
        return (
            "❌ **未提供 Perplexity API Key**\n"
            "無法生成 AI 分析。請在 Streamlit Secrets 中設定 `PERPLEXITY_API_KEY`。"
        )

    @staticmethod
    def _msg_api_error(code: int, text: str) -> str:
        return f"❌ **AI 分析請求失敗**\n狀態碼: {code}\n錯誤: {text[:200]}"

    @staticmethod
    def _msg_unexpected_error(error: str) -> str:
        return f"❌ **系統發生錯誤**: {error[:200]}"


# ================= 便利函式與快取 =================

def create_ai_analyzer(api_key: Optional[str] = None) -> AIAnalyzer:
    """建立新的 AIAnalyzer 實例"""
    return AIAnalyzer(api_key=api_key)


@st.cache_resource
def get_cached_ai_analyzer(api_key: Optional[str] = None) -> AIAnalyzer:
    """建立並快取 AIAnalyzer 實例（Streamlit 跨 session 共用）"""
    return AIAnalyzer(api_key=api_key)
