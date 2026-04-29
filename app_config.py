# app_config.py
"""
應用程式配置與常數定義

使用方式：
    from app_config import PHYSICS, RISK, COMPASS, AppConfig

    # 讀取物理常數
    density = PHYSICS.air_density

    # 讀取 API Key（從環境變數）
    key = AppConfig.perplexity_api_key()
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── 嘗試載入 .env（開發環境用，生產環境直接設定系統環境變數）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 未安裝時靜默跳過


# ================= 物理常數 =================

@dataclass(frozen=True)
class PhysicsConstants:
    """基礎物理常數（不可變）"""
    air_density: float = 1.225      # kg/m³，空氣密度
    gravity: float = 9.81           # m/s²，重力加速度
    knots_to_ms: float = 0.514444   # 節 → 公尺/秒轉換係數

    def ms_to_knots(self, ms: float) -> float:
        return ms / self.knots_to_ms

    def knots_to_ms_val(self, kts: float) -> float:
        return kts * self.knots_to_ms


PHYSICS = PhysicsConstants()


# ================= 安全係數與拖船參數 =================

@dataclass(frozen=True)
class MooringConstants:
    """繫泊與拖船相關常數"""
    fixed_safety_factor: float = 0.33   # 纜繩安全係數：WLL = MBL × SF
    hp_to_bp_factor: float = 0.75       # 馬力 → 推力轉換係數
    tug_utilization: float = 0.85       # 拖船利用率

    def wll_from_mbl(self, mbl: float) -> float:
        """由 MBL 計算工作負荷上限 (WLL)"""
        return mbl * self.fixed_safety_factor

    def tug_bollard_pull(self, hp: float) -> float:
        """由馬力估算拖船推力 (ton)"""
        return (hp / 100.0) * self.hp_to_bp_factor * self.tug_utilization


MOORING = MooringConstants()


# ================= 風險閾值 =================

@dataclass(frozen=True)
class WindThreshold:
    """單一風速指標的四級閾值 (kts)"""
    low: float      # < low      → 低風險
    medium: float   # low–medium → 中風險
    high: float     # medium–high→ 高風險
                    # > high     → 極高風險

    def classify(self, speed: float) -> str:
        """將風速分類為風險等級字串"""
        if speed < self.low:
            return "low"
        if speed < self.medium:
            return "medium"
        if speed < self.high:
            return "high"
        return "extreme"


@dataclass(frozen=True)
class RiskThresholds:
    """所有氣象風險閾值"""
    wind: WindThreshold = field(default_factory=lambda: WindThreshold(25, 35, 45))
    gust: WindThreshold = field(default_factory=lambda: WindThreshold(35, 45, 55))


THRESHOLDS = RiskThresholds()


# ================= 風險等級定義 =================

@dataclass(frozen=True)
class RiskLevelSpec:
    """單一風險等級的完整規格"""
    name_zh: str        # 中文名稱
    score_min: int      # 分數下限（含）
    score_max: int      # 分數上限（不含）
    color_hex: str      # 圖表用色（深色）
    color_bg: str       # 背景用色（淺色）
    label: str          # UI 標籤（含 emoji）


# 風險等級總表（single source of truth）
RISK_LEVEL_SPECS: Dict[str, RiskLevelSpec] = {
    "low": RiskLevelSpec(
        name_zh="低度風險",
        score_min=0,   score_max=25,
        color_hex="#2ecc71", color_bg="#e8f5e9",
        label="🟢 低",
    ),
    "medium": RiskLevelSpec(
        name_zh="中度風險",
        score_min=25,  score_max=50,
        color_hex="#f1c40f", color_bg="#fff9c4",
        label="🟡 中",
    ),
    "high": RiskLevelSpec(
        name_zh="高度風險",
        score_min=50,  score_max=75,
        color_hex="#e67e22", color_bg="#ffe0b2",
        label="🟠 高",
    ),
    "extreme": RiskLevelSpec(
        name_zh="極度危險",
        score_min=75,  score_max=101,
        color_hex="#e74c3c", color_bg="#ffcdd2",
        label="🔴 極高",
    ),
}


def score_to_risk_level(score: float) -> str:
    """將風險分數 (0–100) 轉換為風險等級 key"""
    for key, spec in RISK_LEVEL_SPECS.items():
        if spec.score_min <= score < spec.score_max:
            return key
    return "extreme"


def risk_level_to_zh(level_key: str) -> str:
    """英文 key → 中文名稱（找不到時回傳原始值）"""
    spec = RISK_LEVEL_SPECS.get(level_key)
    return spec.name_zh if spec else level_key


# ── 向下相容的平面字典（供舊程式碼使用）────────────────────────
RISK_LEVELS: Dict[str, Tuple[int, int]] = {
    spec.name_zh: (spec.score_min, spec.score_max)
    for spec in RISK_LEVEL_SPECS.values()
}
RISK_COLORS: Dict[str, str] = {
    spec.name_zh: spec.color_hex
    for spec in RISK_LEVEL_SPECS.values()
}
RISK_LEVEL_COLORS: Dict[str, str] = {
    key: spec.color_bg
    for key, spec in RISK_LEVEL_SPECS.items()
}
RISK_LEVEL_LABELS: Dict[str, str] = {
    key: spec.label
    for key, spec in RISK_LEVEL_SPECS.items()
}


# ================= 羅盤方向 =================

@dataclass(frozen=True)
class CompassConfig:
    """羅盤方向對應度數"""
    directions: Dict[str, float] = field(default_factory=lambda: {
        "N":   0.0,   "NNE": 22.5,  "NE":  45.0,  "ENE": 67.5,
        "E":   90.0,  "ESE": 112.5, "SE":  135.0,  "SSE": 157.5,
        "S":   180.0, "SSW": 202.5, "SW":  225.0,  "WSW": 247.5,
        "W":   270.0, "WNW": 292.5, "NW":  315.0,  "NNW": 337.5,
    })

    def to_degrees(self, compass: str) -> Optional[float]:
        """羅盤方位字串 → 度數，找不到回傳 None"""
        return self.directions.get(compass.upper().strip())

    def to_degrees_safe(self, compass: str, default: float = 0.0) -> float:
        """羅盤方位字串 → 度數，找不到回傳 default"""
        return self.directions.get(compass.upper().strip(), default)


COMPASS = CompassConfig()

# 向下相容的平面字典
COMPASS_DIRECTIONS: Dict[str, float] = dict(COMPASS.directions)


# ================= 蒲福風級 =================

@dataclass(frozen=True)
class BeaufortScale:
    """蒲福風級閾值（m/s）"""
    # 每個元素代表該級別的下限風速（m/s）
    thresholds: Tuple[float, ...] = (
        0.3, 1.6, 3.4, 5.5, 8.0, 10.8,
        13.9, 17.2, 20.8, 24.5, 28.5, 32.7,
    )

    def from_ms(self, wind_ms: float) -> int:
        """由風速 (m/s) 取得蒲福風級"""
        for scale, threshold in enumerate(reversed(self.thresholds)):
            if wind_ms >= threshold:
                return len(self.thresholds) - scale
        return 0

    def from_knots(self, wind_kts: float) -> int:
        """由風速 (kts) 取得蒲福風級"""
        return self.from_ms(wind_kts * PHYSICS.knots_to_ms)


BEAUFORT = BeaufortScale()

# 向下相容的平面列表
BEAUFORT_THRESHOLDS: List[float] = list(BEAUFORT.thresholds)


# ================= 欄位映射 =================

# 將各種輸入欄位名稱統一映射為內部標準名稱
FIELD_MAPPING: Dict[str, str] = {
    # ── 原有欄位（不變）──────────────────────────────────────
    "time":           "time",
    "datetime":       "time",
    "wind_speed":     "wind_speed_kts",
    "wind_speed_kts": "wind_speed_kts",
    "wind_gust":      "wind_gust_kts",
    "wind_gust_kts":  "wind_gust_kts",
    "wind_direction": "wind_dir",
    "wind_dir":       "wind_dir",
    "wind_dir_deg":   "wind_dir_deg",
    "wave_height":    "wave_sig_m",
    "wave_sig_m":     "wave_sig_m",
    "wave_max":       "wave_max_m",
    "wave_max_m":     "wave_max_m",
    "wave_period":    "wave_period_s",
    "wave_period_s":  "wave_period_s",
    # ── 新增：AWT API 特有欄位映射 ───────────────────────────
    # AwtWeatherRecord.to_dict() 輸出的欄位，供 normalize_dataframe 識別
    "visibility_nm":        "visibility_nm",
    "pilot_visibility_nm":  "pilot_visibility_nm",
    "berthing_risk_score":  "berthing_risk_score",
    "berthing_risk_label":  "berthing_risk_label",
    "current_speed_ms":     "current_speed_ms",
    "current_dir_deg":      "current_dir_deg",
    "relative_humidity":    "relative_humidity",
    "air_temp_c":           "air_temp_c",
}


# ================= API 設定 =================

class AppConfig:
    """
    應用程式動態設定（從環境變數讀取）。
    所有敏感資訊皆不寫死在程式碼中。
    """

    @staticmethod
    def perplexity_api_key() -> str:
        """
        讀取 Perplexity API Key。
        來源優先順序：環境變數 PERPLEXITY_API_KEY → Streamlit Secrets → 空字串

        若回傳空字串，呼叫端應顯示適當的錯誤提示。
        """
        # 1. 環境變數
        key = os.getenv("PERPLEXITY_API_KEY", "")
        if key:
            return key

        # 2. Streamlit Secrets（僅在 Streamlit 環境中有效）
        try:
            import streamlit as st
            key = st.secrets.get("PERPLEXITY_API_KEY", "")
            if key:
                return key
        except Exception:
            pass

        logger.warning(
            "找不到 PERPLEXITY_API_KEY。"
            "請在 .env 或 Streamlit Secrets 中設定此變數。"
        )
        return ""

    @staticmethod
    def validate() -> bool:
        ok = True
        if not AppConfig.perplexity_api_key():
            logger.warning("❌ 缺少 PERPLEXITY_API_KEY")
            ok = False
        # 新增：AWT 憑證檢查
        if not os.getenv("AWT_USERNAME"):
            logger.warning("❌ 缺少 AWT_USERNAME")
            ok = False
        if not os.getenv("AWT_PASSWORD"):
            logger.warning("❌ 缺少 AWT_PASSWORD")
            ok = False
        return ok
    @staticmethod
    def awt_api_base() -> str:
        """讀取 AWT API Base URL，預設為 StormGeo 正式環境"""
        return os.getenv(
            "AWT_API_BASE_URL",
            "https://api-shipping.stormgeo.com/api/v1",
        )
