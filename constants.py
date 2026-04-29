# constants.py
"""
船舶靠泊繫留風險評估系統 — 領域專用常數

設計原則：
  - 物理基礎常數（空氣密度、重力等）統一由 app_config.PHYSICS 提供
  - 本模組僅定義「繫泊分析領域」專用的閾值、權重與效率係數
  - 所有常數封裝為 frozen dataclass，防止意外修改
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

# ── 從 app_config 引入共用基礎常數（避免重複定義）────────────────
from app_config import (
    BEAUFORT,
    BEAUFORT_THRESHOLDS,
    COMPASS,
    COMPASS_DIRECTIONS,
    MOORING,
    PHYSICS,
)

logger = logging.getLogger(__name__)

# ── 向下相容別名（讓舊程式碼不需修改 import）─────────────────────
AIR_DENSITY         = PHYSICS.air_density
KTS_TO_MS           = PHYSICS.knots_to_ms
FIXED_SAFETY_FACTOR = MOORING.fixed_safety_factor
TUG_UTILIZATION     = MOORING.tug_utilization
COMPASS_MAP         = COMPASS_DIRECTIONS
BEAUFORT_THRESH     = BEAUFORT_THRESHOLDS


# ================= 氣象閾值 =================

@dataclass(frozen=True)
class WeatherConditionThresholds:
    """
    氣象條件門檻值。

    分為兩組：
    - 高風險觸發門檻（用於警示）
    - 平靜窗口判斷門檻（用於尋找最佳靠/離泊時窗）
    """
    # ── 高風險觸發門檻 ──
    high_wind_speed_kts:  float = 20.0   # 持續風速警戒值 (kts)
    high_gust_speed_kts:  float = 35.0   # 陣風警戒值 (kts)
    high_wave_sig_m:      float = 2.5    # 顯著波高警戒值 (m)
    very_high_wave_sig_m: float = 3.5    # 極高波高警戒值 (m)

    # ── 平靜窗口判斷門檻 ──
    calmer_wind_max_kts:  float = 18.0   # 平靜窗口最大風速 (kts)
    calmer_wave_max_m:    float = 1.5    # 平靜窗口最大波高 (m)
    calmer_window_hours:  int   = 2      # 平靜窗口最短持續時數
    calmer_search_hours:  int   = 12     # 向前搜尋的最大時數

    # ── 夜間定義 ──
    night_start_hour: int = 20           # 夜間開始（含）
    night_end_hour:   int = 6            # 夜間結束（不含）

    def is_night(self, hour: int) -> bool:
        """判斷給定小時是否為夜間"""
        s, e = self.night_start_hour, self.night_end_hour
        return hour >= s or hour < e if s > e else s <= hour < e

    def is_high_risk(self, wind_kts: float, gust_kts: float, wave_m: float) -> bool:
        """判斷氣象條件是否達到高風險門檻"""
        return (
            wind_kts >= self.high_wind_speed_kts
            or gust_kts >= self.high_gust_speed_kts
            or wave_m  >= self.high_wave_sig_m
        )

    def is_calmer_window(self, wind_kts: float, wave_m: float) -> bool:
        """判斷氣象條件是否符合平靜窗口標準"""
        return (
            wind_kts <= self.calmer_wind_max_kts
            and wave_m <= self.calmer_wave_max_m
        )


WEATHER = WeatherConditionThresholds()

# ── 向下相容的模組層級常數 ────────────────────────────────────────
HIGH_WIND_SPEED     = WEATHER.high_wind_speed_kts
HIGH_GUST_SPEED     = WEATHER.high_gust_speed_kts
HIGH_WAVE_SIG       = WEATHER.high_wave_sig_m
VERY_HIGH_WAVE_SIG  = WEATHER.very_high_wave_sig_m
CALMER_WIND_MAX     = WEATHER.calmer_wind_max_kts
CALMER_WAVE_MAX     = WEATHER.calmer_wave_max_m
CALMER_WINDOW_HOURS = WEATHER.calmer_window_hours
CALMER_SEARCH_HOURS = WEATHER.calmer_search_hours
NIGHT_HOURS: Tuple[int, int] = (WEATHER.night_start_hour, WEATHER.night_end_hour)


# ================= 風險權重 =================

@dataclass(frozen=True)
class RiskWeights:
    """
    風險分數各項權重。

    ⚠️ 所有權重總和必須為 1.0。
    原始版本總和為 1.20，已修正。
    """
    wind_mean:    float = 0.15
    gust:         float = 0.30
    wave_sig:     float = 0.10
    wave_max:     float = 0.10
    persist_wind: float = 0.15
    persist_wave: float = 0.10
    direction:    float = 0.05
    night:        float = 0.05

    def __post_init__(self):
        total = sum([
            self.wind_mean, self.gust, self.wave_sig, self.wave_max,
            self.persist_wind, self.persist_wave,
            self.direction, self.night,
        ])
        if abs(total - 1.0) > 1e-6:
            logger.warning(
                "RiskWeights 總和為 %.4f，應為 1.0。請檢查權重設定。", total
            )

    def as_dict(self) -> Dict[str, float]:
        """回傳字典格式（供舊程式碼使用）"""
        return {
            "wind_mean":    self.wind_mean,
            "gust":         self.gust,
            "wave_sig":     self.wave_sig,
            "wave_max":     self.wave_max,
            "persist_wind": self.persist_wind,
            "persist_wave": self.persist_wave,
            "direction":    self.direction,
            "night":        self.night,
        }


WEIGHTS = RiskWeights()

# ── 向下相容的平面字典 ────────────────────────────────────────────
RISK_WEIGHTS: Dict[str, float] = WEIGHTS.as_dict()


# ================= 繫泊效率係數 =================

@dataclass(frozen=True)
class MooringEfficiencyFactors:
    """
    纜繩方向效率係數。

    說明：
    - head（頭纜/尾纜）：與船身垂直，橫向效率高
    - spring（倒纜）：與船身平行，縱向效率高
    """
    head_transverse:      float = 1.0   # 頭纜橫向效率
    head_longitudinal:    float = 0.3   # 頭纜縱向效率
    spring_transverse:    float = 0.4   # 倒纜橫向效率
    spring_longitudinal:  float = 1.0   # 倒纜縱向效率
    bow_force_share:      float = 0.5   # 船首分擔風力比例
    max_add_iterations:   int   = 6     # 最大補充纜繩迭代次數

    def transverse_capacity(
        self,
        head_count: int,
        spring_count: int,
        wll_per_line: float,
    ) -> float:
        """計算總橫向抗力 (與 wll_per_line 同單位)"""
        return (
            head_count   * wll_per_line * self.head_transverse
            + spring_count * wll_per_line * self.spring_transverse
        )

    def longitudinal_capacity(
        self,
        head_count: int,
        spring_count: int,
        wll_per_line: float,
    ) -> float:
        """計算總縱向抗力 (與 wll_per_line 同單位)"""
        return (
            head_count   * wll_per_line * self.head_longitudinal
            + spring_count * wll_per_line * self.spring_longitudinal
        )


MOORING_EFF = MooringEfficiencyFactors()

# ── 向下相容的模組層級常數 ────────────────────────────────────────
HEAD_TRANS_EFF   = MOORING_EFF.head_transverse
HEAD_LONG_EFF    = MOORING_EFF.head_longitudinal
SPRING_TRANS_EFF = MOORING_EFF.spring_transverse
SPRING_LONG_EFF  = MOORING_EFF.spring_longitudinal
BOW_FORCE_SHARE  = MOORING_EFF.bow_force_share
MAX_ADD_ITER     = MOORING_EFF.max_add_iterations


# ================= 物理轉換常數 =================

@dataclass(frozen=True)
class PhysicsConversionFactors:
    """
    繫泊計算專用的物理轉換係數。
    （基礎物理常數請使用 app_config.PHYSICS）
    """
    bp_ton_to_newton:         float = 9810.0  # 推力噸 → 牛頓
    avg_wind_force_coeff:     float = 1.00    # 平均風力係數（保守值）
    hp_to_bp_factor_internal: float = 0.09   # 馬力 → 推力（內部估算用）

    def hp_to_bollard_pull_kN(self, hp: float) -> float:
        """由馬力估算拖船推力 (kN)，內部簡易估算版"""
        return hp * self.hp_to_bp_factor_internal


CONVERSION = PhysicsConversionFactors()

# ── 向下相容的模組層級常數 ────────────────────────────────────────
BP_TON_TO_NEWTON         = CONVERSION.bp_ton_to_newton
AVG_WIND_FORCE_COEFF     = CONVERSION.avg_wind_force_coeff
HP_TO_BP_FACTOR_INTERNAL = CONVERSION.hp_to_bp_factor_internal


# ================= awt_parser.py 相容別名 =================
# awt_parser.py 使用的所有額外常數，統一在此定義一次。
# WEATHER 與 BEAUFORT 在本檔案上方已定義/匯入，直接使用即可。

# ── 風速別名（帶 _kts 後綴 + Beaufort 換算）────────────────────────
HIGH_WIND_SPEED_kts = WEATHER.high_wind_speed_kts           # 20.0 kts
HIGH_WIND_SPEED_Bft = BEAUFORT.from_knots(HIGH_WIND_SPEED_kts)

HIGH_GUST_SPEED_kts = WEATHER.high_gust_speed_kts           # 35.0 kts
HIGH_GUST_SPEED_Bft = BEAUFORT.from_knots(HIGH_GUST_SPEED_kts)

# ── 靠泊風險評分門檻 ─────────────────────────────────────────────
BERTHING_SCORE_HIGH   = 60    # 高風險門檻分數
BERTHING_SCORE_MEDIUM = 35    # 中風險門檻分數
BERTHING_SCORE_LOW    = 15    # 低風險門檻分數

# ── 陣風係數門檻 ─────────────────────────────────────────────────
GUST_FACTOR_EXTREME  = 2.0    # 極不穩定
GUST_FACTOR_HIGH     = 1.7    # 不穩定
GUST_FACTOR_MODERATE = 1.4    # 略不穩定

# ── 流速門檻 m/s ─────────────────────────────────────────────────
CURRENT_STRONG   = 1.0    # 強流
CURRENT_MODERATE = 0.6    # 中強流
CURRENT_MILD     = 0.3    # 中流

# ── 海霧判斷門檻 ─────────────────────────────────────────────────
FOG_HUMIDITY_THRESHOLD = 85.0   # 高濕度門檻 (%)
FOG_TEMP_DIFF_HIGH     = 5.0    # 氣溫 - 海溫差：極高霧風險 (°C)
FOG_TEMP_DIFF_MEDIUM   = 3.0    # 中高霧風險 (°C)
FOG_TEMP_DIFF_LOW      = 1.5    # 中霧風險 (°C)
FOG_RADIATION_HUMIDITY = 90.0   # 輻射霧濕度門檻 (%)
FOG_RADIATION_TEMP     = 10.0   # 輻射霧氣溫門檻 (°C)

# ── 港區 vs 引水點差異警示門檻 ───────────────────────────────────
DIVERGE_WIND_KTS = 10.0   # 風速差警示門檻 (kts)
DIVERGE_VIS_NM   = 2.0    # 能見度差警示門檻 (NM)
DIVERGE_WAVE_M   = 0.5    # 浪高差警示門檻 (m)
DIVERGE_CURR_MS  = 0.3    # 流速差警示門檻 (m/s)

# ── 浪陡度門檻 ───────────────────────────────────────────────────
WAVE_STEEPNESS_DANGER  = 0.142   # 陡浪危險門檻（理論破碎極限 1/7）
WAVE_STEEPNESS_CAUTION = 0.05    # 較陡警示門檻
def kts_to_bft(knots: float) -> int:
    """將風速（節）轉換為蒲福風級，委派給 BEAUFORT.from_knots"""
    return BEAUFORT.from_knots(float(knots))