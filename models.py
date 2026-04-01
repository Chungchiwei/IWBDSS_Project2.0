# models.py
"""
船舶靠泊繫留風險評估系統 — 核心資料模型

設計原則：
  - 所有轉換係數引用 app_config.PHYSICS，不硬寫數值
  - 羅盤轉換引用 app_config.COMPASS，不重複定義
  - AnalysisResult 完全型別化，不使用裸露 dict
  - __post_init__ 同時負責型別轉換與業務邏輯驗證
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from app_config import COMPASS, PHYSICS

logger = logging.getLogger(__name__)


# ================= 氣象記錄 =================

@dataclass
class WeatherRecord:
    """
    單筆氣象觀測記錄。

    原始欄位以 knots / meters / seconds 儲存，
    衍生單位透過 property 提供，不重複儲存。
    """
    time:           datetime
    wind_direction: str     # 羅盤方位，例如 'NNE'
    wind_speed:     float   # 持續風速 (kts)
    wind_gust:      float   # 陣風風速 (kts)
    wave_direction: str     # 浪向羅盤方位
    wave_height:    float   # 顯著浪高 (m)
    wave_max:       float   # 最大浪高 (m)
    wave_period:    float   # 波浪週期 (s)

    def __post_init__(self) -> None:
        # ── 型別強制轉換 ──
        self.wind_speed    = float(self.wind_speed)
        self.wind_gust     = float(self.wind_gust)
        self.wave_height   = float(self.wave_height)
        self.wave_max      = float(self.wave_max)
        self.wave_period   = float(self.wave_period)
        self.wind_direction = str(self.wind_direction).strip().upper()
        self.wave_direction = str(self.wave_direction).strip().upper()

        # ── 業務邏輯驗證 ──
        if self.wind_speed < 0:
            raise ValueError(f"wind_speed 不可為負數：{self.wind_speed}")
        if self.wind_gust < self.wind_speed:
            logger.warning(
                "wind_gust (%.1f kts) 小於 wind_speed (%.1f kts)，資料可能有誤。",
                self.wind_gust, self.wind_speed,
            )
        if self.wave_height < 0:
            raise ValueError(f"wave_height 不可為負數：{self.wave_height}")
        if self.wave_max < self.wave_height:
            logger.warning(
                "wave_max (%.1f m) 小於 wave_height (%.1f m)，資料可能有誤。",
                self.wave_max, self.wave_height,
            )

    # ── 風速換算 ──────────────────────────────────────────────

    @property
    def wind_speed_kts(self) -> float:
        """持續風速 (kts)，與原始欄位相同，提供語意明確的別名"""
        return self.wind_speed

    @property
    def wind_gust_kts(self) -> float:
        """陣風風速 (kts)"""
        return self.wind_gust

    @property
    def wind_speed_ms(self) -> float:
        """持續風速 (m/s)"""
        return self.wind_speed * PHYSICS.knots_to_ms

    @property
    def wind_gust_ms(self) -> float:
        """陣風風速 (m/s)"""
        return self.wind_gust * PHYSICS.knots_to_ms

    # ── 風向換算 ──────────────────────────────────────────────

    @property
    def wind_dir_deg(self) -> float:
        """風向度數 (0–360)，找不到對應時回傳 0.0"""
        return COMPASS.to_degrees_safe(self.wind_direction)

    @property
    def wave_dir_deg(self) -> float:
        """浪向度數 (0–360)，找不到對應時回傳 0.0"""
        return COMPASS.to_degrees_safe(self.wave_direction)

    # ── 浪高別名 ──────────────────────────────────────────────

    @property
    def wave_sig_m(self) -> float:
        """顯著浪高 (m)，wave_height 的語意別名"""
        return self.wave_height

    @property
    def wave_max_m(self) -> float:
        """最大浪高 (m)，wave_max 的語意別名"""
        return self.wave_max

    @property
    def wave_period_s(self) -> float:
        """波浪週期 (s)，wave_period 的語意別名"""
        return self.wave_period

    def __repr__(self) -> str:
        return (
            f"WeatherRecord({self.time:%Y-%m-%d %H:%M} "
            f"Wind={self.wind_speed:.1f}/{self.wind_gust:.1f}kts@{self.wind_direction} "
            f"Wave={self.wave_height:.1f}m)"
        )


# ================= 天氣狀況記錄 =================

@dataclass
class ConditionRecord:
    """
    天氣狀況記錄（溫度、能見度、降雨、天氣代碼）。

    從 WNI 資料的 2. WEATHER 區段解析。
    """
    time:          datetime
    temperature:   Optional[float]  # °C，可能為 None
    precipitation: float            # mm/h
    visibility:    str              # 原始字串，e.g. '10km<', '500', '<1km'
    weather_code:  str              # 'CLR'|'FOG'|'MIST'|'RAIN'|'SNOW'|'THUNDER'|...

    _WEATHER_ZH: Dict[str, str] = field(
        default_factory=lambda: {
            'CLR': '晴朗', 'CLOUDY': '多雲', 'OVERCAST': '陰天',
            'FOG': '霧', 'MIST': '薄霧', 'HAZE': '霾',
            'RAIN': '雨', 'DRIZZLE': '毛毛雨', 'THUNDER': '雷暴',
            'SNOW': '雪', 'SLEET': '雨夾雪', 'N/A': '無資料',
        },
        init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        if self.temperature is not None:
            try:
                v = float(str(self.temperature).replace('*', '').strip())
                self.temperature = v if -100 < v < 100 else None
            except (ValueError, TypeError):
                self.temperature = None
        self.precipitation = float(self.precipitation) if self.precipitation else 0.0
        self.visibility    = str(self.visibility).strip()
        self.weather_code  = str(self.weather_code).strip().upper()

    @property
    def visibility_m(self) -> Optional[float]:
        """將能見度字串解析為公尺（10km< → 10000，500 → 500）"""
        vis = self.visibility.replace('<', '').replace('>', '').strip()
        if not vis:
            return None
        m = re.search(r'([\d.]+)\s*km', vis, re.IGNORECASE)
        if m:
            return float(m.group(1)) * 1000.0
        m = re.search(r'([\d.]+)', vis)
        if m:
            return float(m.group(1))
        return None

    @property
    def weather_zh(self) -> str:
        return self._WEATHER_ZH.get(self.weather_code, self.weather_code)

    # ── 風險旗標 ──────────────────────────────────────────────

    @property
    def is_fog(self) -> bool:
        return self.weather_code in ('FOG', 'MIST')

    @property
    def is_low_visibility(self) -> bool:
        """能見度 < 1 km"""
        vm = self.visibility_m
        return vm is not None and vm < 1000.0

    @property
    def is_mod_visibility(self) -> bool:
        """能見度 1–3 km（moderate）"""
        vm = self.visibility_m
        return vm is not None and 1000.0 <= vm < 3000.0

    @property
    def is_low_temp(self) -> bool:
        """溫度 < 5 °C"""
        return self.temperature is not None and self.temperature < 5.0

    @property
    def is_heavy_precip(self) -> bool:
        """強降雨 ≥ 10 mm/h"""
        return self.precipitation >= 10.0

    @property
    def is_thunder(self) -> bool:
        return self.weather_code == 'THUNDER'


# ================= 船舶資訊 =================

@dataclass
class VesselInfo:
    """
    船舶與靠泊作業資訊。

    所有數值欄位在 __post_init__ 中強制轉型並驗證合理範圍。
    """
    berth_direction:    float     # 泊位方向 (度, 0–360)
    berthing_side:      str       # 靠泊側：'port' | 'starboard'
    arrival_time:       datetime  # 靠港時間
    departure_time:     datetime  # 離港時間
    stay_duration:      str       # 停留時間描述
    draft_bow:          float     # 船艏吃水 (m)
    draft_stern:        float     # 船艉吃水 (m)
    mbl:                float     # 纜繩破斷負荷 MBL (N)
    total_lines:        int       # 纜繩總數
    safety_factor:      float     # 安全係數
    tug_hp:             float     # 拖船馬力 (hp)
    bow_lines:          int       # 船艏頭纜數
    bow_spring_lines:   int       # 船艏倒纜數
    stern_lines:        int       # 船艉尾纜數
    stern_spring_lines: int       # 船艉倒纜數
    wind_area:          float     # 受風面積 (m²)
    tug_count:          int = 2   # 拖船數量（預設 2 艘）
    wind_drag_coef:     float = 1.0  # 風阻係數（預設保守值）

    def __post_init__(self) -> None:
        # ── 型別強制轉換 ──
        self.berth_direction    = float(self.berth_direction) % 360
        self.berthing_side      = str(self.berthing_side).strip().lower()
        self.draft_bow          = float(self.draft_bow)
        self.draft_stern        = float(self.draft_stern)
        self.mbl                = float(self.mbl)
        self.total_lines        = int(self.total_lines)
        self.safety_factor      = float(self.safety_factor)
        self.tug_hp             = float(self.tug_hp)
        self.bow_lines          = int(self.bow_lines)
        self.bow_spring_lines   = int(self.bow_spring_lines)
        self.stern_lines        = int(self.stern_lines)
        self.stern_spring_lines = int(self.stern_spring_lines)
        self.wind_area          = float(self.wind_area)
        self.tug_count          = int(self.tug_count)
        self.wind_drag_coef     = float(self.wind_drag_coef)

        # ── 業務邏輯驗證 ──
        errors: List[str] = []

        if self.draft_bow <= 0:
            errors.append(f"draft_bow 必須大於 0，實際：{self.draft_bow}")
        if self.draft_stern <= 0:
            errors.append(f"draft_stern 必須大於 0，實際：{self.draft_stern}")
        if self.mbl <= 0:
            errors.append(f"mbl 必須大於 0，實際：{self.mbl}")
        if self.safety_factor <= 0:
            errors.append(f"safety_factor 必須大於 0，實際：{self.safety_factor}")
        if self.wind_area <= 0:
            errors.append(f"wind_area 必須大於 0，實際：{self.wind_area}")
        if any(n < 0 for n in [self.bow_lines, self.bow_spring_lines,
                                self.stern_lines, self.stern_spring_lines]):
            errors.append("纜繩數量不可為負數")
        if self.arrival_time >= self.departure_time:
            errors.append(
                f"arrival_time ({self.arrival_time}) 必須早於 "
                f"departure_time ({self.departure_time})"
            )
        if self.berthing_side not in ("port", "starboard", "左", "右"):
            logger.warning("berthing_side 值 '%s' 不在標準清單中", self.berthing_side)

        if errors:
            raise ValueError("VesselInfo 驗證失敗：\n" + "\n".join(f"  - {e}" for e in errors))

    @property
    def mean_draft(self) -> float:
        """平均吃水 (m)"""
        return (self.draft_bow + self.draft_stern) / 2.0

    @property
    def total_mooring_lines(self) -> int:
        """實際纜繩總數（由各分組加總）"""
        return (
            self.bow_lines + self.bow_spring_lines
            + self.stern_lines + self.stern_spring_lines
        )

    def __repr__(self) -> str:
        return (
            f"VesselInfo(side={self.berthing_side} "
            f"berth={self.berth_direction:.0f}° "
            f"lines={self.total_mooring_lines} "
            f"area={self.wind_area:.0f}m²)"
        )


# ================= 繫泊狀態 =================

@dataclass
class MooringStatus:
    """單端（船艏或船艉）繫泊狀態"""
    status:              str    # 'OK' | 'WARNING' | 'CRITICAL'
    current_lines:       int    # 當前纜繩數
    required_lines:      int    # 建議纜繩數
    utilization:         float  # 利用率 (%)
    recommendation_text: str    # 建議文字

    @property
    def is_adequate(self) -> bool:
        """纜繩數量是否充足"""
        return self.current_lines >= self.required_lines

    @property
    def additional_lines_needed(self) -> int:
        """需補充的纜繩數（已足夠時回傳 0）"""
        return max(0, self.required_lines - self.current_lines)


@dataclass
class MooringSplit:
    """船艏 / 船艉繫泊分配"""
    bow:   MooringStatus
    stern: MooringStatus

    @property
    def overall_status(self) -> str:
        """
        取兩端中較嚴重的狀態。
        優先順序：CRITICAL > WARNING > OK
        """
        _ORDER = {"OK": 0, "WARNING": 1, "CRITICAL": 2}
        bow_val   = _ORDER.get(self.bow.status,   0)
        stern_val = _ORDER.get(self.stern.status, 0)
        if bow_val >= stern_val:
            return self.bow.status
        return self.stern.status


# ================= 拖船建議 =================

@dataclass
class TugRecommendation:
    """拖船配置建議"""
    base_tug_count:      int        # 原始拖船數
    final_tug_count:     int        # 建議最終拖船數
    adequacy:            bool       # 現有配置是否足夠
    enforcement_reasons: List[str] = field(default_factory=list)  # 需增加的原因

    @property
    def additional_tugs(self) -> int:
        """需增加的拖船數"""
        return max(0, self.final_tug_count - self.base_tug_count)

    def to_dict(self) -> Dict[str, object]:
        """轉為字典（向下相容）"""
        return {
            "base_tug_count":      self.base_tug_count,
            "final_tug_count":     self.final_tug_count,
            "adequacy":            self.adequacy,
            "enforcement_reasons": self.enforcement_reasons,
        }


# ================= 風力摘要 =================

@dataclass
class WindForceSummary:
    """最惡劣條件下的風力計算摘要"""
    max_gust_force_N:    float           # 最大陣風力 (N)
    max_trans_force_N:   float           # 最大橫向力 (N)
    max_long_force_N:    float           # 最大縱向力 (N)
    safety_factor:       float           # 實際安全係數
    wind_type:           str             # 'offshore' | 'onshore' | 'parallel'
    mooring_capacity_N:  float           # 纜繩總抗力 (N)
    total_restraint_kN:  float           # 總抗力（含拖船）(kN)
    required_force_kN:   float           # 所需抗力 (kN)
    max_gust_record:     Dict[str, object] = field(default_factory=dict)

    @property
    def safety_margin_kN(self) -> float:
        """安全餘裕 (kN)，正值表示充足"""
        return self.total_restraint_kN - self.required_force_kN

    @property
    def is_structurally_safe(self) -> bool:
        """安全係數是否 ≥ 1.0（最低物理安全要求）"""
        return self.safety_factor >= 1.0

    def to_dict(self) -> Dict[str, object]:
        """轉為字典（向下相容）"""
        return {
            "max_gust_force_N":   self.max_gust_force_N,
            "max_trans_force_N":  self.max_trans_force_N,
            "max_long_force_N":   self.max_long_force_N,
            "safety_factor":      self.safety_factor,
            "wind_type":          self.wind_type,
            "mooring_capacity_N": self.mooring_capacity_N,
            "total_restraint_kN": self.total_restraint_kN,
            "required_force_kN":  self.required_force_kN,
            "max_gust_record":    self.max_gust_record,
        }


# ================= 分析結果 =================

@dataclass
class AnalysisResult:
    """
    完整靠泊風險分析結果。

    所有子結構均為型別化 dataclass，不使用裸露 dict。
    提供 to_dict() 方法供向下相容。
    """
    risk_score:           float             # 風險分數 (0–100)
    risk_level:           str               # 'low' | 'medium' | 'high' | 'extreme'
    mitigated_risk_score: float             # 採取緩解措施後的預估分數
    mooring_split:        MooringSplit      # 船艏/船艉繫泊狀態
    tug_recommendation:   TugRecommendation # 拖船建議
    wind_force_summary:   WindForceSummary  # 風力摘要
    recommendations:      List[str] = field(default_factory=list)  # 建議文字清單
    arr_window_result:    Optional[object] = None  # 靠泊時窗詳細結果
    dep_window_result:    Optional[object] = None  # 離泊時窗詳細結果

    def __post_init__(self) -> None:
        if not (0.0 <= self.risk_score <= 100.0):
            logger.warning("risk_score 超出範圍 [0, 100]：%.2f", self.risk_score)
        if self.mitigated_risk_score > self.risk_score:
            logger.warning(
                "mitigated_risk_score (%.2f) 大於 risk_score (%.2f)，請確認計算邏輯。",
                self.mitigated_risk_score, self.risk_score,
            )

    @property
    def risk_level_zh(self) -> str:
        """風險等級中文名稱"""
        from app_config import RISK_LEVEL_SPECS
        spec = RISK_LEVEL_SPECS.get(self.risk_level)
        return spec.name_zh if spec else self.risk_level

    @property
    def is_safe(self) -> bool:
        """整體評估是否安全（分數 < 50 且纜繩充足）"""
        return (
            self.risk_score < 50.0
            and self.mooring_split.overall_status != "CRITICAL"
        )

    def to_dict(self) -> Dict[str, object]:
        """轉為巢狀字典（向下相容舊版 dict 介面）"""
        return {
            "risk_score":           self.risk_score,
            "risk_level":           self.risk_level,
            "risk_level_zh":        self.risk_level_zh,
            "mitigated_risk_score": self.mitigated_risk_score,
            "mooring_split": {
                "bow":   vars(self.mooring_split.bow),
                "stern": vars(self.mooring_split.stern),
            },
            "tug_recommendation":  self.tug_recommendation.to_dict(),
            "wind_force_summary":  self.wind_force_summary.to_dict(),
            "recommendations":     self.recommendations,
        }

    def __repr__(self) -> str:
        return (
            f"AnalysisResult("
            f"score={self.risk_score:.1f} "
            f"level={self.risk_level} "
            f"safe={self.is_safe} "
            f"mooring={self.mooring_split.overall_status})"
        )
