# awt_parser.py
# 將 AwtWeatherFetcher 回傳的資料轉換為與 WeatherParser 相同的資料結構
# 輸出 48h / 7d 兩種切片，方便後續報告模組統一使用

import math
from datetime import datetime, timezone, timedelta
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from constants import (
    kts_to_bft,
    HIGH_WIND_SPEED_kts, HIGH_WIND_SPEED_Bft,
    HIGH_GUST_SPEED_kts, HIGH_GUST_SPEED_Bft,
    HIGH_WAVE_SIG,
    BERTHING_SCORE_HIGH, BERTHING_SCORE_MEDIUM, BERTHING_SCORE_LOW,
    GUST_FACTOR_EXTREME, GUST_FACTOR_HIGH, GUST_FACTOR_MODERATE,
    CURRENT_STRONG, CURRENT_MODERATE, CURRENT_MILD,
    FOG_HUMIDITY_THRESHOLD,
    FOG_TEMP_DIFF_HIGH, FOG_TEMP_DIFF_MEDIUM, FOG_TEMP_DIFF_LOW,
    FOG_RADIATION_HUMIDITY, FOG_RADIATION_TEMP,
    DIVERGE_WIND_KTS, DIVERGE_VIS_NM, DIVERGE_WAVE_M, DIVERGE_CURR_MS,
    WAVE_STEEPNESS_DANGER, WAVE_STEEPNESS_CAUTION,
)

# ── 換算常數 ─────────────────────────────────────────────────────────────────
# AWT API wind.speed / wind.gust 欄位單位為 knots（非 m/s）。
# awt_crawler._parse_new_format() 已直接輸出正確 kts 值。
# MS_TO_KTS 僅供 _ms_to_kts() fallback 使用（當 kts 欄位缺值時）。
MS_TO_KTS: float = 1.0 / 0.514444   # m/s → knots（fallback 換算用）
KTS_TO_MS: float = 0.514444         # knots → m/s（顯示用）

_COMPASS_TO_DEG: Dict[str, float] = {
    'N': 0,   'NNE': 22.5,  'NE': 45,   'ENE': 67.5,
    'E': 90,  'ESE': 112.5, 'SE': 135,  'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225,  'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315,  'NNW': 337.5,
}

_DEG_TO_DIR: List[Tuple[float, str]] = [
    (11.25,  'N'),   (33.75,  'NNE'), (56.25,  'NE'),  (78.75,  'ENE'),
    (101.25, 'E'),   (123.75, 'ESE'), (146.25, 'SE'),  (168.75, 'SSE'),
    (191.25, 'S'),   (213.75, 'SSW'), (236.25, 'SW'),  (258.75, 'WSW'),
    (281.25, 'W'),   (303.75, 'WNW'), (326.25, 'NW'),  (348.75, 'NNW'),
]


# ================= 工具函式 =================

def _ms_to_kts(ms: Optional[float]) -> float:
    """m/s → knots，None 回傳 0.0。僅供 kts 欄位缺值時 fallback 使用。"""
    return (ms * MS_TO_KTS) if ms is not None else 0.0


def _safe_kts(kts_val: Optional[float], ms_val: Optional[float]) -> float:
    """
    優先使用已換算好的 kts 欄位；
    若無則從 ms fallback 換算；兩者都沒有則回傳 0.0。
    """
    if kts_val is not None and kts_val > 0:
        return float(kts_val)
    return _ms_to_kts(ms_val)


def _deg_to_compass(degrees: Optional[float]) -> str:
    if degrees is None:
        return 'N/A'
    d = degrees % 360
    for threshold, label in _DEG_TO_DIR:
        if d < threshold:
            return label
    return 'N'


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


# ================= 衍生指標計算函式 =================

def calc_gust_factor(wind_kts: float, gust_kts: float) -> Dict[str, Any]:
    """
    陣風係數（Gust Factor）= 陣風 / 平均風速。
    係數越高代表風況越不穩定，對靠離泊操作影響越大。
    風速 < 0.5 kts 時視為靜風，gust_factor 回傳 None。
    """
    if wind_kts < 0.5:
        return {'gust_factor': None, 'gust_stability': '🌀 靜風'}
    factor = gust_kts / wind_kts
    if   factor >= GUST_FACTOR_EXTREME:  stability = '⛔ 極不穩定'
    elif factor >= GUST_FACTOR_HIGH:     stability = '⚠️ 不穩定'
    elif factor >= GUST_FACTOR_MODERATE: stability = '⚡ 略不穩定'
    else:                                stability = '✅ 穩定'
    return {
        'gust_factor':    round(factor, 2),
        'gust_stability': stability,
    }


def calc_wave_steepness(height_m: float, period_s: float) -> Dict[str, Any]:
    """
    浪陡度（Wave Steepness）= H / L。
    L 由深水近似公式推算：L = 1.56 × T²（T 單位秒，L 單位公尺）。
    陡峭的浪對船體衝擊遠大於平緩的浪，即使浪高相同。
    """
    if not period_s or period_s < 0.1 or height_m <= 0:
        return {
            'wave_steepness':  None,
            'steepness_label': 'N/A',
            'wavelength_m':    None,
        }
    wavelength = 1.56 * (period_s ** 2)
    steepness  = height_m / wavelength
    if   steepness > WAVE_STEEPNESS_DANGER:  label = '⛔ 陡浪（Breaking Risk）'
    elif steepness > WAVE_STEEPNESS_CAUTION: label = '⚠️ 較陡'
    else:                                    label = '✅ 平緩'
    return {
        'wavelength_m':    round(wavelength, 1),
        'wave_steepness':  round(steepness, 4),
        'steepness_label': label,
    }


def calc_current_risk(
    current_speed_ms:  float,
    current_dir_deg:   Optional[float] = None,
    berth_heading_deg: Optional[float] = None,
) -> Dict[str, Any]:
    """
    流速風險評估。
    若提供泊位方位（berth_heading_deg），額外計算橫流分量。
    """
    if   current_speed_ms >= CURRENT_STRONG:
        level, label = 3, '⛔ 強流（靠泊高風險）'
    elif current_speed_ms >= CURRENT_MODERATE:
        level, label = 2, '⚠️ 中強流（建議加派拖船）'
    elif current_speed_ms >= CURRENT_MILD:
        level, label = 1, '⚡ 中流（注意）'
    else:
        level, label = 0, '✅ 弱流（安全）'

    result: Dict[str, Any] = {
        'current_speed_ms':   current_speed_ms,
        'current_risk_level': level,
        'current_risk_label': label,
    }

    if berth_heading_deg is not None and current_dir_deg is not None:
        angle_diff = abs(current_dir_deg - berth_heading_deg) % 360
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        cross = abs(current_speed_ms * math.sin(math.radians(angle_diff)))
        result['cross_current_ms']    = round(cross, 3)
        result['cross_current_label'] = (
            '⛔ 強橫流' if cross >= 0.8 else
            '⚠️ 中橫流' if cross >= 0.5 else '✅ 可接受'
        )
    return result


def calc_fog_risk(
    air_temp_c:   float,
    sea_temp_c:   float,
    humidity_pct: float,
) -> Dict[str, Any]:
    """
    海霧風險評估，區分平流霧與輻射霧兩種成因。
    平流霧：暖濕空氣流過冷海面（氣溫 > 海溫 + 高濕度）
    輻射霧：高濕度 + 低氣溫（通常夜間至清晨）
    """
    temp_diff = air_temp_c - sea_temp_c   # 正值 = 氣溫高於海溫（平流霧條件）

    if temp_diff > 0 and humidity_pct >= FOG_HUMIDITY_THRESHOLD:
        if   temp_diff >= FOG_TEMP_DIFF_HIGH:   fog_risk = '⛔ 極高（平流霧極易生成）'
        elif temp_diff >= FOG_TEMP_DIFF_MEDIUM: fog_risk = '⚠️ 高（平流霧風險）'
        elif temp_diff >= FOG_TEMP_DIFF_LOW:    fog_risk = '⚡ 中（注意能見度變化）'
        else:                                   fog_risk = '✅ 低'
    elif humidity_pct >= FOG_RADIATION_HUMIDITY and air_temp_c < FOG_RADIATION_TEMP:
        fog_risk = '⚠️ 高（輻射霧風險）'
    else:
        fog_risk = '✅ 低'

    return {
        'sea_air_temp_diff_c': round(temp_diff, 2),
        'fog_risk_label':      fog_risk,
        'advection_fog_risk':  (temp_diff > FOG_TEMP_DIFF_LOW
                                and humidity_pct >= FOG_HUMIDITY_THRESHOLD),
    }


def calc_port_pilot_divergence(
    port_wind_kts:    float,
    pilot_wind_kts:   float,
    port_vis_nm:      Optional[float],
    pilot_vis_nm:     Optional[float],
    port_wave_m:      float,
    pilot_wave_m:     float,
    port_current_ms:  float,
    pilot_current_ms: float,
) -> Dict[str, Any]:
    """
    港區（portForecast）vs 引水點（pilotForecast）差異警示。
    差異大代表進港過程中條件會急遽變化，船長需提前準備。
    """
    wind_diff = abs(port_wind_kts   - pilot_wind_kts)
    vis_diff  = abs((port_vis_nm or 99.0) - (pilot_vis_nm or 99.0))
    wave_diff = abs(port_wave_m     - pilot_wave_m)
    curr_diff = abs(port_current_ms - pilot_current_ms)

    alerts: List[str] = []
    if wind_diff >= DIVERGE_WIND_KTS:
        alerts.append(f"⚠️ 港區↔引水點風速差 {wind_diff:.1f} kts")
    if vis_diff >= DIVERGE_VIS_NM:
        alerts.append(f"⚠️ 港區↔引水點能見度差 {vis_diff:.1f} NM")
    if wave_diff >= DIVERGE_WAVE_M:
        alerts.append(f"⚠️ 港區↔引水點浪高差 {wave_diff:.2f} m")
    if curr_diff >= DIVERGE_CURR_MS:
        alerts.append(f"⚠️ 港區↔引水點流速差 {curr_diff:.2f} m/s")

    return {
        'wind_divergence_kts':   round(wind_diff, 2),
        'vis_divergence_nm':     round(vis_diff,  2),
        'wave_divergence_m':     round(wave_diff, 3),
        'current_divergence_ms': round(curr_diff, 3),
        'divergence_alerts':     alerts,
        'high_divergence':       len(alerts) >= 2,
    }


def calc_berthing_risk_score(
    wind_kts:    float,
    gust_factor: Optional[float],
    current_ms:  float,
    vis_nm:      Optional[float],
    wave_m:      float,
) -> Dict[str, Any]:
    """
    綜合靠泊風險評分（0 = 最安全，100 = 最危險）。
    加權：風速 30% + 陣風係數 20% + 流速 20% + 能見度 20% + 浪高 10%
    vis_nm 直接就是 NM，不需換算。
    """
    score = 0.0

    # ── 風速（30 分）
    if   wind_kts >= 34: score += 30
    elif wind_kts >= 28: score += 22
    elif wind_kts >= 22: score += 14
    elif wind_kts >= 15: score += 7

    # ── 陣風係數（20 分）
    if gust_factor is not None:
        if   gust_factor >= GUST_FACTOR_EXTREME:  score += 20
        elif gust_factor >= GUST_FACTOR_HIGH:     score += 13
        elif gust_factor >= GUST_FACTOR_MODERATE: score += 6

    # ── 流速（20 分）
    if   current_ms >= CURRENT_STRONG:   score += 20
    elif current_ms >= CURRENT_MODERATE: score += 13
    elif current_ms >= CURRENT_MILD:     score += 6

    # ── 能見度（20 分）— vis_nm 直接就是 NM
    v = vis_nm if vis_nm is not None else 99.0
    if   v < 0.3: score += 20
    elif v < 0.5: score += 16
    elif v < 1.0: score += 12
    elif v < 1.5: score += 8
    elif v < 3.0: score += 4

    # ── 浪高（10 分）
    if   wave_m >= 4.0: score += 10
    elif wave_m >= 3.5: score += 8
    elif wave_m >= 2.5: score += 5
    elif wave_m >= 1.5: score += 2

    if   score >= BERTHING_SCORE_HIGH:   label = '⛔ 高風險'
    elif score >= BERTHING_SCORE_MEDIUM: label = '⚠️ 中風險'
    elif score >= BERTHING_SCORE_LOW:    label = '⚡ 低風險'
    else:                                label = '✅ 安全'

    return {
        'berthing_risk_score': round(score, 1),
        'berthing_risk_label': label,
    }


# ================= AWT 資料結構 =================

@dataclass
class AwtWeatherRecord:
    """
    AWT Port Forecast 風浪記錄。
    ✅ visibility_nm / pilot_visibility_nm 欄位單位統一為 NM。
    wind_speed_kts / wind_gust_kts 為 API 原始 kts 值。
    """
    time:            datetime
    lct_time:        datetime
    wind_direction:  str
    wind_speed_kts:  float
    wind_gust_kts:   float
    wave_direction:  str
    wave_height:     float
    wave_max:        float
    wave_period:     float

    # ── portForecast 額外欄位 ────────────────────────────────────────
    swell_height_m:      Optional[float] = None
    current_dir_deg:     Optional[float] = None
    current_speed_ms:    Optional[float] = None
    visibility_nm:       Optional[float] = None   # ✅ 單位：NM
    relative_humidity:   Optional[float] = None
    air_temp_c:          Optional[float] = None
    precip_6h_mm:        Optional[float] = None
    wind_wave_height_m:  Optional[float] = None

    # ── pilotForecast 欄位 ───────────────────────────────────────────
    pilot_swell_height_m:     Optional[float] = None
    pilot_swell_period_s:     Optional[float] = None
    pilot_swell_dir_deg:      Optional[float] = None
    pilot_wave_height_m:      Optional[float] = None
    pilot_wave_max_m:         Optional[float] = None
    pilot_wave_dir_deg:       Optional[float] = None
    pilot_sea_surface_temp_c: Optional[float] = None
    pilot_current_dir_deg:    Optional[float] = None
    pilot_current_speed_ms:   Optional[float] = None
    pilot_visibility_nm:      Optional[float] = None   # ✅ 單位：NM

    raw_data: Optional[Dict[str, Any]] = field(default=None, repr=False)

    # ── 基本衍生屬性 ────────────────────────────────────────────────────

    @property
    def wind_speed_ms(self) -> float:
        """kts 反向換算為 m/s（顯示用）。"""
        return self.wind_speed_kts * KTS_TO_MS

    @property
    def wind_speed_bft(self) -> int:
        return kts_to_bft(self.wind_speed_kts)

    @property
    def wind_gust_ms(self) -> float:
        """kts 反向換算為 m/s（顯示用）。"""
        return self.wind_gust_kts * KTS_TO_MS

    @property
    def wind_gust_bft(self) -> int:
        return kts_to_bft(self.wind_gust_kts)

    @property
    def wind_dir_deg(self) -> float:
        return _COMPASS_TO_DEG.get(self.wind_direction, 0.0)

    @property
    def wave_dir_deg(self) -> float:
        return _COMPASS_TO_DEG.get(self.wave_direction, 0.0)

    @property
    def wave_sig_m(self) -> float:
        return self.wave_height

    @property
    def wave_max_m(self) -> float:
        return self.wave_max

    @property
    def wave_period_s(self) -> float:
        return self.wave_period

    # ── 向下相容 property（已棄用，避免舊程式碼 crash）──────────────

    @property
    def visibility_km(self) -> Optional[float]:
        """⚠️ 已棄用，請改用 visibility_nm。"""
        return None

    @property
    def pilot_visibility_km(self) -> Optional[float]:
        """⚠️ 已棄用，請改用 pilot_visibility_nm。"""
        return None

    # ── 安全分析衍生屬性 ────────────────────────────────────────────────

    @property
    def gust_factor_info(self) -> Dict[str, Any]:
        """陣風係數：評估風況穩定性，靠離泊操作參考。"""
        return calc_gust_factor(self.wind_speed_kts, self.wind_gust_kts)

    @property
    def gust_factor(self) -> Optional[float]:
        return self.gust_factor_info.get('gust_factor')

    @property
    def wave_steepness_info(self) -> Dict[str, Any]:
        """
        浪陡度：優先使用 pilotForecast 湧浪週期與浪高；
        pilotForecast 無資料時 fallback 用 portForecast 數值。
        """
        period = self.pilot_swell_period_s or self.wave_period
        height = self.pilot_wave_height_m  or self.wave_height
        return calc_wave_steepness(height, period)

    @property
    def steepness_label(self) -> str:
        return self.wave_steepness_info.get('steepness_label', 'N/A')

    @property
    def current_risk_info(self) -> Dict[str, Any]:
        """流速風險：優先使用 pilotForecast 流速。"""
        speed = self.pilot_current_speed_ms or self.current_speed_ms or 0.0
        dir_  = self.pilot_current_dir_deg  or self.current_dir_deg
        return calc_current_risk(speed, dir_)

    @property
    def fog_risk_info(self) -> Dict[str, Any]:
        """
        海霧風險：海溫來自 pilotForecast，氣溫來自 portForecast。
        任一缺值時回傳低風險（資料不足）。
        """
        if self.pilot_sea_surface_temp_c is None or self.air_temp_c is None:
            return {
                'sea_air_temp_diff_c': None,
                'fog_risk_label':      '✅ 低（資料不足）',
                'advection_fog_risk':  False,
            }
        return calc_fog_risk(
            air_temp_c   = self.air_temp_c,
            sea_temp_c   = self.pilot_sea_surface_temp_c,
            humidity_pct = self.relative_humidity or 0.0,
        )

    @property
    def fog_risk_label(self) -> str:
        return self.fog_risk_info.get('fog_risk_label', '✅ 低')

    @property
    def port_pilot_divergence_info(self) -> Dict[str, Any]:
        """
        港區 vs 引水點差異警示。
        pilot_wind_kts 從 raw_data 取得。
        """
        pilot_wind_kts = (
            _safe_float(self.raw_data.get('pilot_wind_speed_kts'))
            if self.raw_data and self.raw_data.get('pilot_wind_speed_kts')
            else self.wind_speed_kts
        )
        return calc_port_pilot_divergence(
            port_wind_kts    = self.wind_speed_kts,
            pilot_wind_kts   = pilot_wind_kts,
            port_vis_nm      = self.visibility_nm,       # ✅ 直接 NM
            pilot_vis_nm     = self.pilot_visibility_nm, # ✅ 直接 NM
            port_wave_m      = self.wave_height,
            pilot_wave_m     = self.pilot_wave_height_m or self.wave_height,
            port_current_ms  = self.current_speed_ms    or 0.0,
            pilot_current_ms = self.pilot_current_speed_ms or 0.0,
        )

    @property
    def divergence_alerts(self) -> List[str]:
        return self.port_pilot_divergence_info.get('divergence_alerts', [])

    @property
    def high_divergence(self) -> bool:
        return self.port_pilot_divergence_info.get('high_divergence', False)

    @property
    def berthing_risk_info(self) -> Dict[str, Any]:
        """
        綜合靠泊風險評分（0–100）。
        全部使用 pilotForecast 數值，更能代表實際靠泊環境。
        """
        pilot_wind_kts = (
            _safe_float(self.raw_data.get('pilot_wind_speed_kts'))
            if self.raw_data and self.raw_data.get('pilot_wind_speed_kts')
            else self.wind_speed_kts
        )
        return calc_berthing_risk_score(
            wind_kts    = pilot_wind_kts,
            gust_factor = self.gust_factor,
            current_ms  = self.pilot_current_speed_ms or self.current_speed_ms or 0.0,
            vis_nm      = self.pilot_visibility_nm or self.visibility_nm,  # ✅ 直接 NM
            wave_m      = self.pilot_wave_height_m or self.wave_height,
        )

    @property
    def berthing_risk_score(self) -> float:
        return self.berthing_risk_info['berthing_risk_score']

    @property
    def berthing_risk_label(self) -> str:
        return self.berthing_risk_info['berthing_risk_label']

    # ── 序列化 ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        gust_info  = self.gust_factor_info
        steep_info = self.wave_steepness_info
        curr_info  = self.current_risk_info
        fog_info   = self.fog_risk_info
        div_info   = self.port_pilot_divergence_info
        bert_info  = self.berthing_risk_info

        return {
            'time':           self.time,
            'lct_time':       self.lct_time,
            'wind_direction': self.wind_direction,
            'wind_speed_kts': self.wind_speed_kts,
            'wind_speed_ms':  self.wind_speed_ms,
            'wind_speed_bft': self.wind_speed_bft,
            'wind_gust_kts':  self.wind_gust_kts,
            'wind_gust_ms':   self.wind_gust_ms,
            'wind_gust_bft':  self.wind_gust_bft,
            'wave_direction': self.wave_direction,
            'wave_height':    self.wave_height,
            'wave_max':       self.wave_max,
            'wave_period':    self.wave_period,
            'wind_dir_deg':   self.wind_dir_deg,
            'wave_dir_deg':   self.wave_dir_deg,
            'swell_height_m':           self.swell_height_m,
            'current_dir_deg':          self.current_dir_deg,
            'current_speed_ms':         self.current_speed_ms,
            'visibility_nm':            self.visibility_nm,       # ✅
            'relative_humidity':        self.relative_humidity,
            'air_temp_c':               self.air_temp_c,
            'precip_6h_mm':             self.precip_6h_mm,
            'wind_wave_height_m':       self.wind_wave_height_m,
            'pilot_swell_height_m':     self.pilot_swell_height_m,
            'pilot_swell_period_s':     self.pilot_swell_period_s,
            'pilot_swell_dir_deg':      self.pilot_swell_dir_deg,
            'pilot_wave_height_m':      self.pilot_wave_height_m,
            'pilot_wave_max_m':         self.pilot_wave_max_m,
            'pilot_wave_dir_deg':       self.pilot_wave_dir_deg,
            'pilot_sea_surface_temp_c': self.pilot_sea_surface_temp_c,
            'pilot_current_dir_deg':    self.pilot_current_dir_deg,
            'pilot_current_speed_ms':   self.pilot_current_speed_ms,
            'pilot_visibility_nm':      self.pilot_visibility_nm,  # ✅
            'gust_factor':           gust_info.get('gust_factor'),
            'gust_stability':        gust_info.get('gust_stability'),
            'wave_steepness':        steep_info.get('wave_steepness'),
            'steepness_label':       steep_info.get('steepness_label'),
            'wavelength_m':          steep_info.get('wavelength_m'),
            'current_risk_level':    curr_info.get('current_risk_level'),
            'current_risk_label':    curr_info.get('current_risk_label'),
            'sea_air_temp_diff_c':   fog_info.get('sea_air_temp_diff_c'),
            'fog_risk_label':        fog_info.get('fog_risk_label'),
            'advection_fog_risk':    fog_info.get('advection_fog_risk'),
            'divergence_alerts':     div_info.get('divergence_alerts'),
            'high_divergence':       div_info.get('high_divergence'),
            'wind_divergence_kts':   div_info.get('wind_divergence_kts'),
            'vis_divergence_nm':     div_info.get('vis_divergence_nm'),
            'berthing_risk_score':   bert_info.get('berthing_risk_score'),
            'berthing_risk_label':   bert_info.get('berthing_risk_label'),
        }

    def __repr__(self) -> str:
        return (
            f"AwtWeatherRecord("
            f"time={self.time.strftime('%Y-%m-%d %H:%M')}, "
            f"wind={self.wind_direction} {self.wind_speed_kts:.1f}kts "
            f"(gust {self.wind_gust_kts:.1f}kts / GF:{self.gust_factor}), "
            f"LCT={self.lct_time.strftime('%H:%M')}, "
            f"wave={self.wave_direction} {self.wave_height:.2f}m, "
            f"vis={self.visibility_nm} NM, "
            f"berthing={self.berthing_risk_label})"
        )


# ================= AWT 天氣狀況記錄 =================

@dataclass
class AwtConditionRecord:
    """
    AWT Port Forecast 天氣狀況記錄。
    visibility 欄位儲存 NM 數值字串，例如 "10.0"。
    """
    time:          datetime
    lct_time:      datetime
    temperature:   Optional[float]
    precipitation: float
    pressure:      Optional[float]
    visibility:    str
    weather_code:  str

    relative_humidity:   Optional[float] = None
    current_dir_deg:     Optional[float] = None
    current_speed_ms:    Optional[float] = None
    sea_surface_temp_c:  Optional[float] = None

    @property
    def visibility_nm(self) -> Optional[float]:
        """直接回傳 NM 值。"""
        try:
            return float(self.visibility)
        except (ValueError, TypeError):
            return None

    @property
    def visibility_meters(self) -> Optional[float]:
        """NM → 公尺，供需要公尺單位的模組使用。"""
        nm = self.visibility_nm
        return nm * 1852 if nm is not None else None

    @property
    def weather_description(self) -> str:
        return 'N/A（AWT 無天氣代碼）'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'time':                self.time,
            'lct_time':            self.lct_time,
            'temperature':         self.temperature,
            'precipitation':       self.precipitation,
            'pressure':            self.pressure,
            'visibility':          self.visibility,
            'visibility_nm':       self.visibility_nm,
            'visibility_meters':   self.visibility_meters,
            'weather_code':        self.weather_code,
            'weather_description': self.weather_description,
            'relative_humidity':   self.relative_humidity,
            'current_dir_deg':     self.current_dir_deg,
            'current_speed_ms':    self.current_speed_ms,
            'sea_surface_temp_c':  self.sea_surface_temp_c,
        }

    def __repr__(self) -> str:
        return (
            f"AwtConditionRecord("
            f"time={self.time.strftime('%Y-%m-%d %H:%M')}, "
            f"LCT={self.lct_time.strftime('%H:%M')}, "
            f"temp={self.temperature}°C, "
            f"vis={self.visibility} NM, "
            f"humidity={self.relative_humidity}%)"
        )


# ================= AWT 解析器 =================

class AwtParser:
    """
    將 AwtWeatherFetcher.fetch_port_weather() 回傳的 List[Dict] 轉換為
    AwtWeatherRecord / AwtConditionRecord 列表，
    並提供 48h / 7d / custom 三種切片方法。
    """

    def parse(
        self,
        raw_records:     List[Dict[str, Any]],
        port_name:       str = "Unknown Port",
        tz_offset_hours: int = 8,
    ) -> Tuple[str, List[AwtWeatherRecord], List[AwtConditionRecord]]:
        if not raw_records:
            return port_name, [], []

        lct_tz = timezone(timedelta(hours=tz_offset_hours))
        wind_records: List[AwtWeatherRecord]   = []
        cond_records: List[AwtConditionRecord] = []

        for rec in raw_records:
            # ── 時間 ──────────────────────────────────────────────────
            valid_time: datetime = rec['valid_time']
            if valid_time.tzinfo is None:
                valid_time = valid_time.replace(tzinfo=timezone.utc)

            tz_offset = rec.get('timezone_offset')
            if tz_offset is not None:
                try:
                    lct_tz = timezone(timedelta(hours=int(tz_offset)))
                except (TypeError, ValueError):
                    pass
            lct_time = valid_time.astimezone(lct_tz)

            # ── 風向 ──────────────────────────────────────────────────
            wind_dir_str = _deg_to_compass(rec.get('wind_dir_deg'))
            wave_dir_str = _deg_to_compass(rec.get('wave_dir_deg'))

            # ── 風速（優先 kts，fallback ms）─────────────────────────
            wind_speed_kts = _safe_kts(
                rec.get('wind_speed_kts'), rec.get('wind_speed_ms'))
            wind_gust_kts  = _safe_kts(
                rec.get('wind_gust_kts'),  rec.get('wind_gust_ms'))

            # ── 浪高 / 週期 ───────────────────────────────────────────
            wave_height = _safe_float(
                rec.get('sig_wave_m') or rec.get('wave_height_m'))
            wave_max    = _safe_float(
                rec.get('max_wave_m') or rec.get('wave_max_m'))
            wave_period = _safe_float(rec.get('wave_period_s'))

            # ── 能見度（AWT 直接回傳 NM）─────────────────────────────
            raw_vis_nm = rec.get('visibility_nm')
            vis_nm     = float(raw_vis_nm) if raw_vis_nm is not None else None
            vis_str    = f"{vis_nm:.1f}" if vis_nm is not None else 'N/A'

            # ── pilot 能見度（直接 NM）───────────────────────────────
            raw_pilot_vis_nm = rec.get('pilot_visibility_nm')
            pilot_vis_nm     = (float(raw_pilot_vis_nm)
                                if raw_pilot_vis_nm is not None else None)

            # ── 建立 AwtWeatherRecord ─────────────────────────────────
            wind_records.append(AwtWeatherRecord(
                time           = valid_time,
                lct_time       = lct_time,
                wind_direction = wind_dir_str,
                wind_speed_kts = wind_speed_kts,
                wind_gust_kts  = wind_gust_kts,
                wave_direction = wave_dir_str,
                wave_height    = wave_height,
                wave_max       = wave_max,
                wave_period    = wave_period,
                swell_height_m           = rec.get('swell_height_m'),
                current_dir_deg          = rec.get('current_dir_deg'),
                current_speed_ms         = rec.get('current_speed_ms'),
                visibility_nm            = vis_nm,        # ✅
                relative_humidity        = rec.get('relative_humidity'),
                air_temp_c               = rec.get('air_temp_c'),
                precip_6h_mm             = rec.get('precip_6h_mm'),
                wind_wave_height_m       = rec.get('wind_wave_height_m'),
                pilot_swell_height_m     = rec.get('pilot_swell_height_m'),
                pilot_swell_period_s     = rec.get('pilot_swell_period_s'),
                pilot_swell_dir_deg      = rec.get('pilot_swell_dir_deg'),
                pilot_wave_height_m      = rec.get('pilot_wave_height_m'),
                pilot_wave_max_m         = rec.get('pilot_wave_max_m'),
                pilot_wave_dir_deg       = rec.get('pilot_wave_dir_deg'),
                pilot_sea_surface_temp_c = rec.get('pilot_sea_surface_temp_c'),
                pilot_current_dir_deg    = rec.get('pilot_current_dir_deg'),
                pilot_current_speed_ms   = rec.get('pilot_current_speed_ms'),
                pilot_visibility_nm      = pilot_vis_nm,  # ✅
                raw_data = rec,
            ))

            # ── 建立 AwtConditionRecord ───────────────────────────────
            cond_records.append(AwtConditionRecord(
                time               = valid_time,
                lct_time           = lct_time,
                temperature        = rec.get('air_temp_c'),
                precipitation      = _safe_float(rec.get('precip_6h_mm')),
                pressure           = None,
                visibility         = vis_str,
                weather_code       = 'N/A',
                relative_humidity  = rec.get('relative_humidity'),
                current_dir_deg    = rec.get('current_dir_deg'),
                current_speed_ms   = rec.get('current_speed_ms'),
                sea_surface_temp_c = rec.get('pilot_sea_surface_temp_c'),
            ))

        return port_name, wind_records, cond_records

    # ── 切片方法 ──────────────────────────────────────────────────────

    def slice_48h(
        self,
        wind_records:   List[AwtWeatherRecord],
        cond_records:   List[AwtConditionRecord],
        reference_time: Optional[datetime] = None,
    ) -> Tuple[List[AwtWeatherRecord], List[AwtConditionRecord]]:
        if not wind_records:
            return [], []
        ref = reference_time or wind_records[0].time
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        cutoff = ref + timedelta(hours=48)
        return ([r for r in wind_records if r.time < cutoff],
                [r for r in cond_records if r.time < cutoff])

    def slice_7d(
        self,
        wind_records:   List[AwtWeatherRecord],
        cond_records:   List[AwtConditionRecord],
        reference_time: Optional[datetime] = None,
    ) -> Tuple[List[AwtWeatherRecord], List[AwtConditionRecord]]:
        if not wind_records:
            return [], []
        ref = reference_time or wind_records[0].time
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        cutoff = ref + timedelta(days=7)
        return ([r for r in wind_records if r.time < cutoff],
                [r for r in cond_records if r.time < cutoff])

    def slice_custom(
        self,
        wind_records:   List[AwtWeatherRecord],
        cond_records:   List[AwtConditionRecord],
        hours:          int,
        reference_time: Optional[datetime] = None,
    ) -> Tuple[List[AwtWeatherRecord], List[AwtConditionRecord]]:
        if not wind_records:
            return [], []
        ref = reference_time or wind_records[0].time
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        cutoff = ref + timedelta(hours=hours)
        return ([r for r in wind_records if r.time < cutoff],
                [r for r in cond_records if r.time < cutoff])

    # ── 統計 ──────────────────────────────────────────────────────────

    @staticmethod
    def get_statistics(records: List[AwtWeatherRecord]) -> Dict[str, Any]:
        if not records:
            return {}

        speeds_kts  = [r.wind_speed_kts     for r in records]
        gusts_kts   = [r.wind_gust_kts       for r in records]
        waves       = [r.wave_height         for r in records]
        wave_maxes  = [r.wave_max            for r in records]
        bert_scores = [r.berthing_risk_score for r in records]

        best_window  = min(records, key=lambda r: r.berthing_risk_score)
        worst_window = max(records, key=lambda r: r.berthing_risk_score)

        return {
            'total_records': len(records),
            'time_range': {
                'start': min(r.time for r in records),
                'end':   max(r.time for r in records),
            },
            'wind': {
                'min_kts':      min(speeds_kts),
                'max_kts':      max(speeds_kts),
                'avg_kts':      mean(speeds_kts),
                'min_ms':       min(r.wind_speed_ms  for r in records),
                'max_ms':       max(r.wind_speed_ms  for r in records),
                'avg_ms':       mean(r.wind_speed_ms for r in records),
                'min_bft':      min(r.wind_speed_bft for r in records),
                'max_bft':      max(r.wind_speed_bft for r in records),
                'max_gust_kts': max(gusts_kts),
                'max_gust_ms':  max(r.wind_gust_ms   for r in records),
                'max_gust_bft': max(r.wind_gust_bft  for r in records),
            },
            'wave': {
                'min':      min(waves),
                'max':      max(waves),
                'avg':      mean(waves),
                'max_wave': max(wave_maxes),
            },
            'berthing': {
                'best_time_lct':   best_window.lct_time.strftime('%m/%d %H:%M'),
                'best_score':      best_window.berthing_risk_score,
                'best_label':      best_window.berthing_risk_label,
                'worst_time_lct':  worst_window.lct_time.strftime('%m/%d %H:%M'),
                'worst_score':     worst_window.berthing_risk_score,
                'worst_label':     worst_window.berthing_risk_label,
                'avg_score':       round(mean(bert_scores), 1),
                'high_risk_count': sum(
                    1 for s in bert_scores if s >= BERTHING_SCORE_HIGH),
            },
        }

    @staticmethod
    def filter_high_risk_records(
        records:            List[AwtWeatherRecord],
        wind_kts_threshold: float = HIGH_WIND_SPEED_kts,
        wind_bft_threshold: int   = HIGH_WIND_SPEED_Bft,
        gust_kts_threshold: float = HIGH_GUST_SPEED_kts,
        gust_bft_threshold: int   = HIGH_GUST_SPEED_Bft,
        wave_threshold:     float = HIGH_WAVE_SIG,
    ) -> List[AwtWeatherRecord]:
        return [
            r for r in records
            if (r.wind_speed_kts >= wind_kts_threshold
                or r.wind_speed_bft >= wind_bft_threshold
                or r.wind_gust_kts  >= gust_kts_threshold
                or r.wind_gust_bft  >= gust_bft_threshold
                or r.wave_height    >= wave_threshold)
        ]

    @staticmethod
    def get_best_berthing_windows(
        records: List[AwtWeatherRecord],
        top_n:   int = 3,
    ) -> List[Dict[str, Any]]:
        """回傳靠泊風險評分最低的前 top_n 個時間點。"""
        sorted_recs = sorted(records, key=lambda r: r.berthing_risk_score)
        result = []
        for r in sorted_recs[:top_n]:
            result.append({
                'time_utc':          r.time.strftime('%m/%d %H:%M UTC'),
                'time_lct':          r.lct_time.strftime('%m/%d %H:%M LT'),
                'berthing_score':    r.berthing_risk_score,
                'berthing_label':    r.berthing_risk_label,
                'wind_kts':          r.wind_speed_kts,
                'gust_kts':          r.wind_gust_kts,
                'gust_factor':       r.gust_factor,
                'gust_stability':    r.gust_factor_info.get('gust_stability'),
                'current_ms':        r.pilot_current_speed_ms or r.current_speed_ms,
                'current_label':     r.current_risk_info.get('current_risk_label'),
                'vis_nm':            r.pilot_visibility_nm or r.visibility_nm,  # ✅
                'fog_risk':          r.fog_risk_info.get('fog_risk_label'),
                'wave_m':            r.pilot_wave_height_m or r.wave_height,
                'steepness_label':   r.wave_steepness_info.get('steepness_label'),
                'divergence_alerts': r.port_pilot_divergence_info.get(
                    'divergence_alerts', []),
            })
        return result


# ================= 便利函式 =================

def parse_awt_forecast(
    raw_records:     List[Dict[str, Any]],
    port_name:       str = "Unknown Port",
    tz_offset_hours: int = 8,
) -> Dict[str, Any]:
    """一次性解析並回傳 48h / 7d 兩種切片與最佳靠泊時窗。"""
    parser = AwtParser()
    port_name, all_wind, all_cond = parser.parse(
        raw_records, port_name, tz_offset_hours)
    w48, c48 = parser.slice_48h(all_wind, all_cond)
    w7d, c7d = parser.slice_7d(all_wind, all_cond)

    return {
        'port_name': port_name,
        '48h': {
            'wind':          w48,
            'condition':     c48,
            'stats':         AwtParser.get_statistics(w48),
            'best_berthing': AwtParser.get_best_berthing_windows(w48),
        },
        '7d': {
            'wind':          w7d,
            'condition':     c7d,
            'stats':         AwtParser.get_statistics(w7d),
            'best_berthing': AwtParser.get_best_berthing_windows(w7d),
        },
        'all': {
            'wind':      all_wind,
            'condition': all_cond,
        },
    }


# ================= 測試 =================

if __name__ == "__main__":

    def _make_mock_records(n: int = 40) -> List[Dict[str, Any]]:
        """產生 n 筆模擬資料（每 6 小時一筆）。"""
        base = datetime(2026, 4, 14, 6, 0, tzinfo=timezone.utc)
        records = []
        for i in range(n):
            t         = base + timedelta(hours=6 * i)
            has_pilot = (i % 4 != 0)
            ws_kts    = round(5.0 + i * 0.5, 1)
            gust_kts  = round(8.0 + i * 0.5, 1)
            records.append({
                'valid_time':      t,
                'timezone_offset': 8,
                'wind_speed_kts':  ws_kts,
                'wind_gust_kts':   gust_kts,
                'wind_speed_ms':   round(ws_kts   * KTS_TO_MS, 4),
                'wind_gust_ms':    round(gust_kts * KTS_TO_MS, 4),
                'wind_dir_deg':    (270.0 + i * 5) % 360,
                'sig_wave_m':      round(0.5 + i * 0.08, 2),
                'max_wave_m':      round(0.9 + i * 0.08, 2),
                'wave_dir_deg':    230.0,
                'wave_period_s':   6.0 if has_pilot else None,
                'swell_height_m':  round(0.4 + i * 0.05, 2),
                'air_temp_c':      round(26.0 - i * 0.1, 1),
                # ✅ 測試資料使用 visibility_nm（API 直接回傳 NM）
                'visibility_nm':   max(1.0, 10.0 - i * 0.3),
                'relative_humidity':  min(99.0, 60.0 + i * 1.2),
                'current_dir_deg':    350.0,
                'current_speed_ms':   round(0.08 + i * 0.02, 3),
                'precip_6h_mm':       0.0,
                'wind_wave_height_m': 0.1,
                'pilot_wind_speed_kts':     round(ws_kts * 1.1, 1) if has_pilot else None,
                'pilot_swell_height_m':     0.5   if has_pilot else None,
                'pilot_swell_period_s':     6.2   if has_pilot else None,
                'pilot_swell_dir_deg':      234.0 if has_pilot else None,
                'pilot_wave_height_m':      0.54  if has_pilot else None,
                'pilot_wave_max_m':         1.0   if has_pilot else None,
                'pilot_wave_dir_deg':       236.0 if has_pilot else None,
                'pilot_sea_surface_temp_c': round(24.0 - i * 0.1, 1) if has_pilot else None,
                'pilot_current_dir_deg':    24.0  if has_pilot else None,
                'pilot_current_speed_ms':   round(0.105 + i * 0.02, 3) if has_pilot else None,
                # ✅ 測試資料使用 pilot_visibility_nm
                'pilot_visibility_nm':      max(0.5, 10.8 - i * 0.4) if has_pilot else None,
            })
        return records

    mock_data = _make_mock_records(n=40)
    result    = parse_awt_forecast(
        mock_data, port_name="高雄港 TWKHH", tz_offset_hours=8)

    for label in ('48h', '7d'):
        d = result[label]
        w = d['wind']
        s = d['stats']
        print("=" * 75)
        print(f"【{label.upper()} 切片】{result['port_name']}  ({len(w)} 筆)")
        if not w:
            continue

        print(f"\n  ── 風浪統計 ──")
        print(f"  風速: {s['wind']['min_kts']:.1f} ~ {s['wind']['max_kts']:.1f} kts")
        print(f"  最大陣風: {s['wind']['max_gust_kts']:.1f} kts")
        print(f"  浪高: {s['wave']['min']:.2f} ~ {s['wave']['max']:.2f} m")

        bs = s['berthing']
        print(f"\n  ── 靠泊安全統計 ──")
        print(f"  最佳時窗: {bs['best_time_lct']}  "
              f"評分 {bs['best_score']}  {bs['best_label']}")
        print(f"  最差時窗: {bs['worst_time_lct']}  "
              f"評分 {bs['worst_score']}  {bs['worst_label']}")
        print(f"  平均評分: {bs['avg_score']}  "
              f"高風險時段: {bs['high_risk_count']} 筆")

        print(f"\n  ── 建議靠泊時窗（前 3）──")
        for j, bw in enumerate(d['best_berthing'], 1):
            vis  = bw['vis_nm']
            curr = bw['current_ms']
            print(f"  {j}. {bw['time_lct']}  "
                  f"評分 {bw['berthing_score']}  {bw['berthing_label']}")
            if curr:
                print(f"     風 {bw['wind_kts']:.1f} kts  "
                      f"陣風係數 {bw['gust_factor']}（{bw['gust_stability']}）  "
                      f"流速 {curr:.2f} m/s")
            else:
                print(f"     風 {bw['wind_kts']:.1f} kts  "
                      f"陣風係數 {bw['gust_factor']}（{bw['gust_stability']}）  "
                      f"流速 N/A")
            if vis:
                print(f"     能見度 {vis:.1f} NM  "
                      f"海霧: {bw['fog_risk']}  "
                      f"浪陡: {bw['steepness_label']}")
            else:
                print(f"     能見度 N/A  "
                      f"海霧: {bw['fog_risk']}  "
                      f"浪陡: {bw['steepness_label']}")
            for alert in bw['divergence_alerts']:
                print(f"     {alert}")

        print(f"\n  ── 前 4 筆詳細預覽 ──")
        for r in w[:4]:
            vis_str = f"{r.visibility_nm:.1f} NM" if r.visibility_nm else "N/A"
            print(f"  {r.lct_time.strftime('%m/%d %H:%M')} LT  "
                  f"風 {r.wind_direction:>4} {r.wind_speed_kts:>5.1f}kts  "
                  f"GF:{r.gust_factor}({r.gust_factor_info['gust_stability']})  "
                  f"浪 {r.wave_height:.2f}m({r.steepness_label})  "
                  f"能見 {vis_str}  "
                  f"霧:{r.fog_risk_label}  "
                  f"靠泊:{r.berthing_risk_label}({r.berthing_risk_score})")

    print("\n✅ AwtParser 測試完成")

# ── 檔案結束：awt_parser.py ───────────────────────────────────────────────────
