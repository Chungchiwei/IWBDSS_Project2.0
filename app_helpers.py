# app_helpers.py
"""
應用程式輔助函數

提供單位轉換、風險評估、港口資訊格式化、
DataFrame 處理與力學計算等工具函式。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from app_config import (
    BEAUFORT,
    COMPASS,
    FIELD_MAPPING,
    MOORING,
    PHYSICS,
    RISK_LEVEL_SPECS,
    THRESHOLDS,
    AppConfig,
)

logger = logging.getLogger(__name__)


# ================= 向下相容常數（供舊模組 import）=================
AIR_DENSITY = PHYSICS.air_density
GRAVITY     = PHYSICS.gravity
KNOTS_TO_MS = PHYSICS.knots_to_ms


# ================= 資料結構 =================

@dataclass
class WindForceResult:
    """風力計算結果"""
    total_force_N:        float
    transverse_force_N:   float
    longitudinal_force_N: float
    angle_diff:           float
    is_offshore:          bool


@dataclass
class MooringCapacityResult:
    """纜繩容量計算結果"""
    total_capacity_kN:        float
    bow_capacity_kN:          float
    bow_spring_capacity_kN:   float
    stern_capacity_kN:        float
    stern_spring_capacity_kN: float
    wll_per_line_kN:          float


@dataclass
class TugCapacityResult:
    """拖船容量計算結果"""
    total_push_kN:             float
    bp_per_tug_ton:            float
    effective_push_per_tug_kN: float


@dataclass
class PortDisplayInfo:
    """港口顯示資訊"""
    port_name:  str
    port_code:  str
    country:    str
    station_id: str
    lat_ns:     str
    lon_ew:     str
    latitude:   float = 0.0
    longitude:  float = 0.0


# ================= 單位轉換 =================

def knots_to_ms(knots: float) -> float:
    """節 → 公尺/秒"""
    return float(knots) * PHYSICS.knots_to_ms


def ms_to_knots(ms: float) -> float:
    """公尺/秒 → 節"""
    return float(ms) / PHYSICS.knots_to_ms


def knots_to_beaufort(knots: float) -> int:
    """節 → 蒲福風級"""
    return BEAUFORT.from_knots(knots)


def beaufort_to_description(scale: int, lang: str = "zh") -> str:
    """蒲福風級 → 描述文字"""
    _DESCRIPTIONS_ZH = {
        0: "無風", 1: "軟風", 2: "輕風",  3: "微風",
        4: "和風", 5: "清風", 6: "強風",  7: "疾風",
        8: "大風", 9: "烈風", 10: "狂風", 11: "暴風", 12: "颶風",
    }
    return _DESCRIPTIONS_ZH.get(scale, f"{scale}級風")


def compass_to_degrees(direction: Any, default: float = 0.0) -> float:
    """
    羅盤方位 → 度數。

    Args:
        direction: 字串（如 'NNE'）或數值（直接回傳）
        default:   找不到對應時的預設值

    Returns:
        度數 (0–360)
    """
    if isinstance(direction, (int, float)):
        return float(direction)
    if isinstance(direction, str):
        return COMPASS.to_degrees_safe(direction, default=default)
    return default


def convert_lat_to_ns(lat: float) -> str:
    """緯度數值 → 度分格式字串（如 `25°03.60'N`）"""
    return _decimal_to_dms(lat, pos="N", neg="S")


def convert_lon_to_ew(lon: float) -> str:
    """經度數值 → 度分格式字串（如 `121°30.00'E`）"""
    return _decimal_to_dms(lon, pos="E", neg="W")


def _decimal_to_dms(value: float, pos: str, neg: str) -> str:
    """十進位度數 → 度分格式（內部共用）"""
    if value == 0.0:
        return "N/A"
    direction = pos if value >= 0 else neg
    abs_val   = abs(value)
    degrees   = int(abs_val)
    minutes   = (abs_val - degrees) * 60
    return f"{degrees}°{minutes:.2f}'{direction}"


# ================= 風險評估 =================

def get_risk_description(level: str, lang: str = "zh") -> str:
    """
    英文風險等級 key → 中文描述。
    找不到時回傳原始值。
    """
    spec = RISK_LEVEL_SPECS.get(level)
    return spec.name_zh if spec else level


def classify_wind_risk(wind_speed_kts: float, gust_speed_kts: float) -> str:
    """
    依風速與陣風速度分類風險等級。

    Returns:
        'low' | 'medium' | 'high' | 'extreme'
    """
    gust_level = THRESHOLDS.gust.classify(gust_speed_kts)
    wind_level = THRESHOLDS.wind.classify(wind_speed_kts)
    _ORDER     = ["low", "medium", "high", "extreme"]
    return _ORDER[max(_ORDER.index(gust_level), _ORDER.index(wind_level))]


def get_risk_color(risk_level: str) -> str:
    """風險等級 → 背景色（十六進位）"""
    spec = RISK_LEVEL_SPECS.get(risk_level)
    return spec.color_bg if spec else "#ffffff"


# ================= 港口資訊處理 =================

def get_port_full_info(port_code: str, crawler: Any) -> PortDisplayInfo:
    """
    取得港口完整資訊並格式化經緯度顯示。

    Args:
        port_code: 港口代碼
        crawler:   PortWeatherCrawler 實例

    Returns:
        PortDisplayInfo dataclass
    """
    port_info = crawler.get_port_info(port_code)

    if not port_info:
        logger.warning("找不到港口資訊：%s", port_code)
        return PortDisplayInfo(
            port_name="N/A", port_code=port_code,
            country="N/A",   station_id="N/A",
            lat_ns="N/A",    lon_ew="N/A",
        )

    # ── 修正 Bug 1：PortInfo dataclass 沒有 .name 屬性 ───────────────────────
    # 原版：port_info.name → AttributeError（PortInfo 欄位為 port_name_en）
    # 改寫：統一優先讀取 dict 格式（crawler.get_port_info() 回傳 dict），
    #       fallback 至 dataclass 屬性，確保兩種格式都能正確讀取。
    if isinstance(port_info, dict):
        lat        = port_info.get("latitude",     0.0)
        lon        = port_info.get("longitude",    0.0)
        name       = port_info.get("port_name_en", port_info.get("port_name", "N/A"))
        country    = port_info.get("country",      "N/A")
        station_id = port_info.get("station_id",   "N/A")
    else:
        # fallback：相容未來可能直接回傳 PortInfo dataclass 的情況
        lat        = getattr(port_info, "latitude",     0.0)
        lon        = getattr(port_info, "longitude",    0.0)
        name       = getattr(port_info, "port_name_en",
                        getattr(port_info, "port_name", "N/A"))
        country    = getattr(port_info, "country",      "N/A")
        station_id = getattr(port_info, "station_id",   "N/A")

    return PortDisplayInfo(
        port_name  = name,
        port_code  = port_code,
        country    = country,
        station_id = station_id,
        lat_ns     = convert_lat_to_ns(lat),
        lon_ew     = convert_lon_to_ew(lon),
        latitude   = lat,
        longitude  = lon,
    )


# ================= 力學計算 =================

def calculate_wind_force(
    wind_speed_ms: float,
    wind_dir_deg:  float,
    vessel_area:   float,
    cd:            float,
    berth_angle:   float,
    rho:           float = PHYSICS.air_density,
) -> WindForceResult:
    """
    計算風力（OCIMF 方法）。

    Args:
        wind_speed_ms: 風速 (m/s)
        wind_dir_deg:  風向 (度)
        vessel_area:   受風面積 (m²)
        cd:            風阻係數
        berth_angle:   碼頭方向 (度)
        rho:           空氣密度 (kg/m³)，預設使用標準大氣值

    Returns:
        WindForceResult dataclass
    """
    angle_diff = abs(float(wind_dir_deg) - float(berth_angle))
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    is_offshore = 45 < angle_diff < 135
    total_N     = 0.5 * rho * cd * vessel_area * float(wind_speed_ms) ** 2
    angle_rad   = math.radians(angle_diff)

    return WindForceResult(
        total_force_N        = total_N,
        transverse_force_N   = total_N * abs(math.cos(angle_rad)),
        longitudinal_force_N = total_N * abs(math.sin(angle_rad)),
        angle_diff           = angle_diff,
        is_offshore          = is_offshore,
    )


def calculate_mooring_capacity(
    num_bow_lines:          int,
    num_bow_spring_lines:   int,
    num_stern_lines:        int,
    num_stern_spring_lines: int,
    mbl_per_line:           float,
    safety_factor:          float,
) -> MooringCapacityResult:
    """
    計算纜繩容量。

    Args:
        mbl_per_line:  單根纜繩破斷負荷 (kN)
        safety_factor: 安全係數（WLL = MBL × SF）
    """
    wll             = mbl_per_line * safety_factor
    bow_kN          = num_bow_lines          * wll
    bow_spring_kN   = num_bow_spring_lines   * wll
    stern_kN        = num_stern_lines        * wll
    stern_spring_kN = num_stern_spring_lines * wll

    return MooringCapacityResult(
        total_capacity_kN        = bow_kN + bow_spring_kN + stern_kN + stern_spring_kN,
        bow_capacity_kN          = bow_kN,
        bow_spring_capacity_kN   = bow_spring_kN,
        stern_capacity_kN        = stern_kN,
        stern_spring_capacity_kN = stern_spring_kN,
        wll_per_line_kN          = wll,
    )


def calculate_tug_capacity(
    num_tugs: int, hp_per_tug: float
) -> TugCapacityResult:
    """
    計算拖船容量。

    Args:
        num_tugs:    拖船數量
        hp_per_tug:  每艘拖船馬力
    """
    bp_ton   = MOORING.tug_bollard_pull(hp_per_tug)
    eff_kN   = bp_ton * PHYSICS.gravity
    total_kN = num_tugs * eff_kN

    return TugCapacityResult(
        total_push_kN             = total_kN,
        bp_per_tug_ton            = bp_ton,
        effective_push_per_tug_kN = eff_kN,
    )


# ================= DataFrame 處理 =================

def _get_rec_attr(rec: Any, *candidates: str, default: Any = None) -> Any:
    """
    依序嘗試多個屬性名稱，回傳第一個存在且非 None 的值。

    用途：相容 AwtWeatherRecord（新版欄位名）與舊版 WeatherRecord（舊版欄位名）。

    Examples:
        # AwtWeatherRecord 有 wind_speed_kts，舊版有 wind_speed
        speed = _get_rec_attr(rec, "wind_speed_kts", "wind_speed", default=0.0)
    """
    for attr in candidates:
        val = getattr(rec, attr, None)
        if val is not None:
            return val
    return default


def normalize_dataframe(analyzer: Any, vessel_info: Any) -> pd.DataFrame:
    """
    從 WeatherAnalyzer 建立標準化 DataFrame，並預先計算風力欄位。

    變更說明：
      - 修正 Bug 2：原版直接存取 rec.wind_speed / rec.wind_gust /
        rec.wave_direction 等屬性，這些在 AwtWeatherRecord 中不存在
        （AWT 版欄位為 wind_speed_kts / wind_gust_kts / wave_direction）。
      - 改用 _get_rec_attr() 依序嘗試新舊欄位名稱，確保兩種記錄格式都能正確讀取。
      - 風力計算邏輯不變。

    Args:
        analyzer:    WeatherAnalyzer 實例（含 .data: List[AwtWeatherRecord]）
        vessel_info: VesselInfo 實例

    Returns:
        標準化 DataFrame
    """
    cd          = getattr(vessel_info, "wind_drag_coef", 1.0)
    berth_angle = vessel_info.berth_direction

    rows = []
    for rec in analyzer.data:

        # ── 修正：風速欄位相容新舊格式 ───────────────────────────────────────
        # AwtWeatherRecord：wind_speed_kts / wind_gust_kts（無 wind_speed）
        # 舊版 WeatherRecord：wind_speed / wind_gust（無 _kts 後綴）
        wind_speed_kts = _get_rec_attr(rec, "wind_speed_kts", "wind_speed", default=0.0)
        wind_gust_kts  = _get_rec_attr(rec, "wind_gust_kts",  "wind_gust",  default=0.0)

        # ── 修正：風速 m/s 欄位相容新舊格式 ──────────────────────────────────
        # AwtWeatherRecord 有 .wind_speed_ms property（由 kts 換算）
        # 舊版無此 property，改用 knots_to_ms() 換算
        wind_speed_ms = _get_rec_attr(rec, "wind_speed_ms", default=None)
        if wind_speed_ms is None:
            wind_speed_ms = knots_to_ms(wind_speed_kts)

        wind_gust_ms = _get_rec_attr(rec, "wind_gust_ms", default=None)
        if wind_gust_ms is None:
            wind_gust_ms = knots_to_ms(wind_gust_kts)

        # ── 修正：風向欄位相容新舊格式 ───────────────────────────────────────
        # AwtWeatherRecord：wind_direction（羅盤字串）+ wind_dir_deg（property）
        # 舊版 WeatherRecord：wind_direction（羅盤字串），無 wind_dir_deg
        wind_direction = _get_rec_attr(rec, "wind_direction", default="N")
        wind_dir_deg   = _get_rec_attr(rec, "wind_dir_deg",   default=None)
        if wind_dir_deg is None:
            wind_dir_deg = compass_to_degrees(wind_direction)

        # ── 修正：浪高欄位相容新舊格式 ───────────────────────────────────────
        # AwtWeatherRecord：wave_height（sig wave）/ wave_max / wave_period
        # 舊版：wave_height / wave_max / wave_period（相同名稱，直接相容）
        wave_height = _get_rec_attr(rec, "wave_height", "wave_sig_m", default=0.0)
        wave_max    = _get_rec_attr(rec, "wave_max",    "wave_max_m", default=0.0)
        wave_period = _get_rec_attr(rec, "wave_period", "wave_period_s", default=0.0)

        # ── 修正：波浪方向欄位相容新舊格式 ───────────────────────────────────
        # AwtWeatherRecord：wave_direction（羅盤字串）
        # 舊版：wave_direction（相同名稱，直接相容）
        wave_direction = _get_rec_attr(rec, "wave_direction", default="N")

        # ── 基礎氣象欄位 ──────────────────────────────────────────────────────
        row: Dict[str, Any] = {
            "time":           rec.time,
            "wind_direction": wind_direction,
            "wind_speed":     wind_speed_kts,   # 向下相容：保留舊欄位名
            "wind_gust":      wind_gust_kts,    # 向下相容：保留舊欄位名
            "wind_speed_kts": wind_speed_kts,
            "wind_gust_kts":  wind_gust_kts,
            "wind_speed_ms":  wind_speed_ms,
            "wind_gust_ms":   wind_gust_ms,
            "wind_dir_deg":   wind_dir_deg,
            "wave_direction": wave_direction,
            "wave_height":    wave_height,
            "wave_max":       wave_max,
            "wave_period":    wave_period,
            "wave_sig_m":     wave_height,
            "wave_max_m":     wave_max,
            "wave_period_s":  wave_period,
        }

        # ── 新增：AWT 額外欄位（舊版無此欄位，用 None 填充）────────────────
        # 讓下游模組（plotting、ui_components）能直接讀取，不需再 getattr
        row["visibility_nm"]       = _get_rec_attr(rec, "visibility_nm")
        row["pilot_visibility_nm"] = _get_rec_attr(rec, "pilot_visibility_nm")
        row["current_speed_ms"]    = _get_rec_attr(rec, "current_speed_ms")
        row["current_dir_deg"]     = _get_rec_attr(rec, "current_dir_deg")
        row["relative_humidity"]   = _get_rec_attr(rec, "relative_humidity")
        row["air_temp_c"]          = _get_rec_attr(rec, "air_temp_c")
        row["swell_height_m"]      = _get_rec_attr(rec, "swell_height_m")
        row["berthing_risk_score"] = _get_rec_attr(rec, "berthing_risk_score")
        row["berthing_risk_label"] = _get_rec_attr(rec, "berthing_risk_label")

        # ── 風力計算（每筆只算兩次：平均風速 + 陣風）────────────────────────
        avg_res  = calculate_wind_force(
            wind_speed_ms, wind_dir_deg, vessel_info.wind_area, cd, berth_angle)
        gust_res = calculate_wind_force(
            wind_gust_ms,  wind_dir_deg, vessel_info.wind_area, cd, berth_angle)

        row.update({
            "avg_force_N":   avg_res.total_force_N,
            "gust_force_N":  gust_res.total_force_N,
            "trans_force_N": avg_res.transverse_force_N,
            "long_force_N":  avg_res.longitudinal_force_N,
            "is_offshore":   avg_res.is_offshore,
            "risk_level":    classify_wind_risk(wind_speed_kts, wind_gust_kts),
        })
        rows.append(row)

    return pd.DataFrame(rows)


def style_dataframe_with_risk(df: pd.DataFrame):
    """為 DataFrame 套用風險等級背景色"""
    def _highlight(row: pd.Series):
        risk  = classify_wind_risk(row["wind_speed_kts"], row["wind_gust_kts"])
        color = get_risk_color(risk)
        return [f"background-color: {color}"] * len(row)

    return df.style.apply(_highlight, axis=1)
