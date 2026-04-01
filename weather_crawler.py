# weather_crawler.py  ── 整合優化版（含 WeatherParser）
"""
港口氣象資料爬蟲模組

包含：
  - AedynLoginManager  : Selenium 自動登入、Cookie 管理
  - WeatherDatabase    : SQLite 資料庫存取（支援 48h / 7d 雙資料表）
  - PortWeatherCrawler : 港口資料下載與管理
  - WeatherParser      : WNI 氣象文字解析（風浪 / 天氣狀況，含時區修正）

資料來源：WHL_all_ports_list.xlsx（萬海港口清單）

安全性設計：
  - 帳號密碼從環境變數讀取，不硬寫在程式碼中
  - Cookie 以 pickle 儲存（含過期檢查）
  - JWT Token 不寫入磁碟，僅保存在記憶體中

修正說明（v2.1）：
  - 時區轉換：統一以 UTC 儲存，顯示時才加上港口偏移，修正 JPYKK 大風漏判問題
  - 星號清洗：_safe_float() 在轉換前自動移除 WNI 警示標記（*, **）
  - WeatherParser 整合：下載後可直接解析為結構化 WeatherRecord / WeatherConditionRecord
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import re
import secrets
import sqlite3
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 設定區（全部可透過環境變數覆寫）
# ═══════════════════════════════════════════════════════════

DB_FILE        = Path(os.getenv("IWBDSS_DB_FILE",               "WNI_port_weather.db"))
EXCEL_FILE_WHL = Path(os.getenv("IWBDSS_EXCEL_WHL",             "WHL_all_ports_list.xlsx"))
COOKIE_FILE    = Path(os.getenv("IWBDSS_COOKIE_FILE",           "aedyn_cookies.pkl"))
TIMEOUT        = int(os.getenv("IWBDSS_TIMEOUT",                "30"))
MAX_RETRIES    = int(os.getenv("IWBDSS_MAX_RETRIES",             "3"))
COOKIE_MAX_AGE = timedelta(hours=int(os.getenv("IWBDSS_COOKIE_MAX_AGE_HOURS", "24")))

_AEDYN_BASE_URL = "https://aedyn.weathernews.com"
_AEDYN_USER_API = f"{_AEDYN_BASE_URL}/api/account/user"
_BASE_LOGIN_URL = (
    "https://idp.aedyn.wni.com/auth/realms/aedyn/protocol/openid-connect/auth"
    "?response_type=id_token%20token&scope=openid&client_id=aedyn"
    "&redirect_uri=https%3A%2F%2Faedyn.weathernews.com%2Fhttpd-auth%2Fredirect_uri"
)
_PORT_DATA_URL = (
    f"{_AEDYN_BASE_URL}/api/business/sea/portstatus/content/{{endpoint}}/{{station_id}}.txt"
)

# ── 風浪警戒閾值（可由 constant.py 覆寫）────────────────────
HIGH_WIND_SPEED_KTS: float = 20.0
HIGH_WIND_SPEED_BFT: int   = 6
HIGH_GUST_SPEED_KTS: float = 30.0
HIGH_GUST_SPEED_BFT: int   = 7
HIGH_WAVE_SIG:       float = 2.5

KTS_TO_MS: float = 0.514444   # 節 → 公尺/秒


def _build_login_url() -> str:
    """動態產生登入 URL（每次新的 state / nonce，避免 IdP 拒絕重複請求）"""
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    return f"{_BASE_LOGIN_URL}&state={state}&nonce={nonce}"


# ═══════════════════════════════════════════════════════════
# 通用工具函式
# ═══════════════════════════════════════════════════════════

def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    安全轉換為浮點數。

    - 移除 WNI 警示標記 '*'（例如 '20*' → 20.0）後嘗試轉換
    - 空字串、'-'、None 回傳 default
    - NaN 回傳 default

    Args:
        value  : 待轉換的值（字串、數字或其他型別）
        default: 轉換失敗時的預設值（預設 None）
    """
    if value is None:
        return default
    clean = str(value).replace("*", "").strip()
    if not clean or clean == "-":
        return default
    try:
        result = float(clean)
        return default if np.isnan(result) else result
    except (ValueError, TypeError):
        return default


def _safe_float_nn(value: Any, default: float = 0.0) -> float:
    """_safe_float 的非 None 版本，保證回傳 float（供風速等不允許 None 的欄位使用）"""
    result = _safe_float(value, default=default)
    return result if result is not None else default


def _kts_to_bft(kts: float) -> int:
    """節 → 蒲福風級"""
    thresholds = [1, 3, 6, 10, 16, 21, 27, 33, 40, 47, 55, 63]
    for bft, threshold in enumerate(thresholds):
        if kts < threshold:
            return bft
    return 12


def _wind_dir_deg(direction: str) -> float:
    """風向字串 → 角度（找不到時回傳 0.0）"""
    _DIR_MAP: Dict[str, float] = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
        "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
        "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
        "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
    }
    return _DIR_MAP.get(direction.strip().upper(), 0.0)


def _parse_utc_lct(
    utc_date: str,
    utc_time: str,
    local_date: str,
    local_time: str,
    base_year: int,
    lct_offset: Optional[timezone],
) -> Tuple[datetime, datetime, timezone]:
    """
    將 WNI 格式的 MMDD / HHMM 字串解析為帶時區的 datetime。

    ✅ 修正重點：
      - 統一以 UTC 儲存，顯示時才加上港口偏移
      - 跨年判斷：UTC 月份從 12 跳回 01 時，年份 +1
      - lct_offset 首次自動計算，後續複用（避免重複計算）

    Returns:
        (dt_utc, dt_lct, lct_offset)
    """
    dt_utc_naive = datetime.strptime(f"{base_year}{utc_date}{utc_time}", "%Y%m%d%H%M")
    dt_lct_naive = datetime.strptime(f"{base_year}{local_date}{local_time}", "%Y%m%d%H%M")

    # 第一筆資料時自動計算 LCT 偏移（整數小時）
    if lct_offset is None:
        diff_seconds  = (dt_lct_naive - dt_utc_naive).total_seconds()
        offset_hours  = int(diff_seconds / 3600)
        lct_offset    = timezone(timedelta(hours=offset_hours))

    dt_utc = dt_utc_naive.replace(tzinfo=timezone.utc)
    dt_lct = dt_lct_naive.replace(tzinfo=lct_offset)
    return dt_utc, dt_lct, lct_offset


def _parse_issued_time_dt(content: str) -> Optional[datetime]:
    """
    從氣象文字解析 ISSUED AT 時間，回傳帶 UTC 時區的 datetime。

    格式範例：ISSUED AT: 20260205 0000 UTC
    """
    for line in content.splitlines():
        if "ISSUED AT:" in line.upper():
            m = re.search(r"(\d{8})\s+(\d{4})", line)
            if m:
                try:
                    return datetime.strptime(
                        f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M"
                    ).replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
    return None


def _parse_issued_time(content: str) -> str:
    """
    從氣象文字解析 ISSUED AT 時間，回傳字串（供資料庫儲存用）。

    格式：ISSUED AT: 202506010600 UTC → '20260601_0600'
    找不到時回傳當前時間字串。
    """
    dt = _parse_issued_time_dt(content)
    if dt:
        return dt.strftime("%Y%m%d_%H%M")
    logger.warning("⚠️ 無法解析 ISSUED AT，使用當前時間作為 issued_time")
    return datetime.now().strftime("%Y%m%d%H%M")


# ═══════════════════════════════════════════════════════════
# 氣象解析資料結構
# ═══════════════════════════════════════════════════════════

@dataclass
class WeatherRecord:
    """氣象記錄資料結構（風浪資料）"""
    time:            datetime   # UTC 時間
    lct_time:        datetime   # LCT 當地時間
    wind_direction:  str        # 風向（例如: NNE）
    wind_speed_kts:  float      # 風速（knots）
    wind_gust_kts:   float      # 陣風（knots）
    wave_direction:  str        # 浪向
    wave_height:     float      # 顯著浪高（meters）
    wave_max:        float      # 最大浪高（meters）
    wave_period:     float      # 週期（seconds）

    def __post_init__(self) -> None:
        self.wind_speed_kts = float(self.wind_speed_kts)
        self.wind_gust_kts  = float(self.wind_gust_kts)
        self.wave_height    = float(self.wave_height)
        self.wave_max       = float(self.wave_max)
        self.wave_period    = float(self.wave_period)
        self.wind_direction = str(self.wind_direction).strip().upper()
        self.wave_direction = str(self.wave_direction).strip().upper()

    # ── 衍生屬性 ──────────────────────────────────────────────

    @property
    def wind_speed_ms(self) -> float:
        return self.wind_speed_kts * KTS_TO_MS

    @property
    def wind_speed_bft(self) -> int:
        return _kts_to_bft(self.wind_speed_kts)

    @property
    def wind_gust_ms(self) -> float:
        return self.wind_gust_kts * KTS_TO_MS

    @property
    def wind_gust_bft(self) -> int:
        return _kts_to_bft(self.wind_gust_kts)

    @property
    def wind_dir_deg(self) -> float:
        return _wind_dir_deg(self.wind_direction)

    @property
    def wave_dir_deg(self) -> float:
        return _wind_dir_deg(self.wave_direction)

    # 向下相容別名
    @property
    def wave_sig_m(self) -> float:
        return self.wave_height

    @property
    def wave_max_m(self) -> float:
        return self.wave_max

    @property
    def wave_period_s(self) -> float:
        return self.wave_period

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time":           self.time,
            "lct_time":       self.lct_time,
            "wind_direction": self.wind_direction,
            "wind_speed_kts": self.wind_speed_kts,
            "wind_speed_ms":  self.wind_speed_ms,
            "wind_speed_bft": self.wind_speed_bft,
            "wind_gust_kts":  self.wind_gust_kts,
            "wind_gust_ms":   self.wind_gust_ms,
            "wind_gust_bft":  self.wind_gust_bft,
            "wave_direction": self.wave_direction,
            "wave_height":    self.wave_height,
            "wave_max":       self.wave_max,
            "wave_period":    self.wave_period,
            "wind_dir_deg":   self.wind_dir_deg,
            "wave_dir_deg":   self.wave_dir_deg,
        }

    def __repr__(self) -> str:
        return (
            f"WeatherRecord("
            f"time={self.time.strftime('%Y-%m-%d %H:%M UTC')}, "
            f"LCT={self.lct_time.strftime('%H:%M')}, "
            f"wind={self.wind_direction} {self.wind_speed_kts:.1f}kts "
            f"(gust {self.wind_gust_kts:.1f}kts), "
            f"wave={self.wave_direction} {self.wave_height:.1f}m)"
        )


@dataclass
class WeatherConditionRecord:
    """天氣狀況記錄資料結構（溫度、降雨、氣壓、能見度等）"""
    time:          datetime
    lct_time:      datetime
    temperature:   Optional[float]   # °C，可能為 None
    precipitation: float             # mm/h
    pressure:      Optional[float]   # hPa，可能為 None
    visibility:    str
    weather_code:  str

    _WEATHER_DESCRIPTIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "CLR":      "晴朗",
            "FOG":      "霧",
            "MIST":     "薄霧",
            "HAZE":     "霾",
            "RAIN":     "雨",
            "DRIZZLE":  "毛毛雨",
            "SNOW":     "雪",
            "SLEET":    "雨夾雪",
            "THUNDER":  "雷暴",
            "CLOUDY":   "多雲",
            "CLDY":     "多雲",
            "OVERCAST": "陰天",
            "N/A":      "無資料",
        },
        init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        # 溫度：允許 None，排除物理不合理值
        if self.temperature is not None:
            val = _safe_float(self.temperature)
            self.temperature = val if (val is not None and -100 < val < 100) else None

        self.precipitation = _safe_float_nn(self.precipitation, default=0.0)

        # 氣壓：允許 None，排除物理不合理值
        if self.pressure is not None:
            val = _safe_float(self.pressure)
            self.pressure = val if (val is not None and 800 < val < 1100) else None

        self.visibility   = str(self.visibility).strip()
        self.weather_code = str(self.weather_code).strip().upper()

    # ── 能見度解析 ────────────────────────────────────────────

    _VIS_KM_RE  = re.compile(r"([\d.]+)\s*km", re.IGNORECASE)
    _VIS_NUM_RE = re.compile(r"([\d.]+)")

    @property
    def visibility_meters(self) -> Optional[float]:
        """
        將能見度字串解析為公尺。

        支援：'100' → 100m、'10km<' → 10000m、'2.5km' → 2500m
        """
        vis = self.visibility.replace("<", "").replace(">", "").strip()
        if not vis:
            return None
        m = self._VIS_KM_RE.search(vis)
        if m:
            return float(m.group(1)) * 1000
        m = self._VIS_NUM_RE.search(vis)
        if m:
            return float(m.group(1))
        return None

    @property
    def weather_description(self) -> str:
        return self._WEATHER_DESCRIPTIONS.get(self.weather_code, self.weather_code)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time":                self.time,
            "lct_time":            self.lct_time,
            "temperature":         self.temperature,
            "precipitation":       self.precipitation,
            "pressure":            self.pressure,
            "visibility":          self.visibility,
            "visibility_meters":   self.visibility_meters,
            "weather_code":        self.weather_code,
            "weather_description": self.weather_description,
        }

    def __repr__(self) -> str:
        return (
            f"WeatherConditionRecord("
            f"time={self.time.strftime('%Y-%m-%d %H:%M UTC')}, "
            f"LCT={self.lct_time.strftime('%H:%M')}, "
            f"temp={self.temperature}°C, "
            f"precip={self.precipitation}mm/h, "
            f"pressure={self.pressure}hPa, "
            f"vis={self.visibility}, wx={self.weather_code})"
        )


# ═══════════════════════════════════════════════════════════
# 港口資料結構
# ═══════════════════════════════════════════════════════════

@dataclass
class PortInfo:
    """單一港口的完整資訊（對應 WHL_all_ports_list.xlsx）"""
    whl_port_code: str
    wni_port_code: str
    port_name_en:  str
    port_name_zh:  str
    country:       str
    station_id:    str
    latitude:      float = 0.0
    longitude:     float = 0.0

    def display_str(self) -> str:
        return f"{self.whl_port_code} - {self.port_name_en} ({self.port_name_zh})"

    def to_dict(self) -> Dict[str, Any]:
        """轉為字典（供 get_port_info 回傳，相容 app_helpers）"""
        return {
            "port_name":     self.port_name_en,
            "port_name_en":  self.port_name_en,
            "port_name_zh":  self.port_name_zh,
            "port_code":     self.whl_port_code,
            "whl_port_code": self.whl_port_code,
            "wni_port_code": self.wni_port_code,
            "country":       self.country,
            "station_id":    self.station_id,
            "latitude":      self.latitude,
            "longitude":     self.longitude,
        }


# ═══════════════════════════════════════════════════════════
# 氣象解析器
# ═══════════════════════════════════════════════════════════

class WeatherParser:
    """
    WNI 氣象資料解析器（支援 48h 和 7d 預報）。

    ✅ 修正：
      - 時區統一以 UTC 儲存，本地時間由 LCT 欄位自動計算偏移
      - 風速 / 陣風的 '*' 標記在 _safe_float_nn() 中自動清除
      - 截止時間從「發布時間」起算，而非「現在」
    """

    # 資料行特徵：以 4 個數字區塊開頭（MMDD HHMM MMDD HHMM）
    _LINE_PATTERN      = re.compile(r"^\s*\d{4}\s+\d{4}\s+\d{4}\s+\d{4}")
    _WIND_BLOCK_KEY    = "WIND kts"
    _WEATHER_BLOCK_KEY = "2. WEATHER"

    # ── 公開 API ──────────────────────────────────────────────

    def detect_forecast_type(self, content: str) -> str:
        """自動偵測預報類型，回傳 '7d' 或 '48h'"""
        first_line = content.strip().split("\n")[0].upper()
        if any(kw in first_line for kw in ("7 DAY", "7-DAY", "7DAY")):
            return "7d"
        return "48h"

    def parse_content_48h(
        self, content: str
    ) -> Tuple[str, List[WeatherRecord], List[WeatherConditionRecord], List[str]]:
        """解析 48 小時預報（限制 48 小時內的資料）"""
        return self._parse(content, max_hours=48)

    def parse_content_7d(
        self, content: str
    ) -> Tuple[str, List[WeatherRecord], List[WeatherConditionRecord], List[str]]:
        """解析 7 天預報（不限制時間）"""
        return self._parse(content, max_hours=None)

    def parse_content(
        self,
        content: str,
        port_timezone: Optional[str] = None,
        max_hours: Optional[int] = 48,
    ) -> Tuple[str, List[WeatherRecord], List[WeatherConditionRecord], List[str]]:
        """通用解析入口（向下相容舊呼叫方式）"""
        return self._parse(content, max_hours=max_hours)

    def parse_file(
        self,
        file_path: str,
        forecast_type: str = "auto",
        encoding: str = "utf-8",
    ) -> Tuple[str, List[WeatherRecord], List[WeatherConditionRecord], List[str]]:
        """從檔案解析氣象資料（自動 fallback 編碼）"""
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()

        if forecast_type == "auto":
            forecast_type = self.detect_forecast_type(content)

        return self.parse_content_7d(content) if forecast_type == "7d" \
            else self.parse_content_48h(content)

    # ── 靜態分析工具 ──────────────────────────────────────────

    @staticmethod
    def filter_high_risk_records(
        records: List[WeatherRecord],
        wind_kts_threshold: float = HIGH_WIND_SPEED_KTS,
        wind_bft_threshold: int   = HIGH_WIND_SPEED_BFT,
        gust_kts_threshold: float = HIGH_GUST_SPEED_KTS,
        gust_bft_threshold: int   = HIGH_GUST_SPEED_BFT,
        wave_threshold:     float = HIGH_WAVE_SIG,
    ) -> List[WeatherRecord]:
        """篩選高風險時段（風浪超過閾值）"""
        return [
            r for r in records
            if r.wind_speed_kts >= wind_kts_threshold
            or r.wind_speed_bft >= wind_bft_threshold
            or r.wind_gust_kts  >= gust_kts_threshold
            or r.wind_gust_bft  >= gust_bft_threshold
            or r.wave_height    >= wave_threshold
        ]

    @staticmethod
    def get_statistics(records: List[WeatherRecord]) -> Dict[str, Any]:
        """計算風浪統計資訊"""
        if not records:
            return {}

        speeds_kts = [r.wind_speed_kts for r in records]
        speeds_ms  = [r.wind_speed_ms  for r in records]
        speeds_bft = [r.wind_speed_bft for r in records]
        gusts_kts  = [r.wind_gust_kts  for r in records]
        gusts_ms   = [r.wind_gust_ms   for r in records]
        gusts_bft  = [r.wind_gust_bft  for r in records]
        waves      = [r.wave_height    for r in records]

        return {
            "total_records": len(records),
            "time_range": {
                "start": min(r.time for r in records),
                "end":   max(r.time for r in records),
            },
            "wind": {
                "min_kts":      min(speeds_kts),
                "max_kts":      max(speeds_kts),
                "avg_kts":      mean(speeds_kts),
                "min_ms":       min(speeds_ms),
                "max_ms":       max(speeds_ms),
                "avg_ms":       mean(speeds_ms),
                "min_bft":      min(speeds_bft),
                "max_bft":      max(speeds_bft),
                "max_gust_kts": max(gusts_kts),
                "max_gust_ms":  max(gusts_ms),
                "max_gust_bft": max(gusts_bft),
            },
            "wave": {
                "min":      min(waves),
                "max":      max(waves),
                "avg":      mean(waves),
                "max_wave": max(r.wave_max for r in records),
            },
        }

    @staticmethod
    def get_weather_statistics(records: List[WeatherConditionRecord]) -> Dict[str, Any]:
        """計算天氣狀況統計資訊（temperature / pressure 為 Optional，統計前過濾 None）"""
        if not records:
            return {}

        temps     = [r.temperature   for r in records if r.temperature   is not None]
        pressures = [r.pressure      for r in records if r.pressure      is not None]
        precips   = [r.precipitation for r in records]

        temp_stats = (
            {"min": min(temps), "max": max(temps), "avg": mean(temps)}
            if temps else {"min": None, "max": None, "avg": None}
        )
        pressure_stats = (
            {"min": min(pressures), "max": max(pressures), "avg": mean(pressures)}
            if pressures else {"min": None, "max": None, "avg": None}
        )

        return {
            "total_records": len(records),
            "time_range": {
                "start": min(r.time for r in records),
                "end":   max(r.time for r in records),
            },
            "temperature":   temp_stats,
            "precipitation": {
                "total":       sum(precips),
                "max":         max(precips),
                "rainy_hours": sum(1 for p in precips if p > 0),
            },
            "pressure": pressure_stats,
            "weather_codes": {
                code: sum(1 for r in records if r.weather_code == code)
                for code in {r.weather_code for r in records}
            },
        }

    # ── 私有核心解析邏輯 ──────────────────────────────────────

    def _parse(
        self,
        content: str,
        max_hours: Optional[int],
    ) -> Tuple[str, List[WeatherRecord], List[WeatherConditionRecord], List[str]]:
        """
        核心解析方法。

        Returns:
            (port_name, wind_records, weather_records, warnings)
        """
        lines    = content.strip().split("\n")
        warnings: List[str] = []

        port_name   = self._parse_port_name(lines)
        issued_time = _parse_issued_time_dt(content)

        # ✅ 截止時間從「發布時間」起算，而非「現在」
        cutoff_time: Optional[datetime] = (
            issued_time + timedelta(hours=max_hours)
            if (issued_time and max_hours)
            else None
        )

        wind_records, warnings = self._parse_wind_section(lines, cutoff_time, warnings)
        weather_records, warnings = self._parse_weather_section(
            lines, cutoff_time,
            wind_records[0].lct_time.tzinfo if wind_records else None,
            warnings,
        )

        return port_name, wind_records, weather_records, warnings

    @staticmethod
    def _parse_port_name(lines: List[str]) -> str:
        for line in lines:
            if "PORT NAME" in line.upper():
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
        return "Unknown Port"

    def _parse_wind_section(
        self,
        lines: List[str],
        cutoff_time: Optional[datetime],
        warnings: List[str],
    ) -> Tuple[List[WeatherRecord], List[str]]:
        """解析 WINDS and WAVES 區段"""
        section_start = None
        for i, line in enumerate(lines):
            if self._WIND_BLOCK_KEY in line and "WAVE" in line:
                section_start = i + 2
                break

        if section_start is None:
            raise ValueError("找不到 WIND 資料區段 (WIND kts)")

        records: List[WeatherRecord] = []
        base_year  = datetime.now().year
        prev_month: Optional[int] = None
        lct_offset: Optional[timezone] = None

        for line in lines[section_start:]:
            line = line.strip()
            if not line or line.startswith(("*", "=")):
                break
            if not self._LINE_PATTERN.match(line):
                continue

            try:
                parts = line.split()
                if len(parts) < 11:
                    warnings.append(f"風浪欄位不足（需 ≥ 11）: {line}")
                    continue

                utc_date, utc_time = parts[0], parts[1]
                loc_date, loc_time = parts[2], parts[3]

                # ✅ 跨年：月份數字比較
                cur_month = int(utc_date[:2])
                if prev_month == 12 and cur_month == 1:
                    base_year += 1
                prev_month = cur_month

                dt_utc, dt_lct, lct_offset = _parse_utc_lct(
                    utc_date, utc_time, loc_date, loc_time, base_year, lct_offset
                )

                if cutoff_time and dt_utc > cutoff_time:
                    continue

                records.append(WeatherRecord(
                    time           = dt_utc,
                    lct_time       = dt_lct,
                    wind_direction = parts[4],
                    wind_speed_kts = _safe_float_nn(parts[5]),   # ✅ '*' 自動清除
                    wind_gust_kts  = _safe_float_nn(parts[6]),   # ✅ '*' 自動清除
                    wave_direction = parts[7],
                    wave_height    = _safe_float_nn(parts[8]),
                    wave_max       = _safe_float_nn(parts[9]),
                    wave_period    = _safe_float_nn(parts[10]),
                ))

            except Exception as exc:
                warnings.append(f"風浪解析失敗 [{line}]: {exc}")

        if not records:
            raise ValueError("未成功解析任何風浪資料")

        return records, warnings

    def _parse_weather_section(
        self,
        lines: List[str],
        cutoff_time: Optional[datetime],
        lct_offset: Optional[timezone],
        warnings: List[str],
    ) -> Tuple[List[WeatherConditionRecord], List[str]]:
        """解析 WEATHER 區段"""
        section_start = None
        for i, line in enumerate(lines):
            if self._WEATHER_BLOCK_KEY in line:
                for j in range(i + 1, min(i + 6, len(lines))):
                    if "deg" in lines[j] and "mm/h" in lines[j] and "hPa" in lines[j]:
                        section_start = j + 2
                        break
                break

        if section_start is None:
            warnings.append("⚠️ 未找到 WEATHER 資料區段")
            return [], warnings

        records: List[WeatherConditionRecord] = []
        base_year  = datetime.now().year
        prev_month: Optional[int] = None

        for line in lines[section_start:]:
            line = line.strip()
            if not line or line.startswith(("*", "=")):
                break
            if not self._LINE_PATTERN.match(line):
                continue

            try:
                parts = line.split()
                if len(parts) < 8:
                    warnings.append(f"天氣欄位不足（需 ≥ 8）: {line}")
                    continue

                utc_date, utc_time = parts[0], parts[1]
                loc_date, loc_time = parts[2], parts[3]

                cur_month = int(utc_date[:2])
                if prev_month == 12 and cur_month == 1:
                    base_year += 1
                prev_month = cur_month

                dt_utc, dt_lct, lct_offset = _parse_utc_lct(
                    utc_date, utc_time, loc_date, loc_time, base_year, lct_offset
                )

                if cutoff_time and dt_utc > cutoff_time:
                    continue

                records.append(WeatherConditionRecord(
                    time          = dt_utc,
                    lct_time      = dt_lct,
                    temperature   = _safe_float(parts[4], default=None),
                    precipitation = _safe_float_nn(parts[5]),
                    pressure      = _safe_float(parts[6], default=None),
                    visibility    = parts[7],
                    weather_code  = parts[8] if len(parts) > 8 else "N/A",
                ))

            except Exception as exc:
                warnings.append(f"天氣解析失敗 [{line}]: {exc}")

        return records, warnings


# ═══════════════════════════════════════════════════════════
# 登入管理器
# ═══════════════════════════════════════════════════════════

class AedynLoginManager:
    """
    負責自動登入 Aedyn 並管理 Cookie 與 JWT Token。

    帳號密碼優先從建構子參數取得，
    未傳入時從環境變數 AEDYN_USERNAME / AEDYN_PASSWORD 讀取。
    """

    def __init__(
        self,
        username:    str  = "",
        password:    str  = "",
        cookie_file: Path = COOKIE_FILE,
    ) -> None:
        self.username    = username or os.environ.get("AEDYN_USERNAME", "")
        self.password    = password or os.environ.get("AEDYN_PASSWORD", "")
        self.cookie_file = Path(cookie_file)
        self.cookies:          Dict[str, str]     = {}
        self.jwt_token:        str                = ""
        self.cookie_timestamp: Optional[datetime] = None

        if not self.username or not self.password:
            logger.warning("AEDYN_USERNAME / AEDYN_PASSWORD 未設定，登入功能將無法使用。")

    # ── Cookie 持久化 ────────────────────────────────────────

    def save_cookies(self) -> bool:
        try:
            data = {
                "cookies":   self.cookies,
                "jwt_token": self.jwt_token,
                "timestamp": datetime.now(),
            }
            self.cookie_file.write_bytes(pickle.dumps(data))
            logger.info("✅ Cookie 已儲存至 %s", self.cookie_file)
            return True
        except OSError:
            logger.warning("⚠️ Cookie 儲存失敗", exc_info=True)
            return False

    def load_cookies(self) -> bool:
        if not self.cookie_file.exists():
            logger.info("ℹ️ Cookie 檔案不存在：%s", self.cookie_file)
            return False
        try:
            data = pickle.loads(self.cookie_file.read_bytes())
            self.cookies          = data.get("cookies", {})
            self.jwt_token        = data.get("jwt_token", "")
            self.cookie_timestamp = data.get("timestamp")

            if self.cookie_timestamp:
                age = datetime.now() - self.cookie_timestamp
                logger.info(
                    "ℹ️ Cookie 年齡：%.1f 小時（建立於 %s）",
                    age.total_seconds() / 3600,
                    self.cookie_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                )
                if age > COOKIE_MAX_AGE:
                    logger.warning("⚠️ Cookie 已過期（超過 %s）", COOKIE_MAX_AGE)
                    return False

            logger.info("✅ 已載入 Cookie（數量：%d）", len(self.cookies))
            return True
        except Exception:
            logger.warning("⚠️ Cookie 載入失敗，將重新登入", exc_info=True)
            return False

    def verify_cookies(self) -> bool:
        if not self.cookies:
            return False
        try:
            resp = requests.get(
                _AEDYN_USER_API, headers=self.get_headers(), timeout=10, verify=False
            )
            if resp.status_code == 200:
                logger.info("✅ Cookie 有效，使用者：%s", resp.json().get("user_disp_name", "Unknown"))
                return True
            logger.warning("❌ Cookie 驗證失敗（HTTP %d）", resp.status_code)
            return False
        except requests.RequestException:
            logger.warning("❌ Cookie 驗證請求失敗", exc_info=True)
            return False

    # ── Selenium 登入 ────────────────────────────────────────

    def login_and_get_cookies(self, headless: bool = True) -> Dict[str, Any]:
        if not self.username or not self.password:
            raise RuntimeError("AEDYN_USERNAME / AEDYN_PASSWORD 未設定，無法登入。")

        driver = self._build_driver(headless)
        try:
            return self._execute_login(driver)
        except Exception as exc:
            _save_error_screenshot(driver, "login_error.png")
            raise RuntimeError(f"登入失敗：{exc}") from exc
        finally:
            driver.quit()

    def _build_driver(self, headless: bool) -> webdriver.Chrome:
        opts = webdriver.ChromeOptions()
        opts.add_argument("--start-maximized")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        if headless:
            opts.add_argument("--headless=new")
        return webdriver.Chrome(options=opts)

    def _execute_login(self, driver: webdriver.Chrome) -> Dict[str, Any]:
        wait = WebDriverWait(driver, 30)
        logger.info("🔐 正在登入 Aedyn...")
        driver.get(_build_login_url())

        try:
            user_el = wait.until(EC.visibility_of_element_located((By.ID, "username")))
            pwd_el  = wait.until(EC.visibility_of_element_located((By.ID, "password")))
            user_el.clear(); user_el.send_keys(self.username)
            pwd_el.clear();  pwd_el.send_keys(self.password)
            pwd_el.send_keys(Keys.ENTER)
            wait.until(
                lambda d: _AEDYN_BASE_URL in d.current_url
                and "redirect_uri" not in d.current_url
            )
            logger.info("✅ 登入成功")
        except TimeoutException:
            if _AEDYN_BASE_URL not in driver.current_url:
                raise RuntimeError("登入頁面等待超時，請確認帳號密碼是否正確")
            logger.info("✅ 偵測到已登入狀態，跳過輸入步驟")

        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(2)

        cookie_dict: Dict[str, str] = {c["name"]: c["value"] for c in driver.get_cookies()}
        logger.info("✅ 已從瀏覽器取得 %d 個 Cookie", len(cookie_dict))

        driver.get(_AEDYN_BASE_URL + "/")
        time.sleep(1)
        driver.get(_AEDYN_USER_API)
        time.sleep(1)

        for c in driver.get_cookies():
            cookie_dict[c["name"]] = c["value"]

        self.jwt_token = self._extract_jwt(driver, cookie_dict)
        self._verify_after_login(cookie_dict)

        self.cookies          = cookie_dict
        self.cookie_timestamp = datetime.now()
        self.save_cookies()

        return {"cookies": cookie_dict, "jwt_token": self.jwt_token}

    def _extract_jwt(self, driver: webdriver.Chrome, cookie_dict: Dict[str, str]) -> str:
        try:
            token = driver.execute_script(
                "return localStorage.getItem('jwt') || sessionStorage.getItem('jwt');"
            )
            if token:
                logger.info("✅ 已從 localStorage 取得 JWT Token（長度：%d）", len(token))
                return token
        except Exception:
            logger.debug("無法從 localStorage 取得 JWT Token", exc_info=True)

        if "jwt" in cookie_dict:
            token = cookie_dict["jwt"]
            logger.info("✅ 已從 Cookie 取得 JWT Token（長度：%d）", len(token))
            return token

        logger.warning("⚠️ 未能取得 JWT Token")
        return ""

    def _verify_after_login(self, cookie_dict: Dict[str, str]) -> None:
        try:
            resp = requests.get(
                _AEDYN_USER_API,
                headers=self._build_headers(cookie_dict),
                timeout=10, verify=False,
            )
            if resp.status_code == 200:
                logger.info("✅ Cookie 驗證成功，使用者：%s", resp.json().get("user_disp_name", "Unknown"))
            else:
                logger.warning("⚠️ 登入後 Cookie 驗證失敗（HTTP %d）", resp.status_code)
        except requests.RequestException:
            logger.warning("⚠️ 登入後 Cookie 驗證請求失敗", exc_info=True)

    # ── Headers 建構 ─────────────────────────────────────────

    def get_cookie_string(self) -> str:
        return "; ".join(f"{k}={v}" for k, v in self.cookies.items()) if self.cookies else ""

    def get_headers(self) -> Dict[str, str]:
        return self._build_headers(self.cookies)

    def _build_headers(self, cookie_dict: Dict[str, str]) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/143.0.0.0 Safari/537.36"
            ),
            "Accept":             "application/json, text/plain, */*",
            "Accept-Language":    "zh-TW,zh-CN;q=0.9,zh;q=0.8,en-US;q=0.7,en;q=0.6",
            "Referer":            f"{_AEDYN_BASE_URL}/",
            "sec-ch-ua":          '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
            "sec-ch-ua-mobile":   "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest":     "empty",
            "sec-fetch-mode":     "cors",
            "sec-fetch-site":     "same-origin",
        }
        if cookie_dict:
            headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookie_dict.items())
        if self.jwt_token:
            headers["json_web_token"] = self.jwt_token
        return headers


# ═══════════════════════════════════════════════════════════
# 資料庫
# ═══════════════════════════════════════════════════════════

class WeatherDatabase:
    """SQLite 氣象資料存取層（支援 48h / 7d 雙資料表）"""

    TABLE_48H = "weather_data"
    TABLE_7D  = "weather_data_7d"

    _DDL_TEMPLATE = """
        CREATE TABLE IF NOT EXISTS {table} (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            port_name_en  TEXT      NOT NULL,
            port_name_zh  TEXT      NOT NULL,
            wni_port_code TEXT      NOT NULL,
            whl_port_code TEXT,
            country       TEXT      NOT NULL,
            station_id    TEXT      NOT NULL,
            issued_time   TEXT      NOT NULL,
            content       TEXT      NOT NULL,
            download_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(whl_port_code, issued_time)
        )
    """
    _IDX_TEMPLATE = """
        CREATE INDEX IF NOT EXISTS idx_{table}_code
        ON {table} (whl_port_code, issued_time DESC)
    """

    def __init__(self, db_file: Path = DB_FILE) -> None:
        self.db_file = Path(db_file)
        self._conn   = sqlite3.connect(str(self.db_file), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_database()

    def _init_database(self) -> None:
        cur = self._conn.cursor()
        for table in (self.TABLE_48H, self.TABLE_7D):
            cur.execute(self._DDL_TEMPLATE.format(table=table))
            cur.execute(self._IDX_TEMPLATE.format(table=table))
        self._migrate_schema()
        self._conn.commit()

    def _migrate_schema(self) -> None:
        """自動遷移舊版 schema（port_name → port_name_en / port_name_zh）"""
        cur = self._conn.cursor()
        for table in (self.TABLE_48H, self.TABLE_7D):
            cur.execute(f"PRAGMA table_info({table})")
            existing_cols = {row[1] for row in cur.fetchall()}
            if "port_name_en" in existing_cols:
                continue
            logger.info("⏳ 偵測到舊版 schema，開始遷移資料表：%s", table)
            cur.execute(f"ALTER TABLE {table} ADD COLUMN port_name_en TEXT NOT NULL DEFAULT ''")
            cur.execute(f"ALTER TABLE {table} ADD COLUMN port_name_zh TEXT NOT NULL DEFAULT ''")
            if "port_name" in existing_cols:
                cur.execute(f"UPDATE {table} SET port_name_en = port_name WHERE port_name_en = ''")
                logger.info("✅ 已將 port_name 資料複製至 port_name_en（table=%s）", table)
            logger.info("✅ 資料表遷移完成：%s", table)

    def close(self) -> None:
        self._conn.close()

    # ── 私有共用方法 ─────────────────────────────────────────

    def _get_latest_time(self, table: str, whl_port_code: str) -> Optional[str]:
        cur = self._conn.execute(
            f"SELECT issued_time FROM {table} "
            "WHERE whl_port_code = ? ORDER BY issued_time DESC LIMIT 1",
            (whl_port_code,),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def _get_latest_content(
        self, table: str, whl_port_code: str
    ) -> Optional[Tuple[str, str, str, str]]:
        cur = self._conn.execute(
            f"SELECT content, issued_time, port_name_en, port_name_zh "
            f"FROM {table} WHERE whl_port_code = ? "
            "ORDER BY issued_time DESC LIMIT 1",
            (whl_port_code,),
        )
        row = cur.fetchone()
        return tuple(row) if row else None  # type: ignore[return-value]

    def _save(self, table: str, port: PortInfo, issued_time: str, content: str) -> bool:
        try:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {table} "
                "(port_name_en, port_name_zh, wni_port_code, whl_port_code, "
                "country, station_id, issued_time, content, download_time) "
                "VALUES (?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)",
                (
                    port.port_name_en, port.port_name_zh,
                    port.wni_port_code, port.whl_port_code,
                    port.country, port.station_id,
                    issued_time, content,
                ),
            )
            self._conn.commit()
            return True
        except sqlite3.Error:
            logger.error(
                "❌ 資料庫寫入失敗（table=%s, port=%s）",
                table, port.whl_port_code, exc_info=True,
            )
            return False

    # ── 公開 API — 48h ───────────────────────────────────────

    def get_latest_time(self, whl_port_code: str) -> Optional[str]:
        return self._get_latest_time(self.TABLE_48H, whl_port_code)

    def get_latest_content(self, whl_port_code: str) -> Optional[Tuple[str, str, str, str]]:
        return self._get_latest_content(self.TABLE_48H, whl_port_code)

    def save_weather(self, port: PortInfo, issued_time: str, content: str) -> bool:
        return self._save(self.TABLE_48H, port, issued_time, content)

    # ── 公開 API — 7d ────────────────────────────────────────

    def get_latest_time_7d(self, whl_port_code: str) -> Optional[str]:
        return self._get_latest_time(self.TABLE_7D, whl_port_code)

    def get_latest_content_7d(self, whl_port_code: str) -> Optional[Tuple[str, str, str, str]]:
        return self._get_latest_content(self.TABLE_7D, whl_port_code)

    def save_weather_7d(self, port: PortInfo, issued_time: str, content: str) -> bool:
        return self._save(self.TABLE_7D, port, issued_time, content)


# ═══════════════════════════════════════════════════════════
# 爬蟲主體
# ═══════════════════════════════════════════════════════════

class PortWeatherCrawler:
    """
    港口氣象資料爬蟲。

    職責：
      - 從 WHL_all_ports_list.xlsx 載入港口清單
      - 管理登入狀態（智能 Cookie 復用）
      - 下載並儲存 48h / 7d 氣象資料
      - 透過內建 WeatherParser 提供結構化解析介面
    """

    def __init__(
        self,
        username:   str  = "",
        password:   str  = "",
        excel_path: Path = EXCEL_FILE_WHL,
        auto_login: bool = False,
    ) -> None:
        self.excel_path    = Path(excel_path)
        self.db            = WeatherDatabase()
        self.session       = self._create_session()
        self._port_map:    Dict[str, PortInfo] = {}
        self.login_manager = AedynLoginManager(username, password)
        self.headers:      Dict[str, str] = {}
        self.parser        = WeatherParser()   # ✅ 內建解析器

        self._load_port_map()
        self._smart_login(force_login=auto_login)

    # ── 屬性 ──────────────────────────────────────────────────

    @property
    def port_list(self) -> List[str]:
        return list(self._port_map.keys())

    # ── 初始化 ────────────────────────────────────────────────

    @staticmethod
    def _create_session() -> requests.Session:
        session = requests.Session()
        retry   = Retry(
            total=MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    def _load_port_map(self) -> None:
        if not self.excel_path.exists():
            logger.warning("⚠️ 找不到 Excel 檔案：%s", self.excel_path)
            return
        try:
            logger.info("⏳ 正在載入港口資料...")
            df = pd.read_excel(self.excel_path, sheet_name="all_ports_list")
            df.columns = df.columns.str.strip()

            loaded = 0
            for _, row in df.iterrows():
                code   = str(row.get("Port_Code_5", "")).strip()
                obj_id = str(row.get("Station ID (Object_ID)", "")).strip()
                if not code or code == "nan" or not obj_id or obj_id == "nan":
                    continue

                self._port_map[code] = PortInfo(
                    whl_port_code = code,
                    wni_port_code = str(row.get("WNI Port Code", code)).strip(),
                    port_name_en  = str(row.get("Port Name(English)", "")).strip(),
                    port_name_zh  = str(row.get("Port Name(Chinese)", "")).strip(),
                    country       = str(row.get("Country", "N/A")).strip(),
                    station_id    = obj_id,
                    latitude      = _safe_float_nn(row.get("Lat")),
                    longitude     = _safe_float_nn(row.get("Lon")),
                )
                loaded += 1

            logger.info("✅ 已載入 %d 個港口資料", loaded)
        except Exception:
            logger.exception("❌ 讀取 Excel 失敗：%s", self.excel_path)

    def _smart_login(self, force_login: bool = False) -> None:
        if force_login:
            logger.info("🔄 強制重新登入")
            self.refresh_cookies()
            return

        logger.info("🔍 檢查 Cookie 狀態...")
        if self.login_manager.load_cookies() and self.login_manager.verify_cookies():
            logger.info("✅ 使用已儲存的 Cookie")
            self.headers = self.login_manager.get_headers()
            return

        logger.info("🔐 Cookie 不存在或已失效，執行登入流程")
        self.refresh_cookies()

    # ── 公開 API ──────────────────────────────────────────────

    def refresh_cookies(self, headless: bool = True) -> bool:
        try:
            result = self.login_manager.login_and_get_cookies(headless=headless)
            self.headers = self.login_manager.get_headers()
            logger.info(
                "✅ Cookie 已更新（數量：%d，JWT：%s）",
                len(result["cookies"]),
                "✅ 已取得" if result["jwt_token"] else "❌ 未取得",
            )
            return True
        except RuntimeError:
            logger.error("❌ Cookie 更新失敗", exc_info=True)
            return False

    def get_all_ports_display(self) -> List[str]:
        return [info.display_str() for info in self._port_map.values()]

    def get_port_info(self, whl_port_code: str) -> Optional[Dict[str, Any]]:
        port = self._port_map.get(whl_port_code)
        if port is None:
            logger.warning("❌ 港口代碼 %s 不在 port_map 中", whl_port_code)
            return None
        return port.to_dict()

    def get_data_from_db(self, whl_port_code: str) -> Optional[Tuple[str, str, str, str]]:
        """從資料庫讀取 48h 最新氣象內容（content, issued_time, name_en, name_zh）"""
        return self.db.get_latest_content(whl_port_code)

    def get_data_from_db_7d(self, whl_port_code: str) -> Optional[Tuple[str, str, str, str]]:
        """從資料庫讀取 7d 最新氣象內容（content, issued_time, name_en, name_zh）"""
        return self.db.get_latest_content_7d(whl_port_code)

    # ── 解析介面（直接回傳結構化資料）────────────────────────

    def parse_port_data(
        self,
        whl_port_code: str,
        endpoint: str = "48h",
    ) -> Optional[Tuple[str, List[WeatherRecord], List[WeatherConditionRecord], List[str]]]:
        """
        從資料庫讀取原始內容並立即解析為結構化資料。

        ✅ 修正重點：
          - 解析時使用 UTC 時間，不再依賴 LCT 字串做計算
          - 風速 '*' 標記在 _safe_float_nn() 中自動清除

        Args:
            whl_port_code: 港口代碼
            endpoint     : '48h' 或 '7d'

        Returns:
            (port_name, wind_records, weather_records, warnings) 或 None
        """
        if endpoint == "48h":
            row = self.db.get_latest_content(whl_port_code)
        else:
            row = self.db.get_latest_content_7d(whl_port_code)

        if row is None:
            logger.warning("⚠️ 資料庫無 %s 的 %s 資料", whl_port_code, endpoint)
            return None

        content = row[0]
        try:
            if endpoint == "7d":
                return self.parser.parse_content_7d(content)
            return self.parser.parse_content_48h(content)
        except Exception:
            logger.exception("❌ 解析失敗（port=%s, endpoint=%s）", whl_port_code, endpoint)
            return None

    def parse_port_data_48h(
        self, whl_port_code: str
    ) -> Optional[Tuple[str, List[WeatherRecord], List[WeatherConditionRecord], List[str]]]:
        """從資料庫讀取並解析 48h 預報"""
        return self.parse_port_data(whl_port_code, "48h")

    def parse_port_data_7d(
        self, whl_port_code: str
    ) -> Optional[Tuple[str, List[WeatherRecord], List[WeatherConditionRecord], List[str]]]:
        """從資料庫讀取並解析 7d 預報"""
        return self.parse_port_data(whl_port_code, "7d")

    def get_high_risk_periods(
        self,
        whl_port_code: str,
        endpoint: str = "48h",
        wind_kts_threshold: float = HIGH_WIND_SPEED_KTS,
        gust_kts_threshold: float = HIGH_GUST_SPEED_KTS,
        wave_threshold:     float = HIGH_WAVE_SIG,
    ) -> List[WeatherRecord]:
        """
        取得指定港口的高風險時段清單。

        ✅ 此方法整合了時區修正與 '*' 清除，確保 JPYKK 等大風港口不會漏判。

        Args:
            whl_port_code      : 港口代碼
            endpoint           : '48h' 或 '7d'
            wind_kts_threshold : 風速閾值（kts）
            gust_kts_threshold : 陣風閾值（kts）
            wave_threshold     : 浪高閾值（m）

        Returns:
            高風險 WeatherRecord 清單（以 UTC 時間排序）
        """
        result = self.parse_port_data(whl_port_code, endpoint)
        if result is None:
            return []

        _, wind_records, _, _ = result
        return WeatherParser.filter_high_risk_records(
            wind_records,
            wind_kts_threshold=wind_kts_threshold,
            gust_kts_threshold=gust_kts_threshold,
            wave_threshold=wave_threshold,
        )

    # ── 下載核心（私有）─────────────────────────────────────

    def _fetch_port_data(
        self,
        whl_port_code: str,
        endpoint: str,
        retry_login: bool = True,
    ) -> Tuple[bool, str]:
        """通用下載邏輯，由公開方法包裝呼叫"""
        port = self._port_map.get(whl_port_code)
        if port is None:
            return False, f"找不到港口代碼：{whl_port_code}"

        url   = _PORT_DATA_URL.format(endpoint=endpoint, station_id=port.station_id)
        label = "48小時" if endpoint == "48h" else "7天"
        logger.info(
            "📡 下載 %s（%s / %s）- %s 預報",
            whl_port_code, port.port_name_en, port.port_name_zh, label,
        )

        try:
            resp = self.session.get(
                url, headers=self.headers, verify=False, timeout=TIMEOUT
            )
        except requests.exceptions.Timeout:
            return False, f"連線逾時（超過 {TIMEOUT} 秒）"
        except requests.RequestException as exc:
            return False, f"連線錯誤：{exc}"

        if resp.status_code == 200:
            return self._handle_success(port, resp.text, endpoint)

        if resp.status_code in (401, 403):
            if retry_login:
                logger.warning("⚠️ Cookie 已過期（HTTP %d），嘗試重新登入", resp.status_code)
                if self.refresh_cookies():
                    return self._fetch_port_data(whl_port_code, endpoint, retry_login=False)
            return False, f"重新登入後仍無法存取（HTTP {resp.status_code}）"

        return False, f"下載失敗（HTTP {resp.status_code}）"

    def _handle_success(
        self, port: PortInfo, content: str, endpoint: str
    ) -> Tuple[bool, str]:
        """處理成功下載的回應：比對版本並儲存"""
        issued_time = _parse_issued_time(content)

        if endpoint == "48h":
            cached_time = self.db.get_latest_time(port.whl_port_code)
            save_fn     = self.db.save_weather
        else:
            cached_time = self.db.get_latest_time_7d(port.whl_port_code)
            save_fn     = self.db.save_weather_7d

        if cached_time == issued_time:
            return True, f"{endpoint} 資料已是最新（{issued_time}）"

        if save_fn(port, issued_time, content):
            return True, f"{endpoint} 更新成功（{issued_time}）"
        return False, "資料庫寫入失敗"

    # ── 批次下載（私有）─────────────────────────────────────

    def _fetch_all_ports(self, endpoint: str) -> Dict[str, int]:
        label  = "48小時" if endpoint == "48h" else "7天"
        total  = len(self.port_list)
        counts = {"success": 0, "skipped": 0, "failed": 0}

        logger.info("🚀 開始批次下載 %d 個港口（%s）", total, label)

        for i, code in enumerate(self.port_list, 1):
            ok, msg = self._fetch_port_data(code, endpoint)
            logger.info("[%d/%d] %s：%s", i, total, code, msg)

            if not ok:
                counts["failed"] += 1
            elif "已是最新" in msg:
                counts["skipped"] += 1
            else:
                counts["success"] += 1

        logger.info(
            "📊 %s 批次下載完成 — ✅ 成功：%d  ⏭️ 略過：%d  ❌ 失敗：%d",
            label, counts["success"], counts["skipped"], counts["failed"],
        )
        return counts

    # ── 公開下載 API ─────────────────────────────────────────

    def fetch_port_data(
        self, whl_port_code: str, retry_login: bool = True
    ) -> Tuple[bool, str]:
        """下載指定港口 48 小時預報"""
        return self._fetch_port_data(whl_port_code, "48h", retry_login)

    def fetch_port_data_7d(
        self, whl_port_code: str, retry_login: bool = True
    ) -> Tuple[bool, str]:
        """下載指定港口 7 天預報"""
        return self._fetch_port_data(whl_port_code, "7d", retry_login)

    def fetch_all_ports(self) -> Dict[str, int]:
        """批次下載所有港口 48 小時預報"""
        return self._fetch_all_ports("48h")

    def fetch_all_ports_7d(self) -> Dict[str, int]:
        """批次下載所有港口 7 天預報"""
        return self._fetch_all_ports("7d")

    def fetch_all_ports_both(self) -> Dict[str, Dict[str, int]]:
        """批次下載所有港口的 48h + 7d 預報"""
        total  = len(self.port_list)
        s_48h: Dict[str, int] = {"success": 0, "skipped": 0, "failed": 0}
        s_7d:  Dict[str, int] = {"success": 0, "skipped": 0, "failed": 0}

        logger.info("🚀 開始批次下載 %d 個港口（48h + 7d）", total)

        for i, code in enumerate(self.port_list, 1):
            for ep, stats in (("48h", s_48h), ("7d", s_7d)):
                ok, msg = self._fetch_port_data(code, ep)
                logger.info("[%d/%d] %s %s：%s", i, total, code, ep, msg)
                if not ok:
                    stats["failed"] += 1
                elif "已是最新" in msg:
                    stats["skipped"] += 1
                else:
                    stats["success"] += 1

        logger.info("📊 批次下載完成 — 48h：%s / 7d：%s", s_48h, s_7d)
        return {"48h": s_48h, "7d": s_7d}

    def test_api_connection(self) -> None:
        """測試 API 連線與認證狀態（除錯用）"""
        logger.info("🧪 測試 API 連線...")
        for url in [_AEDYN_USER_API, f"{_AEDYN_BASE_URL}/"]:
            try:
                resp = self.session.get(
                    url, headers=self.headers, verify=False, timeout=10
                )
                if resp.status_code == 200:
                    if "application/json" in resp.headers.get("Content-Type", ""):
                        preview = json.dumps(resp.json(), ensure_ascii=False)[:200]
                        logger.info("✅ %s → %s", url, preview)
                    else:
                        logger.info("✅ %s → %d bytes", url, len(resp.text))
                else:
                    logger.warning("❌ %s → HTTP %d", url, resp.status_code)
            except requests.RequestException:
                logger.error("❌ %s → 連線失敗", url, exc_info=True)


# ═══════════════════════════════════════════════════════════
# 私有輔助函式
# ═══════════════════════════════════════════════════════════

def _save_error_screenshot(driver: webdriver.Chrome, path: str) -> None:
    """儲存錯誤截圖（失敗時靜默忽略）"""
    try:
        driver.save_screenshot(path)
        logger.info("📸 錯誤截圖已儲存：%s（當前網址：%s）", path, driver.current_url)
    except Exception:
        logger.debug("無法儲存錯誤截圖", exc_info=True)


# ═══════════════════════════════════════════════════════════
# 使用範例
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _USERNAME = os.getenv("AEDYN_USERNAME", "your_username@example.com")
    _PASSWORD = os.getenv("AEDYN_PASSWORD", "your_password")

    logger.info("=" * 60)
    logger.info("初始化爬蟲系統（WHL 港口清單）")
    logger.info("=" * 60)

    crawler = PortWeatherCrawler(
        username=_USERNAME,
        password=_PASSWORD,
        auto_login=False,
    )
    crawler.test_api_connection()

    # ── 範例 1: 下載單一港口 48h 預報
    ok, msg = crawler.fetch_port_data("JPYKK")
    logger.info("JPYKK 48h 下載結果：%s", msg)

    # ── 範例 2: 解析並取得結構化資料（修正時區 + * 標記）
    result = crawler.parse_port_data_48h("JPYKK")
    if result:
        port_name, winds, wxs, warns = result
        logger.info("港口：%s，風浪記錄：%d 筆，天氣記錄：%d 筆", port_name, len(winds), len(wxs))

        # 統計
        stats = WeatherParser.get_statistics(winds)
        if stats:
            logger.info(
                "風速範圍：%.1f ~ %.1f kts，最大陣風：%.1f kts",
                stats["wind"]["min_kts"],
                stats["wind"]["max_kts"],
                stats["wind"]["max_gust_kts"],
            )

        # 顯示前 3 筆（UTC + LCT 對照）
        for r in winds[:3]:
            logger.info("  %r", r)

        for w in warns:
            logger.warning("  ⚠️ %s", w)

    # ── 範例 3: 取得高風險時段（修正後不再漏判大風）
    high_risk = crawler.get_high_risk_periods("JPYKK", endpoint="48h")
    logger.info("JPYKK 高風險時段：%d 筆", len(high_risk))
    for r in high_risk[:5]:
        logger.info(
            "  UTC %s / LCT %s  風速 %.1f kts（陣風 %.1f kts）",
            r.time.strftime("%m/%d %H:%M"),
            r.lct_time.strftime("%m/%d %H:%M"),
            r.wind_speed_kts,
            r.wind_gust_kts,
        )

    # ── 範例 4: 下載 7d 並解析
    ok, msg = crawler.fetch_port_data_7d("JPYKK")
    logger.info("JPYKK 7d 下載結果：%s", msg)

    result_7d = crawler.parse_port_data_7d("JPYKK")
    if result_7d:
        _, winds_7d, _, _ = result_7d
        logger.info("7d 風浪記錄：%d 筆", len(winds_7d))

    # ── 範例 5: 取得港口資訊
    port_info = crawler.get_port_info("JPYKK")
    if port_info:
        logger.info("港口資訊：%s", json.dumps(port_info, ensure_ascii=False))

    # ── 範例 6: 批次下載（取消註解執行）
    # stats_both = crawler.fetch_all_ports_both()

