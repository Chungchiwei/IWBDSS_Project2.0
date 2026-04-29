# awt_crawler.py
# StormGeo (AWT) s-Insight Port Forecast API 客戶端

import base64
import logging
import time
import urllib3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# ================= 常數 =================

DEFAULT_API_BASE      = "https://api-shipping.stormgeo.com/api/v1"
AWT_API_BASE          = DEFAULT_API_BASE
TOKEN_EXPIRY_MINUTES  = 4
REQUEST_TIMEOUT       = 30
MAX_RETRIES           = 3
RETRY_BACKOFF         = 1.5
FORECAST_DAYS_DEFAULT = 3
SSL_VERIFY            = False

_SESSION_WARMUP_DELAY = 1.5

_WARMUP_ENDPOINTS = [
    '/ports',
    '/user/profile',
    '/dashboard',
    '',
]

_DEG_TO_DIR: List[Tuple[float, str]] = [
    (11.25,  'N'),   (33.75,  'NNE'), (56.25,  'NE'),  (78.75,  'ENE'),
    (101.25, 'E'),   (123.75, 'ESE'), (146.25, 'SE'),  (168.75, 'SSE'),
    (191.25, 'S'),   (213.75, 'SSW'), (236.25, 'SW'),  (258.75, 'WSW'),
    (281.25, 'W'),   (303.75, 'WNW'), (326.25, 'NW'),  (348.75, 'NNW'),
]


def _deg_to_compass(degrees: Optional[float]) -> str:
    if degrees is None:
        return 'N/A'
    d = degrees % 360
    for threshold, label in _DEG_TO_DIR:
        if d < threshold:
            return label
    return 'N'


def _build_basic_auth_header(username: str, password: str) -> str:
    credentials = f"{username}:{password}"
    encoded = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
    return f"Basic {encoded}"


def _build_session(max_retries: int = MAX_RETRIES,
                   backoff: float = RETRY_BACKOFF,
                   verify_ssl: bool = SSL_VERIFY) -> requests.Session:
    session = requests.Session()
    session.verify = verify_ssl
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


# ================= 登入管理器 =================

class AwtLoginManager:

    def __init__(self, username: str, password: str,
                 api_base: str = DEFAULT_API_BASE,
                 verify_ssl: bool = SSL_VERIFY):
        self.username   = username
        self.password   = password
        self.api_base   = api_base.rstrip('/')
        self.verify_ssl = verify_ssl
        self._token: Optional[str]             = None
        self._token_expiry: Optional[datetime] = None
        # ✅ 唯一的 Session，登入 Cookie 與後續 API 請求共用
        self._session = _build_session(verify_ssl=verify_ssl)

    def _is_token_valid(self) -> bool:
        return bool(
            self._token
            and self._token_expiry
            and datetime.now(timezone.utc) < self._token_expiry
        )

    def get_token(self) -> str:
        if not self._is_token_valid():
            logger.info("AWT Token 不存在或已過期，正在重新登入...")
            self._login()
        return self._token  # type: ignore[return-value]

    def _login(self) -> None:
        url = f"{self.api_base}/auth/login"
        headers = {
            "Authorization": _build_basic_auth_header(self.username, self.password),
            "Content-Type": "application/json",
        }
        try:
            resp = self._session.post(url, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data  = resp.json()
            token = (
                data.get('token')
                or data.get('accessToken')
                or data.get('access_token')
                or data.get('jwt')
            )
            if not token:
                raise RuntimeError(
                    f"AWT 登入成功但找不到 Token 欄位，回應: {data}")

            self._token        = token
            self._token_expiry = (datetime.now(timezone.utc)
                                  + timedelta(minutes=TOKEN_EXPIRY_MINUTES))
            logger.info("✅ AWT 登入成功，Token 有效至 %s UTC",
                        self._token_expiry.strftime('%H:%M:%S'))

            # 登入後立即暖機，建立 Web Session
            self._activate_web_session()

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 'N/A'
            body   = e.response.text[:300]  if e.response is not None else ''
            raise RuntimeError(f"AWT 登入失敗 (HTTP {status}): {body}") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"AWT 登入網路錯誤: {e}") from e

    def _activate_web_session(self) -> None:
        """
        登入後暖機：依序嘗試多個端點，確認 Session 建立。
        """
        if not self._token:
            return

        headers = {
            'Authorization': f'Bearer {self._token}',
            'Accept':        'application/json',
            'Content-Type':  'application/json',
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/124.0.0.0 Safari/537.36'
            ),
        }

        warmup_success = False
        for endpoint in _WARMUP_ENDPOINTS:
            url = f"{self.api_base}{endpoint}"
            try:
                resp = self._session.get(
                    url, headers=headers,
                    timeout=10, verify=self.verify_ssl,
                )
                if resp.status_code in (200, 201, 204):
                    logger.info(
                        "✅ Web Session 暖機成功（端點: %s，狀態: %d）",
                        endpoint or '/', resp.status_code,
                    )
                    warmup_success = True
                    break
                else:
                    logger.debug(
                        "   暖機端點 %s 回應 %d，嘗試下一個",
                        endpoint or '/', resp.status_code,
                    )
            except requests.exceptions.RequestException as e:
                logger.debug("   暖機端點 %s 失敗: %s", endpoint or '/', e)

        if not warmup_success:
            logger.warning(
                "⚠️  Web Session 暖機端點均無回應，改用延遲 5.0s 作為 fallback"
            )
            time.sleep(5.0)
            return

        # 暖機成功後等待，確保後端 Session 完全建立
        logger.debug("   等待 Session 建立（3.0s）...")
        time.sleep(3.0)

        # 二次確認：用真實港口請求驗證 Session 可用
        self._verify_session_with_real_request(headers)

    def _verify_session_with_real_request(
            self, headers: Dict[str, str]) -> None:
        """
        暖機後用已知港口（新加坡 SGSIN）發一次真實 forecast 請求，
        確認 Session 真正可用。失敗時最多重試 3 次，每次間隔遞增。
        ✅ 修正：移除重複定義，只保留此有重試邏輯的版本。
        """
        verify_url = f"{self.api_base}/ports/SGSIN/forecasts"

        for attempt in range(1, 4):
            try:
                resp = self._session.get(
                    verify_url, headers=headers,
                    params={"Days": 1},
                    timeout=15, verify=self.verify_ssl,
                )
                if resp.status_code == 200:
                    logger.info(
                        "✅ Session 驗證成功（SGSIN 測試請求 200，嘗試 %d 次）",
                        attempt,
                    )
                    return
                elif resp.status_code == 401:
                    logger.warning(
                        "   Session 驗證 401，重新取得 Token（嘗試 %d）...",
                        attempt,
                    )
                    # ✅ 直接更新 header，不遞迴呼叫 _login() 避免無限循環
                    self._token        = None
                    self._token_expiry = None
                    self.get_token()
                    headers = {
                        'Authorization': f'Bearer {self._token}',
                        'Accept':        'application/json',
                        'Content-Type':  'application/json',
                    }
                else:
                    wait = 2.0 * attempt
                    logger.warning(
                        "   Session 驗證回應 %d（嘗試 %d），等待 %.1fs...",
                        resp.status_code, attempt, wait,
                    )
                    time.sleep(wait)

            except requests.exceptions.RequestException as e:
                wait = 2.0 * attempt
                logger.warning(
                    "   Session 驗證請求失敗（嘗試 %d）: %s，等待 %.1fs...",
                    attempt, e, wait,
                )
                time.sleep(wait)

        logger.warning("⚠️  Session 驗證 3 次均失敗，繼續執行（可能有部分 404）")

    def get_auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.get_token()}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
            "User-Agent": (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/124.0.0.0 Safari/537.36'
            ),
        }

    def log_rate_limit(self, response: requests.Response) -> None:
        limit     = response.headers.get('X-RateLimit-Limit')
        remaining = response.headers.get('X-RateLimit-Remaining')
        reset     = response.headers.get('X-RateLimit-Reset')
        if limit or remaining:
            logger.debug("Rate Limit → 上限: %s，剩餘: %s，重置: %s",
                         limit, remaining, reset)


# ================= Port Forecast 解析 =================
#
# 實際 API 回傳結構（已確認）：
# [
#   {
#     "unLocode": "TWKHH",
#     "portForecast": {
#       "wind":        { "direction": 262, "speed": 5.44, "gust": 8.16 },
#                                          ↑ 單位：knots（已由網站資料交叉驗證確認）
#       "temperature": { "air": 26.3 },
#       "swell":       { "height": 0.57 },
#       "wave":        { "sigWaveHeight": 0.57, "sigWaveDirection": 233,
#                        "windWaveHeight": 0.09, "maxHeight": 0.9 },
#       "precipitation": { "inThreeHours": 0, "inSixHours": 0 },
#       "visibility": 10,
#       "relativeHumidity": 72,
#       "currentDirection": 350,
#       "currentSpeed": 0.08,
#       "date": "2026-04-14T06:00:00Z",
#       "timeZoneOffset": 8
#     },
#     "pilotForecast": {
#       "wind":  { "direction": 280, "speed": 5.44, "gust": 6.86 },
#       "swell": { "height": 0.53, "period": 6.4, "direction": 234 },
#       "wave":  { "sigWaveHeight": 0.54, "sigWaveDirection": 236,
#                  "windWaveHeight": 0.11, "windWavePeriod": 1.7,
#                  "windWaveDirection": 286, "maxHeight": 1.0044 },
#       "temperature":   { "air": 26.7, "seaSurface": 26.75 },
#       "visibility": 10.7991,
#       "relativeHumidity": 72,
#       "currentDirection": 24,
#       "currentSpeed": 0.105,
#       "date": "2026-04-14T12:00:00Z",
#       "timeZoneOffset": 8
#     }
#   },
#   ...
# ]

_KTS_TO_MS = 0.514444   # knots → m/s


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _parse_forecast_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        if 'portForecast' in entry:
            return _parse_new_format(entry)
        else:
            return _parse_legacy_format(entry)
    except Exception as e:
        logger.warning("解析 forecast entry 失敗: %s | entry=%s",
                       e, str(entry)[:200])
        return None


def _parse_new_format(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pf = entry.get('portForecast')
    if not isinstance(pf, dict):
        return None

    date_str = pf.get('date') or entry.get('date', '')
    if not date_str:
        return None
    valid_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))

    wind   = pf.get('wind')          or {}
    wave   = pf.get('wave')          or {}
    swell  = pf.get('swell')         or {}
    temp   = pf.get('temperature')   or {}
    precip = pf.get('precipitation') or {}

    # ── portForecast 風速（API 回傳單位：knots）────────────────────
    wind_speed_kts = _safe_float(wind.get('speed'))
    wind_dir_deg   = wind.get('direction')
    wind_gust_kts  = _safe_float(wind.get('gust'))
    wind_speed_ms  = round(wind_speed_kts * _KTS_TO_MS, 4)
    wind_gust_ms   = round(wind_gust_kts  * _KTS_TO_MS, 4)

    # ── portForecast 浪高（保留原始 port 值，不被 pilot 覆蓋）──────
    port_wave_height_m = (
        _safe_float(wave.get('sigWaveHeight'))
        or _safe_float(swell.get('height'))
    )
    port_wave_max_m  = _safe_float(wave.get('maxHeight'))
    port_wave_dir    = wave.get('sigWaveDirection')
    port_swell_h     = _safe_float(swell.get('height'))

    # ── portForecast 能見度 ────────────────────────────────────────
    # AWT API visibility 欄位：
    #   portForecast.visibility  → 單位 NM（如 10 = 10 NM）
    #   pilotForecast.visibility → 單位 NM（如 10.7991 = 10.8 NM）
    port_vis_raw = pf.get('visibility')
    port_vis_nm  = float(port_vis_raw) if port_vis_raw is not None else None

    air_temp_c   = _safe_float(temp.get('air'))
    precip_3h_mm = _safe_float(precip.get('inThreeHours'))
    precip_6h_mm = _safe_float(precip.get('inSixHours'))

    # ── pilotForecast（允許缺筆，所有欄位可為 None）────────────────
    pilot      = entry.get('pilotForecast')
    pilot_data: Dict[str, Any] = {
        # pilot 缺筆時，所有欄位預設 None
        'pilot_wind_speed_kts':     None,
        'pilot_wind_gust_kts':      None,
        'pilot_wind_speed_ms':      None,
        'pilot_wind_dir_deg':       None,
        'pilot_wind_gust_ms':       None,
        'pilot_swell_height_m':     None,
        'pilot_swell_period_s':     None,
        'pilot_swell_dir_deg':      None,
        'pilot_wave_height_m':      None,   # ✅ pilotForecast.wave.sigWaveHeight
        'pilot_wave_max_m':         None,
        'pilot_wave_dir_deg':       None,
        'pilot_wind_wave_height_m': None,
        'pilot_wind_wave_period_s': None,
        'pilot_wind_wave_dir_deg':  None,
        'pilot_sea_surface_temp_c': None,
        'pilot_current_dir_deg':    None,
        'pilot_current_speed_ms':   None,
        'pilot_visibility_nm':      None,   # ✅ 統一用 _nm，不用 _km
    }

    wave_period_s: Optional[float] = None

    if isinstance(pilot, dict):
        p_wind  = pilot.get('wind')        or {}
        p_swell = pilot.get('swell')       or {}
        p_wave  = pilot.get('wave')        or {}
        p_temp  = pilot.get('temperature') or {}

        wave_period_s = _safe_float(p_swell.get('period')) or None

        p_wind_speed_kts = _safe_float(p_wind.get('speed'))
        p_wind_gust_kts  = _safe_float(p_wind.get('gust'))

        # ✅ pilot visibility 單位：NM，直接使用
        p_vis_raw = pilot.get('visibility')
        p_vis_nm  = float(p_vis_raw) if p_vis_raw is not None else None

        pilot_data = {
            'pilot_wind_speed_kts':     p_wind_speed_kts,
            'pilot_wind_gust_kts':      p_wind_gust_kts,
            'pilot_wind_speed_ms':      round(p_wind_speed_kts * _KTS_TO_MS, 4),
            'pilot_wind_dir_deg':       p_wind.get('direction'),
            'pilot_wind_gust_ms':       round(p_wind_gust_kts  * _KTS_TO_MS, 4),
            'pilot_swell_height_m':     _safe_float(p_swell.get('height'))  or None,
            'pilot_swell_period_s':     _safe_float(p_swell.get('period'))  or None,
            'pilot_swell_dir_deg':      p_swell.get('direction'),
            # ✅ pilotForecast.wave.sigWaveHeight（引水點有效波高）
            'pilot_wave_height_m':      _safe_float(p_wave.get('sigWaveHeight')) or None,
            'pilot_wave_max_m':         _safe_float(p_wave.get('maxHeight'))     or None,
            'pilot_wave_dir_deg':       p_wave.get('sigWaveDirection'),
            'pilot_wind_wave_height_m': _safe_float(p_wave.get('windWaveHeight')) or None,
            'pilot_wind_wave_period_s': _safe_float(p_wave.get('windWavePeriod')) or None,
            'pilot_wind_wave_dir_deg':  p_wave.get('windWaveDirection'),
            'pilot_sea_surface_temp_c': _safe_float(p_temp.get('seaSurface'))    or None,
            'pilot_current_dir_deg':    pilot.get('currentDirection'),
            'pilot_current_speed_ms':   _safe_float(pilot.get('currentSpeed'))   or None,
            # ✅ 統一用 pilot_visibility_nm（awt_parser.py 期待此欄位名稱）
            'pilot_visibility_nm':      p_vis_nm,
        }

    return {
        'un_locode':       entry.get('unLocode'),
        'valid_time':      valid_time,
        'timezone_offset': pf.get('timeZoneOffset'),

        # ── 風（kts，API 原始單位）
        'wind_speed_kts':  wind_speed_kts,
        'wind_dir_deg':    wind_dir_deg,
        'wind_gust_kts':   wind_gust_kts,
        'wind_speed_ms':   wind_speed_ms,
        'wind_gust_ms':    wind_gust_ms,

        # ── portForecast 浪（保留獨立欄位，不被 pilot 覆蓋）
        'port_wave_height_m': port_wave_height_m,   # ✅ 新增，供 UI 對比
        'port_wave_max_m':    port_wave_max_m,       # ✅ 新增，供 UI 對比
        'port_swell_height_m': port_swell_h,

        # ── 向下相容欄位（analysis.py / WeatherRecord 使用）
        'wave_height_m':   port_wave_height_m,
        'wave_max_m':      port_wave_max_m,
        'wave_period_s':   wave_period_s,
        'wave_dir_deg':    port_wave_dir,
        'sig_wave_m':      port_wave_height_m,
        'max_wave_m':      port_wave_max_m,
        'swell_height_m':  port_swell_h,

        # ── 環境
        'air_temp_c':          air_temp_c,
        'wind_wave_height_m':  _safe_float(wave.get('windWaveHeight')),
        # ✅ visibility 單位統一為 NM
        'visibility_nm':       port_vis_nm,
        'visibility_km':       port_vis_nm,   # 保留向下相容，值同 NM
        'relative_humidity':   pf.get('relativeHumidity'),
        'current_dir_deg':     pf.get('currentDirection'),
        'current_speed_ms':    _safe_float(pf.get('currentSpeed')),
        'precip_3h_mm':        precip_3h_mm,
        'precip_6h_mm':        precip_6h_mm,

        # ── pilotForecast（缺筆時全為 None，不影響 port 資料顯示）
        **pilot_data,

        '_raw': entry,
    }


def _parse_legacy_format(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    valid_time_str = entry.get('validTime') or entry.get('valid_time', '')
    if not valid_time_str:
        return None
    valid_time = datetime.fromisoformat(
        valid_time_str.replace('Z', '+00:00'))

    wx = (entry.get('weather')
          or entry.get('weatherConditions')
          or {})

    wind_speed_kts = _safe_float(wx.get('windSpeed'))
    wind_gust_kts  = _safe_float(wx.get('windGust'))
    sig_wave_m     = _safe_float(wx.get('significantWaveHeight'))
    max_wave_m     = _safe_float(wx.get('maxWaveHeight'))
    air_temp_c     = _safe_float(wx.get('airTemperature'))
    visibility_km  = _safe_float(wx.get('visibility'), default=99.0)
    precip_6h_mm   = _safe_float(wx.get('precipitation6h'))
    pressure_hpa   = _safe_float(wx.get('pressure'), default=1013.0)

    pilot          = entry.get('pilotConditions') or {}
    pilot_wind_kts = _safe_float(pilot.get('windSpeed'))
    pilot_gust_kts = _safe_float(pilot.get('windGust'))
    pilot_wave_m   = _safe_float(pilot.get('waveHeight'))

    return {
        'un_locode':       entry.get('unLocode'),
        'valid_time':      valid_time,
        'timezone_offset': None,
        'wind_speed_kts':  wind_speed_kts,
        'wind_gust_kts':   wind_gust_kts,
        'wind_speed_ms':   round(wind_speed_kts * _KTS_TO_MS, 4),
        'wind_gust_ms':    round(wind_gust_kts  * _KTS_TO_MS, 4),
        'wind_dir_deg':    None,
        'sig_wave_m':      sig_wave_m,
        'max_wave_m':      max_wave_m,
        'wave_height_m':   sig_wave_m,
        'wave_max_m':      max_wave_m,
        'wave_period_s':   None,
        'wave_dir_deg':    None,
        'air_temp_c':          air_temp_c,
        'visibility_km':       visibility_km,
        'visibility_nm':       visibility_km,
        'precip_6h_mm':        precip_6h_mm,
        'pressure_hpa':        pressure_hpa,
        'swell_height_m':      None,
        'wind_wave_height_m':  None,
        'relative_humidity':   None,
        'current_dir_deg':     None,
        'current_speed_ms':    None,
        'precip_3h_mm':        None,
        'pilot_wind_speed_kts': pilot_wind_kts,
        'pilot_wind_gust_kts':  pilot_gust_kts,
        'pilot_wind_speed_ms':  round(pilot_wind_kts * _KTS_TO_MS, 4),
        'pilot_wind_gust_ms':   round(pilot_gust_kts * _KTS_TO_MS, 4),
        'pilot_wave_height_m':  pilot_wave_m,
        '_raw': entry,
    }


def _extract_forecast_list(raw: Any) -> List[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ('forecasts', 'forecast', 'data', 'items', 'results'):
            val = raw.get(key)
            if isinstance(val, list):
                logger.debug("找到預報陣列 key='%s'，共 %d 筆", key, len(val))
                return val
    logger.warning("Port Forecast: 無法識別回傳結構，型別=%s", type(raw).__name__)
    return []


# ================= Port Forecast 下載器 =================

class AwtWeatherFetcher:

    def __init__(self, login: AwtLoginManager,
                 api_base: str = AWT_API_BASE):
        self.login    = login
        self.api_base = api_base.rstrip('/')
        # ✅ 修正一：直接共用 LoginManager 的 Session
        #    確保登入後的 Cookie 在後續 API 請求中一併帶出
        self._session = login._session

    def fetch_port_weather(
        self,
        port_code:  str,
        days:       int = 2,
        station_id: Optional[str] = None,   # 現在傳入的是正確的 AWT ID（如 6375）
    ) -> List[Dict[str, Any]]:
        days = max(1, min(days, 10))

        def _clean_station_id(raw: Optional[str]) -> Optional[str]:
            if not raw:
                return None
            s = str(raw).strip()
            if s.lower() in ('', '0', 'nan', 'n/a', '#n/a', 'none', 'null'):
                return None
            try:
                return str(int(float(s)))
            except (ValueError, TypeError):
                return s

        clean_sid = _clean_station_id(station_id)

        # ✅ 優先用 AWT Station ID（如 6375），fallback 才用 LOCODE
        primary_key  = clean_sid if clean_sid else port_code
        fallback_key = port_code if clean_sid else None

        logger.info(
            "🔍 fetch_port_weather: port=%r, AWT_ID=%r → primary=%r",
            port_code, station_id, primary_key,
        )

        try:
            headers = self.login.get_auth_headers()
        except RuntimeError as e:
            logger.error("無法取得 Token，跳過港口 %s: %s", port_code, e)
            return []

        raw_data = self._request_forecast(port_code, primary_key, days, headers)

        # fallback：AWT ID 失敗時改用 LOCODE
        if raw_data is None and fallback_key:
            logger.warning(
                "⚠️  AWT ID [%s] 無資料，fallback 用 LOCODE [%s]",
                primary_key, fallback_key,
            )
            raw_data = self._request_forecast(
                port_code, fallback_key, days, headers)

        if raw_data is None:
            return []

        raw_list = _extract_forecast_list(raw_data)
        if not raw_list:
            logger.warning("Port Forecast %s: 回傳資料為空", port_code)
            return []

        records: List[Dict[str, Any]] = []
        for entry in raw_list:
            parsed = _parse_forecast_entry(entry)
            if parsed is not None:
                records.append(parsed)

        logger.info("AWT %s (AWT_ID:%s): 解析 %d/%d 筆",
                    port_code, primary_key, len(records), len(raw_list))
        records.sort(key=lambda r: r['valid_time'])
        return records


    def _request_forecast(
        self,
        port_code: str,
        api_key:   str,
        days:      int,
        headers:   Dict[str, str],
    ) -> Optional[Any]:
        url    = f"{self.api_base}/ports/{api_key}/forecasts"
        params = {"Days": days}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    "📡 [Port Forecast] %s (ID:%s) days=%d (嘗試 %d/%d)",
                    port_code, api_key, days, attempt, MAX_RETRIES)

                resp = self._session.get(
                    url, headers=headers,
                    params=params, timeout=REQUEST_TIMEOUT)
                self.login.log_rate_limit(resp)

                if resp.status_code == 401:
                    logger.warning("Token 過期，重新登入... (嘗試 %d)", attempt)
                    try:
                        self.login._login()
                        headers = self.login.get_auth_headers()
                    except RuntimeError as e:
                        logger.error("重新登入失敗: %s", e)
                        return None
                    continue

                if resp.status_code == 404:
                    if attempt == 1:
                        # 第一次 404：重新登入 + 暖機後重試
                        logger.warning(
                            "AWT %s (ID:%s) 404（嘗試 %d）— "
                            "可能是 Session 未就緒，重新觸發暖機後重試...",
                            port_code, api_key, attempt,
                        )
                        try:
                            self.login._login()
                            headers = self.login.get_auth_headers()
                        except RuntimeError as e:
                            logger.error("重新登入失敗: %s", e)
                            return None
                        continue
                    else:
                        # 第二次以後的 404：確認無此資料
                        logger.warning(
                            "AWT %s (ID:%s) 404（嘗試 %d）— "
                            "確認無此資料，URL: %s",
                            port_code, api_key, attempt, url,
                        )
                        return None

                if resp.status_code == 429:
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning("Rate Limit，等待 %.1fs", wait)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                logger.info("✅ [Port Forecast] %s 資料取得成功", port_code)
                return resp.json()

            except requests.exceptions.Timeout:
                logger.warning(
                    "AWT %s 請求逾時 (嘗試 %d/%d)",
                    port_code, attempt, MAX_RETRIES)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)

            except requests.exceptions.HTTPError as e:
                status = (e.response.status_code
                          if e.response is not None else 'N/A')
                body   = (e.response.text[:300]
                          if e.response is not None else '')
                logger.error(
                    "AWT %s HTTP 錯誤 (%s): %s", port_code, status, body)
                return None

            except requests.exceptions.RequestException as e:
                logger.warning(
                    "AWT %s 請求失敗 (嘗試 %d/%d): %s",
                    port_code, attempt, MAX_RETRIES, e)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)

        logger.error("AWT %s 已達最大重試次數，放棄", port_code)
        return None


# ================= 便利函式 =================

def get_wind_compass(record: Dict[str, Any]) -> str:
    return _deg_to_compass(record.get('wind_dir_deg'))


def get_wave_compass(record: Dict[str, Any]) -> str:
    return _deg_to_compass(record.get('wave_dir_deg'))
