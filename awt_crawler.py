# awt_crawler.py
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

_WARMUP_ENDPOINTS = [
    '/documentation',
    '/documentation/open-api/v1',
]

_DEG_TO_DIR: List[Tuple[float, str]] = [
    (11.25,  'N'),   (33.75,  'NNE'), (56.25,  'NE'),  (78.75,  'ENE'),
    (101.25, 'E'),   (123.75, 'ESE'), (146.25, 'SE'),  (168.75, 'SSE'),
    (191.25, 'S'),   (213.75, 'SSW'), (236.25, 'SW'),  (258.75, 'WSW'),
    (281.25, 'W'),   (303.75, 'WNW'), (326.25, 'NW'),  (348.75, 'NNW'),
]

# ================= AWT Station ID 完整對照表 =================
# 來源：WHL_all_ports_list.xlsx — 欄位「Station ID (AWT)」
# key   = Port_Code_5（上游傳入）
# value = Station ID (AWT)（實際送入 API 的 unLocode）
# 所有港口一律使用 Station ID (AWT)，不再 fallback 至 Port_Code_5
AWT_STATION_ID_MAP: Dict[str, str] = {
    # United Arab Emirates
    "AEJEA": "AEJEA",   # JEBEL ALI
    # China
    "CNDLC": "CNDAL",   # DALIAN
    "CNFOC": "CNFZH",   # FUZHOU
    "CNHSK": "CNTNJ",   # TIANJIN
    "CNJIA": "CNJIX",   # JIAXING
    "CNLYG": "CNLYG",   # LIANYUNGANG
    "CNNGB": "CNNBO",   # NINGBO
    "CNXIA": "CNXIA",   # 蝦峙門
    "CNNSS": "CNNSA",   # NANSHA
    "CNQZH": "CNQZL",   # QUANZHOU
    "CNRZH": "CNRZH",   # RIZHAO
    "CNSHA": "CNSGH",   # SHANGHAI
    "CNZOS": "CNZOS",   # ZHOUSHAN
    "CNSKU": "CNSHK",   # SHEKOU
    "CNTAO": "CNQIN",   # QINGDAO
    "CNXMN": "CNXAM",   # XIAMEN
    "CNYTN": "CNYTN",   # YANTIAN
    "CNZHA": "CNZHA",   # ZHANJIANG
    # Colombia
    "COBUN": "COBUN",   # BUENAVENTURA
    # Ecuador
    "ECGYE": "ECGYE",   # GUAYAQUIL
    # Egypt
    "EGSOK": "EGSOK",   # SOKHNA
    # Guatemala
    "GTPRQ": "GTPRQ",   # PUERTO QUETZAL
    # Hong Kong
    "HKHKG": "HKHKG",   # HONG KONG
    # Indonesia
    "IDBLW": "IDBLW",   # BELAWAN
    "IDJKT": "IDOJ",    # JAKARTA
    "IDSRG": "IDSRG",   # SEMARANG
    "IDSUB": "IDSUB",   # SURABAYA
    # India
    "INCOK": "INCOK",   # COCHIN
    "INKAT": "INKAT",   # KATTUPALLI
    "INMAA": "INMAA",   # CHENNAI
    "INMUN": "INMUN",   # MUNDRA
    "INNSA": "INNSA1",  # NHAVA SHEVA
    "INTUT": "INTUT",   # TUTICORIN
    "INVIZ": "INVTZ",   # VISAKHAPATNAM
    # Japan
    "JPCHB": "JPCHB",   # CHIBA
    "JPFKY": "JPFKY",   # FUKUYAMA
    "JPHIJ": "JPHIJ",   # HIROSHIMA
    "JPHKT": "JPHKT",   # HAKATA
    "JPKWS": "JPKWS",   # KAWASAKI
    "JPMIZ": "JPMIZ",   # MIZUSHIMA
    "JPMOJ": "JPMOJ",   # MOJI
    "JPNGO": "JPNGO",   # NAGOYA
    "JPOSA": "JPOSA",   # OSAKA
    "JPSMZ": "JPSMZ",   # SHIMIZU
    "JPTYO": "JPTYO",   # TOKYO
    "JPUKB": "JPUKB",   # KOBE
    "JPYKK": "JPYKK",   # YOKKAICHI
    "JPYOK": "JPYOK",   # YOKOHAMA
    # Cambodia
    "KHSIH": "KHKOS",   # SIHANOUKVILLE
    # Korea
    "KRINC": "KRINC",   # INCHEON
    "KRPUS": "KRPUS",   # BUSAN
    "KRUSN": "KRUSN",   # ULSAN
    "KRBNP": "KRBNP",   # PUSAN NEWPORT
    # Sri Lanka
    "LKCMB": "LKCMB",   # COLOMBO
    # Mexico
    "MXESE": "MXESE",   # ENSENADA
    "MXLZC": "MXLZC",   # LAZARO CARDENAS
    "MXZLO": "MXZLO",   # MANZANILLO
    # Malaysia
    "MYBUT": "MYBWH",   # BUTTERWORTH
    "MYPGU": "MYPGU",   # PASIR GUDANG
    "MYPKG": "MYPKG",   # PORT KELANG
    # Panama
    "PACTB": "PACTB",   # CRISTOBAL
    # Peru
    "PECLL": "PECLL",   # CALLAO
    # Philippines
    "PHCEB": "PHCEB",   # CEBU CITY
    "PHCGY": "PHCGY",   # CAGAYAN DE ORO
    "PHDVO": "PHDVO",   # DAVAO
    "PHMNN": "PHMNL",   # MANILA 北港
    "PHMNS": "PHMNL",   # MANILA 南港
    "PHSFS": "PHSFS",   # SUBIC BAY
    # Pakistan
    "PKBQM": "PKBQM1",  # BIN QASIM
    # Saudi Arabia
    "SAJED": "SAJED",   # JEDDAH
    # Singapore
    "SGSIN": "SGSCT",   # SINGAPORE
    # Thailand
    "THBKK": "THBKK",   # BANGKOK
    "THLCH": "THLCH",   # LAEM CHABANG
    # Taiwan
    "TWKEL": "TWKEL",   # KEELUNG
    "TWKHH": "TWKHH",   # KAOHSIUNG
    "TWTPE": "TWTPE",   # TAIPEI
    "TWTXG": "TWTXG",   # TAICHUNG
    # United States
    "USCHS": "USCHS",   # CHARLESTON
    "USLAX": "USLAX",   # LOS ANGELES
    "USNYC": "USNYC",   # NEW YORK
    "USOAK": "USOAK",   # OAKLAND
    "USORF": "USORF",   # NORFOLK
    "USSAV": "USSAV",   # SAVANNAH
    # Vietnam
    "VNCLP": "VNCLN",   # CAI LAN
    "VNHCH": "VNSGN",   # HO CHI MINH
    "VNDAD": "VNDAD",   # DA NANG
    "VNHPH": "VNHPH",   # HAIPHONG
    "VNTCT": "VNCMT",   # Cai Mep
    # Chile
    "CLVAP": "CLVAP",   # VALPARAISO
}

# 無 Port_Code_5 的特殊港口（以自訂 key 存取，Station ID 為數字）
AWT_NUMERIC_STATION_MAP: Dict[str, str] = {
    "VUNG_TAU":        "6373",  # VUNG TAU P/S
    "ZHOUSHAN_ISLAND": "8149",  # ZHOUSHAN ISLAND
    "CNCJK":           "CNCJK", # CHANG JIANG KOU（Port_Code_5 空，WNI=NJK）
}


def _resolve_awt_locode(port_code: str,
                        station_id: Optional[str] = None) -> Optional[str]:
    """
    決定實際送入 AWT API 的 unLocode，優先順序：

    1. 呼叫端明確傳入 station_id → 直接使用（最高優先）
    2. port_code 存在於 AWT_STATION_ID_MAP → 使用對照表的 Station ID (AWT)
    3. port_code 存在於 AWT_NUMERIC_STATION_MAP → 使用數字 Station ID
    4. 以上皆無 → 記錄 WARNING 並回傳 None（不再 fallback 至 port_code）

    回傳 None 表示無法解析，呼叫端應跳過此港口。
    """
    # 1. 明確指定的 station_id 最優先
    if station_id:
        resolved = _normalize_locode(station_id)
        if resolved:
            logger.debug(
                "🗺️  [%s] 使用明確指定的 station_id → %r", port_code, resolved)
            return resolved
        logger.warning(
            "⚠️  station_id=%r 正規化後為空，繼續查詢對照表", station_id)

    normalized_pc = _normalize_locode(port_code)

    # 2. 查詢主對照表
    if normalized_pc and normalized_pc in AWT_STATION_ID_MAP:
        mapped = AWT_STATION_ID_MAP[normalized_pc]
        logger.debug(
            "🗺️  [%s] AWT_STATION_ID_MAP: %r → %r",
            port_code, normalized_pc, mapped,
        )
        return mapped

    # 3. 查詢數字/特殊對照表
    if normalized_pc and normalized_pc in AWT_NUMERIC_STATION_MAP:
        mapped = AWT_NUMERIC_STATION_MAP[normalized_pc]
        logger.debug(
            "🗺️  [%s] AWT_NUMERIC_STATION_MAP: %r → %r",
            port_code, normalized_pc, mapped,
        )
        return mapped

    # 4. 找不到 → 不猜測，直接回 None
    logger.warning(
        "⚠️  [%s] 在 AWT_STATION_ID_MAP 中找不到對應的 Station ID，跳過此港口。"
        "請將此港口加入對照表。",
        port_code,
    )
    return None


# ================= UN/LOCODE 工具 =================

def _normalize_locode(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    s = str(code).strip().upper()
    if s in ('', '0', 'NAN', 'N/A', '#N/A', 'NONE', 'NULL'):
        return None
    return s


# ================= 安全型別轉換 =================

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_float_or_none(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


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
            "Content-Type":  "application/json",
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
            self._activate_web_session()

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 'N/A'
            body   = e.response.text[:300]  if e.response is not None else ''
            raise RuntimeError(f"AWT 登入失敗 (HTTP {status}): {body}") from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"AWT 登入網路錯誤: {e}") from e

    def _activate_web_session(self) -> None:
        if not self._token:
            return
        headers = {
            'Authorization': f'Bearer {self._token}',
            'Accept':        'application/json',
            'Content-Type':  'application/json',
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
                        "✅ API 連線驗證成功（端點: %s，狀態: %d）",
                        endpoint, resp.status_code,
                    )
                    warmup_success = True
                    break
                else:
                    logger.debug(
                        "   驗證端點 %s 回應 %d，嘗試下一個",
                        endpoint, resp.status_code,
                    )
            except requests.exceptions.RequestException as e:
                logger.debug("   驗證端點 %s 失敗: %s", endpoint, e)

        if not warmup_success:
            logger.warning("⚠️  API 文件端點無回應，Token 仍有效，繼續執行")
        else:
            logger.debug("   等待連線穩定（1.0s）...")
            time.sleep(1.0)

    def get_auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.get_token()}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }

    def log_rate_limit(self, response: requests.Response) -> None:
        limit     = response.headers.get('X-RateLimit-Limit')
        remaining = response.headers.get('X-RateLimit-Remaining')
        reset     = response.headers.get('X-RateLimit-Reset')
        if limit or remaining:
            logger.debug("Rate Limit → 上限: %s，剩餘: %s，重置: %s",
                         limit, remaining, reset)


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


# ================= Port Forecast 解析 =================

_KTS_TO_MS = 0.514444


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

    wind   = pf.get('wind')   or {}
    wave   = pf.get('wave')   or {}
    swell  = pf.get('swell')  or {}
    temp   = pf.get('temperature') or {}
    precip = pf.get('precipitation') or {}

    wind_speed_kts = _safe_float(wind.get('speed'))
    wind_dir_deg   = wind.get('direction')
    wind_gust_kts  = _safe_float(wind.get('gust'))
    wind_speed_ms  = round(wind_speed_kts * _KTS_TO_MS, 4)
    wind_gust_ms   = round(wind_gust_kts  * _KTS_TO_MS, 4)

    wave_height_m = _safe_float_or_none(wave.get('sigWaveHeight'))
    if wave_height_m is None:
        wave_height_m = _safe_float_or_none(swell.get('height'))

    wave_max_m    = _safe_float_or_none(wave.get('maxHeight'))
    wave_dir_deg  = wave.get('sigWaveDirection')
    wave_period_s: Optional[float] = None

    air_temp_c    = _safe_float_or_none(temp.get('air'))
    visibility_km = _safe_float(pf.get('visibility'), default=99.0)
    precip_3h_mm  = _safe_float_or_none(precip.get('inThreeHours'))
    precip_6h_mm  = _safe_float_or_none(precip.get('inSixHours'))

    pilot      = entry.get('pilotForecast')
    pilot_data: Dict[str, Any] = {}

    if isinstance(pilot, dict):
        p_wind  = pilot.get('wind')  or {}
        p_swell = pilot.get('swell') or {}
        p_wave  = pilot.get('wave')  or {}
        p_temp  = pilot.get('temperature') or {}

        wave_period_s = _safe_float_or_none(p_swell.get('period'))

        if wave_height_m is None:
            wave_height_m = _safe_float_or_none(p_wave.get('sigWaveHeight'))
        if wave_height_m is None:
            wave_height_m = _safe_float_or_none(p_swell.get('height'))
        if wave_max_m is None:
            wave_max_m = _safe_float_or_none(p_wave.get('maxHeight'))
        if wave_dir_deg is None:
            wave_dir_deg = (p_wave.get('sigWaveDirection')
                            or p_swell.get('direction'))

        p_wind_speed_kts = _safe_float(p_wind.get('speed'))
        p_wind_gust_kts  = _safe_float(p_wind.get('gust'))

        pilot_data = {
            'pilot_wind_speed_kts':     p_wind_speed_kts,
            'pilot_wind_gust_kts':      p_wind_gust_kts,
            'pilot_wind_speed_ms':      round(p_wind_speed_kts * _KTS_TO_MS, 4),
            'pilot_wind_dir_deg':       p_wind.get('direction'),
            'pilot_wind_gust_ms':       round(p_wind_gust_kts  * _KTS_TO_MS, 4),
            'pilot_swell_height_m':     _safe_float_or_none(p_swell.get('height')),
            'pilot_swell_period_s':     _safe_float_or_none(p_swell.get('period')),
            'pilot_swell_dir_deg':      p_swell.get('direction'),
            'pilot_wave_height_m':      _safe_float_or_none(p_wave.get('sigWaveHeight')),
            'pilot_wave_max_m':         _safe_float_or_none(p_wave.get('maxHeight')),
            'pilot_wave_dir_deg':       p_wave.get('sigWaveDirection'),
            'pilot_wind_wave_height_m': _safe_float_or_none(p_wave.get('windWaveHeight')),
            'pilot_wind_wave_period_s': _safe_float_or_none(p_wave.get('windWavePeriod')),
            'pilot_wind_wave_dir_deg':  p_wave.get('windWaveDirection'),
            'pilot_sea_surface_temp_c': _safe_float_or_none(p_temp.get('seaSurface')),
            'pilot_current_dir_deg':    pilot.get('currentDirection'),
            'pilot_current_speed_ms':   _safe_float_or_none(pilot.get('currentSpeed')),
            'pilot_visibility_km':      _safe_float_or_none(pilot.get('visibility')),
        }

    return {
        'un_locode':       entry.get('unLocode'),
        'valid_time':      valid_time,
        'timezone_offset': pf.get('timeZoneOffset'),
        'wind_speed_kts':  wind_speed_kts,
        'wind_dir_deg':    wind_dir_deg,
        'wind_gust_kts':   wind_gust_kts,
        'wind_speed_ms':   wind_speed_ms,
        'wind_gust_ms':    wind_gust_ms,
        'wave_height_m':   wave_height_m,
        'wave_max_m':      wave_max_m,
        'wave_period_s':   wave_period_s,
        'wave_dir_deg':    wave_dir_deg,
        'sig_wave_m':      wave_height_m,
        'max_wave_m':      wave_max_m,
        'air_temp_c':          air_temp_c,
        'swell_height_m':      _safe_float_or_none(swell.get('height')),
        'wind_wave_height_m':  _safe_float_or_none(wave.get('windWaveHeight')),
        'visibility_km':       visibility_km,
        'visibility_nm':       visibility_km,
        'relative_humidity':   pf.get('relativeHumidity'),
        'current_dir_deg':     pf.get('currentDirection'),
        'current_speed_ms':    _safe_float_or_none(pf.get('currentSpeed')),
        'precip_3h_mm':        precip_3h_mm,
        'precip_6h_mm':        precip_6h_mm,
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
    sig_wave_m     = _safe_float_or_none(wx.get('significantWaveHeight'))
    max_wave_m     = _safe_float_or_none(wx.get('maxWaveHeight'))
    air_temp_c     = _safe_float_or_none(wx.get('airTemperature'))
    visibility_km  = _safe_float(wx.get('visibility'), default=99.0)
    precip_6h_mm   = _safe_float_or_none(wx.get('precipitation6h'))
    pressure_hpa   = _safe_float(wx.get('pressure'), default=1013.0)

    pilot          = entry.get('pilotConditions') or {}
    pilot_wind_kts = _safe_float(pilot.get('windSpeed'))
    pilot_gust_kts = _safe_float(pilot.get('windGust'))
    pilot_wave_m   = _safe_float_or_none(pilot.get('waveHeight'))

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
    """
    文件端點：GET /ports/{unLocode}/forecasts?Days={days}

    所有港口一律透過 AWT_STATION_ID_MAP 將 Port_Code_5
    轉換為正確的 Station ID (AWT) 後送入 API。
    """

    def __init__(self, login: AwtLoginManager,
                 api_base: str = AWT_API_BASE):
        self.login    = login
        self.api_base = api_base.rstrip('/')
        self._session = login._session

    def fetch_port_weather(
        self,
        port_code:  str,
        days:       int = 2,
        station_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        取得港口天氣預報。

        Args:
            port_code:  Port_Code_5，例如 'TWKHH'、'CNSHA'
            days:       預報天數，1–10，預設 2
            station_id: 明確指定 AWT Station ID（最高優先，可覆蓋對照表）

        Returns:
            解析後的預報資料列表，依時間排序。
        """
        days = max(1, min(days, 10))

        api_key = _resolve_awt_locode(port_code, station_id)
        if api_key is None:
            # _resolve_awt_locode 已輸出 WARNING，此處直接跳過
            return []

        logger.info(
            "🔗 [%s] AWT Station ID=%r → %s/ports/%s/forecasts?Days=%d",
            port_code, api_key, self.api_base, api_key, days,
        )

        try:
            headers = self.login.get_auth_headers()
        except RuntimeError as e:
            logger.error("無法取得 Token，跳過港口 %s: %s", port_code, e)
            return []

        raw_data = self._request_forecast(port_code, api_key, days, headers)

        if raw_data is None:
            logger.warning("❌ [%s] Station ID=%r 無資料回傳", port_code, api_key)
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

        logger.info("AWT %s (Station ID=%s): 解析 %d/%d 筆",
                    port_code, api_key, len(records), len(raw_list))
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
                    "📡 [Port Forecast] %s (Station ID=%s) days=%d (嘗試 %d/%d)",
                    port_code, api_key, days, attempt, MAX_RETRIES,
                )

                resp = self._session.get(
                    url, headers=headers,
                    params=params, timeout=REQUEST_TIMEOUT,
                )
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
                    if attempt < MAX_RETRIES:
                        wait = 3.0 * attempt
                        logger.warning(
                            "AWT %s 404（嘗試 %d），等待 %.1fs 後重試...",
                            port_code, attempt, wait,
                        )
                        if attempt == 1:
                            try:
                                self.login._login()
                                headers = self.login.get_auth_headers()
                            except RuntimeError as e:
                                logger.error("重新登入失敗: %s", e)
                                return None
                        time.sleep(wait)
                        continue
                    else:
                        logger.warning(
                            "AWT %s 404，已達最大重試次數 — "
                            "請確認 Station ID [%s] 是否在帳號授權清單內",
                            port_code, api_key,
                        )
                        return None

                if resp.status_code == 429:
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "Rate Limit（300 hits/hour），等待 %.1fs", wait)
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                logger.info("✅ [Port Forecast] %s 資料取得成功", port_code)
                return resp.json()

            except requests.exceptions.Timeout:
                logger.warning(
                    "AWT %s 請求逾時 (嘗試 %d/%d)",
                    port_code, attempt, MAX_RETRIES,
                )
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
                    port_code, attempt, MAX_RETRIES, e,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)

        logger.error("AWT %s 已達最大重試次數，放棄", api_key)
        return None


# ================= 便利函式 =================

def get_wind_compass(record: Dict[str, Any]) -> str:
    return _deg_to_compass(record.get('wind_dir_deg'))


def get_wave_compass(record: Dict[str, Any]) -> str:
    return _deg_to_compass(record.get('wave_dir_deg'))