# weather_crawler.py
"""
港口氣象資料爬蟲模組（AWT API 版）

包含：
  - WeatherDatabase：SQLite 資料庫存取（支援 48h / 7d 雙資料表）
  - PortWeatherCrawler：港口資料下載與管理

資料來源：WHL_all_ports_list.xlsx（萬海港口清單）
資料提供：StormGeo (AWT) s-Insight Port Forecast API

安全性設計：
  - 帳號密碼從環境變數讀取，不硬寫在程式碼中
  - JWT Token 不寫入磁碟，僅保存在記憶體中（由 AwtLoginManager 管理）

變更說明：
  - 移除 Selenium 自動登入、Cookie pickle 管理（AWT 使用 Bearer Token，無需瀏覽器）
  - 移除 AedynLoginManager（改用 awt_crawler.AwtLoginManager）
  - 移除 requests HTML 解析（改用 AwtWeatherFetcher + AwtParser）
  - 保留所有原有參數名稱、資料庫結構、公開 API 介面
"""
from __future__ import annotations
import json
from awt_parser import AwtParser, AwtWeatherRecord
import json
import logging
import os
import sqlite3
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from app_config import AppConfig
AWT_API_BASE = AppConfig.awt_api_base()
from dotenv import load_dotenv

# ── dotenv 載入需在所有 os.getenv() 之前執行 ─────────────────────────────────
load_dotenv()

import numpy as np
import pandas as pd
import urllib3

# ── 改寫：引入 AWT 模組取代 Selenium / WNI 爬蟲 ──────────────────────────────
from awt_crawler import AwtLoginManager, AwtWeatherFetcher
from awt_parser import AwtParser, parse_awt_forecast

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


# ================= 私有輔助函式（提前定義，供設定區使用）=================
# 修正：_get_env() 原本在設定區被呼叫時尚未定義，導致 NameError。
# 將輔助函式區塊移至設定區之前即可解決。

def _get_env(key: str, default: str = "") -> str:
    """
    從環境變數讀取字串值。
    未設定時回傳 default，並在 default 為空時記錄警告。
    """
    value = os.environ.get(key, default)
    if not value:
        # 使用 print 而非 logger，因為此時 logging 尚未完整設定
        print(f"[WARNING] 環境變數 {key!r} 未設定")
    return value


def _safe_float(value: object, default: float = 0.0) -> float:
    """安全地將任意值轉為 float，NaN 或轉換失敗時回傳 default"""
    try:
        result = float(value)  # type: ignore[arg-type]
        return default if np.isnan(result) else result
    except (TypeError, ValueError):
        return default


def _json_serializer(obj: Any) -> Any:
    """
    JSON 序列化輔助：處理 datetime 等不可直接序列化的型別。

    變更說明：
      - 原版：content 為純文字，無需 JSON 序列化
      - 改寫：content 為 JSON 字串，需處理 datetime → ISO 格式字串
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ================= 設定區 =================
# 保留原版所有環境變數名稱與預設值，僅移除 Cookie 相關設定。
# 修正：_get_env() 已移至上方，此處呼叫不再有 NameError 風險。

DB_FILE        = Path(os.getenv("IWBDSS_DB_FILE",   "WNI_port_weather.db"))
EXCEL_FILE_WHL = Path(os.getenv("IWBDSS_EXCEL_WHL", "WHL_all_ports_list.xlsx"))
TIMEOUT        = int(os.getenv("IWBDSS_TIMEOUT",    "30"))
MAX_RETRIES    = int(os.getenv("IWBDSS_MAX_RETRIES", "3"))

# ── 改寫：AWT 認證與 API 設定（取代原版 Aedyn Cookie 設定）─────────────────
# 修正：統一使用 AWT_API_BASE（無底線前綴），消除設定區與 __init__ 的命名不一致。
AWT_USERNAME = _get_env("AWT_USERNAME")
AWT_PASSWORD = _get_env("AWT_PASSWORD")
AWT_API_BASE = _get_env(
    "AWT_API_BASE_URL",
    "https://api-shipping.stormgeo.com/api/v1",
)

# ── 改寫：AWT 預報天數對應（取代原版 endpoint "48h" / "7d" 字串）──────────────
# AWT API 以天數（Days 參數）控制預報範圍，48h ≈ 2 天，7d = 7 天。
_ENDPOINT_DAYS: Dict[str, int] = {
    "48h": 2,
    "7d":  7,
}


# ================= 資料結構 =================
# 完整保留原版 PortInfo dataclass，欄位名稱與語意不變。

@dataclass
class PortInfo:
    """單一港口的完整資訊（對應 WHL_all_ports_list.xlsx）"""
    whl_port_code: str    # Port_Code_5
    wni_port_code: str    # WNI Port Code
    port_name_en:  str    # Port Name(English)
    port_name_zh:  str    # Port Name(Chinese)
    country:       str    # Country
    station_id:    str    # Station ID (Object_ID)
    latitude:      float = 0.0
    longitude:     float = 0.0

    def display_str(self) -> str:
        return f"{self.whl_port_code} - {self.port_name_en} ({self.port_name_zh})"


# ================= 資料庫 =================
# 完整保留原版 WeatherDatabase，結構、DDL、公開 API 均不變。
# 唯一差異：content 欄位改為儲存 JSON 序列化的 AWT 預報資料，
# issued_time 改為 API 回傳第一筆預報的 valid_time（YYYYMMDDhhmm 格式）。

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
        self.db_file = db_file
        # 持久連線，check_same_thread=False 允許跨執行緒使用
        self._conn = sqlite3.connect(str(db_file), check_same_thread=False)
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
        """
        自動遷移舊版 schema：
        - 若資料表只有 port_name，自動新增 port_name_en / port_name_zh
        - 並將舊 port_name 資料複製過去
        """
        cur = self._conn.cursor()

        for table in (self.TABLE_48H, self.TABLE_7D):
            cur.execute(f"PRAGMA table_info({table})")
            existing_cols = {row[1] for row in cur.fetchall()}

            if "port_name_en" in existing_cols:
                continue

            logger.info("偵測到舊版 schema，開始遷移資料表：%s", table)

            if "port_name_en" not in existing_cols:
                cur.execute(
                    f"ALTER TABLE {table} ADD COLUMN port_name_en TEXT NOT NULL DEFAULT ''"
                )
            if "port_name_zh" not in existing_cols:
                cur.execute(
                    f"ALTER TABLE {table} ADD COLUMN port_name_zh TEXT NOT NULL DEFAULT ''"
                )

            if "port_name" in existing_cols:
                cur.execute(
                    f"UPDATE {table} SET port_name_en = port_name WHERE port_name_en = ''"
                )
                logger.info("已將 port_name 資料複製至 port_name_en（table=%s）", table)

            logger.info("資料表遷移完成：%s", table)

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
        """回傳 (content, issued_time, port_name_en, port_name_zh) 或 None"""
        cur = self._conn.execute(
            f"SELECT content, issued_time, port_name_en, port_name_zh "
            f"FROM {table} WHERE whl_port_code = ? "
            "ORDER BY issued_time DESC LIMIT 1",
            (whl_port_code,),
        )
        row = cur.fetchone()
        return tuple(row) if row else None  # type: ignore[return-value]

    def _save(
        self,
        table: str,
        port: PortInfo,
        issued_time: str,
        content: str,
    ) -> bool:
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
                "資料庫寫入失敗（table=%s, port=%s）",
                table, port.whl_port_code, exc_info=True,
            )
            return False

    # ── 公開 API — 48h ───────────────────────────────────────

    def get_latest_time(self, whl_port_code: str) -> Optional[str]:
        return self._get_latest_time(self.TABLE_48H, whl_port_code)

    def get_latest_content(
        self, whl_port_code: str
    ) -> Optional[Tuple[str, str, str, str]]:
        """回傳 (content, issued_time, port_name_en, port_name_zh) 或 None"""
        return self._get_latest_content(self.TABLE_48H, whl_port_code)

    def save_weather(self, port: PortInfo, issued_time: str, content: str) -> bool:
        return self._save(self.TABLE_48H, port, issued_time, content)

    # ── 公開 API — 7d ────────────────────────────────────────

    def get_latest_time_7d(self, whl_port_code: str) -> Optional[str]:
        return self._get_latest_time(self.TABLE_7D, whl_port_code)

    def get_latest_content_7d(
        self, whl_port_code: str
    ) -> Optional[Tuple[str, str, str, str]]:
        """回傳 (content, issued_time, port_name_en, port_name_zh) 或 None"""
        return self._get_latest_content(self.TABLE_7D, whl_port_code)

    def save_weather_7d(self, port: PortInfo, issued_time: str, content: str) -> bool:
        return self._save(self.TABLE_7D, port, issued_time, content)


# ================= 爬蟲主體 =================

class PortWeatherCrawler:
    """
    港口氣象資料爬蟲（AWT API 版）。

    職責：
      - 從 WHL_all_ports_list.xlsx 載入港口清單
      - 管理 AWT Bearer Token（由 AwtLoginManager 自動刷新）
      - 透過 AwtWeatherFetcher 下載並儲存 48h / 7d 氣象資料

    變更說明：
      - 移除 Selenium 登入、Cookie pickle、AedynLoginManager
      - 改用 AwtLoginManager（HTTP Basic Auth → Bearer Token）
      - 改用 AwtWeatherFetcher 取代原版 requests.get() 爬蟲
      - 公開介面（fetch_port_data / fetch_all_ports 等）完全保留
    """

    def __init__(
        self,
        username:   str  = "",
        password:   str  = "",
        excel_path: Path = EXCEL_FILE_WHL,
        auto_login: bool = False,   # 保留參數，AWT 版本會在首次請求時自動登入
    ) -> None:
        self.excel_path = Path(excel_path)
        self.db         = WeatherDatabase()
        self._port_map: Dict[str, PortInfo] = {}

        # ── 改寫：以 AwtLoginManager 取代 AedynLoginManager ──────────────────
        # 原版：AedynLoginManager 使用 Selenium 取得 Cookie/JWT
        # 改寫：AwtLoginManager 使用 HTTP Basic Auth 取得 Bearer Token，
        #       Token 自動在記憶體中管理，無需 pickle 或瀏覽器。
        # 修正：優先使用建構子參數，fallback 至模組層級常數（已由 _get_env 載入），
        #       最後才 fallback 至 os.environ，確保三層來源都能正確讀取。
        _username = username or AWT_USERNAME or os.environ.get("AWT_USERNAME", "")
        _password = password or AWT_PASSWORD or os.environ.get("AWT_PASSWORD", "")

        if not _username or not _password:
            logger.warning(
                "AWT_USERNAME / AWT_PASSWORD 未設定，API 請求將無法認證。"
            )

        self.login_manager = AwtLoginManager(
            username = _username,
            password = _password,
            api_base = AWT_API_BASE,   # 修正：使用統一的模組層級常數
        )

        # ── 改寫：以 AwtWeatherFetcher 取代 requests.Session 爬蟲 ────────────
        # 原版：self.session = requests.Session()，手動帶 Cookie/JWT Header
        # 改寫：AwtWeatherFetcher 內部管理 Session 與 Token 刷新
        self.fetcher = AwtWeatherFetcher(
            login    = self.login_manager,
            api_base = AWT_API_BASE,   # 修正：與 login_manager 使用同一常數
        )

        # ── 改寫：AwtParser 取代原版文字解析邏輯 ─────────────────────────────
        # 原版：_parse_issued_time() 從 .txt 純文字解析 ISSUED AT 時間
        # 改寫：AwtParser.parse() 將 API JSON 轉為結構化 AwtWeatherRecord
        self.parser = AwtParser()

        self._load_port_map()

        # auto_login=True 時提前觸發一次登入（取得 Token），否則延遲至首次請求
        if auto_login:
            logger.info("auto_login=True，提前取得 AWT Token")
            try:
                self.login_manager.get_token()
            except RuntimeError:
                logger.error("AWT 提前登入失敗", exc_info=True)

    # ── 屬性 ──────────────────────────────────────────────────

    @property
    def port_list(self) -> List[str]:
        return list(self._port_map.keys())

    # ── 初始化 ────────────────────────────────────────────────

    def _load_port_map(self) -> None:
        if not self.excel_path.exists():
            logger.warning("找不到 Excel 檔案：%s", self.excel_path)
            return

        try:
            # ✅ 明確讀取 all_ports_list 工作表
            df = pd.read_excel(self.excel_path, sheet_name="all_ports_list")
            df.columns = df.columns.str.strip()

            loaded = 0
            for _, row in df.iterrows():
                code = str(row.get("Port_Code_5", "")).strip()

                # ✅ 修正：讀取 AWT 專用的 Station ID
                awt_id = str(row.get("Station ID (AWT)", "")).strip()

                if not code or code == "nan":
                    continue
                if not awt_id or awt_id == "nan":
                    logger.debug("港口 %s 缺少 AWT Station ID，跳過", code)
                    continue

                self._port_map[code] = PortInfo(
                    whl_port_code = code,
                    wni_port_code = str(row.get("WNI_Port Code", code)).strip(),
                    port_name_en  = str(row.get("Port Name(English)", "")).strip(),
                    port_name_zh  = str(row.get("Port Name(Chinese)", "")).strip(),
                    country       = str(row.get("Country", "N/A")).strip(),
                    station_id    = awt_id,   # ✅ 正確的 AWT ID
                    latitude      = _safe_float(row.get("Lat")),
                    longitude     = _safe_float(row.get("Lon")),
                )
                loaded += 1

            logger.info("已載入 %d 個港口資料", loaded)

        except Exception:
            logger.exception("讀取 Excel 失敗：%s", self.excel_path)

    # ── 公開 API ──────────────────────────────────────────────

    def get_all_ports_display(self) -> List[str]:
        """回傳 UI 下拉選單用的字串清單（CODE - 英文名 (中文名)）"""
        return [info.display_str() for info in self._port_map.values()]

    def get_port_info(self, whl_port_code: str) -> Optional[Dict[str, Any]]:
        """
        取得港口完整資訊字典（相容 app_helpers.get_port_full_info）。

        Returns:
            包含 port_name_en/zh、country、latitude、longitude 等的字典，
            或 None（代碼不存在時）
        """
        port = self._port_map.get(whl_port_code)
        if port is None:
            logger.warning("港口代碼 %s 不在 port_map 中", whl_port_code)
            return None

        return {
            "port_name":     port.port_name_en,   # app_helpers 讀取 port_name
            "port_name_en":  port.port_name_en,
            "port_name_zh":  port.port_name_zh,
            "port_code":     whl_port_code,
            "whl_port_code": whl_port_code,
            "wni_port_code": port.wni_port_code,
            "country":       port.country,
            "station_id":    port.station_id,
            "latitude":      port.latitude,
            "longitude":     port.longitude,
        }

    def get_data_from_db(
        self, whl_port_code: str
    ) -> Optional[Tuple[str, str, str, str]]:
        """從資料庫讀取 48h 最新氣象內容（content, issued_time, name_en, name_zh）"""
        return self.db.get_latest_content(whl_port_code)

    def get_data_from_db_7d(
        self, whl_port_code: str
    ) -> Optional[Tuple[str, str, str, str]]:
        """從資料庫讀取 7d 最新氣象內容（content, issued_time, name_en, name_zh）"""
        return self.db.get_latest_content_7d(whl_port_code)

    # ── 下載核心（私有）─────────────────────────────────────

    def _fetch_port_data(
        self,
        whl_port_code: str,
        endpoint: str,          # "48h" 或 "7d"
        retry_login: bool = True,
    ) -> Tuple[bool, str]:
        """
        通用下載邏輯，由公開方法包裝呼叫。

        變更說明：
          - 原版：requests.get() 下載 .txt 檔，解析 ISSUED AT 時間
          - 改寫：AwtWeatherFetcher.fetch_port_weather() 呼叫 REST API，
                  AwtParser.parse() 解析 JSON 回傳結構化資料，
                  issued_time 取第一筆預報的 valid_time（YYYYMMDDhhmm 格式）
          - Token 過期由 AwtLoginManager.get_token() 自動刷新，
                  無需手動 retry_login 邏輯；保留參數以維持介面相容性。
        """
        port = self._port_map.get(whl_port_code)
        if port is None:
            return False, f"找不到港口代碼：{whl_port_code}"

        # ── 改寫：以 _ENDPOINT_DAYS 對應 AWT Days 參數 ───────────────────────
        # 原版：endpoint 直接作為 URL 路徑片段（"48h" / "7d"）
        # 改寫：endpoint 對應 Days 數值，傳入 AwtWeatherFetcher
        days  = _ENDPOINT_DAYS.get(endpoint, 2)
        label = "48小時" if endpoint == "48h" else "7天"

        logger.info(
            "下載 %s（%s / %s）- %s 預報",
            whl_port_code, port.port_name_en, port.port_name_zh, label,
        )

        try:
            # ── 改寫：呼叫 AWT API 取得結構化預報資料 ────────────────────────
            # 原版：self.session.get(url, headers=self.headers, ...)
            # 改寫：self.fetcher.fetch_port_weather() 內部處理 Token、重試、404
            raw_records = self.fetcher.fetch_port_weather(
                port_code  = whl_port_code,
                days       = days,
                station_id = port.station_id,
            )
        except Exception as exc:
            return False, f"API 請求失敗：{exc}"

        if not raw_records:
            return False, f"API 回傳空資料（{endpoint}）"

        return self._handle_success(port, raw_records, endpoint)

    def _handle_success(
        self,
        port:        PortInfo,
        raw_records: List[Dict[str, Any]],
        endpoint:    str,
    ) -> Tuple[bool, str]:
        """
        處理成功下載的回應：解析、比對版本並儲存。

        變更說明：
          - 原版：直接比對 .txt 文字的 ISSUED AT 時間，content 為純文字
          - 改寫：
            1. AwtParser.parse() 解析 raw_records → AwtWeatherRecord 列表
            2. issued_time 取第一筆 valid_time（YYYYMMDDhhmm 格式，與原版一致）
            3. content 序列化為 JSON 字串儲存（保留完整結構化資料）
        """
        # ── 解析 AWT 資料 ─────────────────────────────────────────────────────
        port_name, wind_records, cond_records = self.parser.parse(
            raw_records     = raw_records,
            port_name       = port.port_name_en,
            tz_offset_hours = 8,    # 預設 UTC+8，可依港口時區調整
        )

        if not wind_records:
            return False, "資料解析後為空"

        # ── issued_time：取第一筆預報的 valid_time ────────────────────────────
        # 原版：從 .txt 文字解析 "ISSUED AT: 202506010600 UTC"，格式 YYYYMMDDhhmm
        # 改寫：直接從 AwtWeatherRecord.time（datetime with tzinfo）取得，
        #       保持相同的 YYYYMMDDhhmm 格式，確保資料庫 UNIQUE 去重邏輯不受影響。
        issued_time = wind_records[0].time.strftime("%Y%m%d%H%M")

        # ── content：序列化為 JSON 字串 ───────────────────────────────────────
        # 原版：content 為原始 .txt 純文字
        # 改寫：content 為 JSON 序列化的預報資料（保留結構，方便後續解析）
        content = json.dumps(
            [r.to_dict() for r in wind_records],
            ensure_ascii=False,
            default=_json_serializer,
        )

        # ── 比對版本並儲存（邏輯與原版相同）─────────────────────────────────
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
        """通用批次下載（邏輯與原版完全相同）"""
        label  = "48小時" if endpoint == "48h" else "7天"
        total  = len(self.port_list)
        counts = {"success": 0, "skipped": 0, "failed": 0}

        logger.info("開始批次下載 %d 個港口（%s）", total, label)

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
            "%s 批次下載完成 — 成功：%d，略過：%d，失敗：%d",
            label, counts["success"], counts["skipped"], counts["failed"],
        )
        return counts

    # ── 公開下載 API ─────────────────────────────────────────
    # 以下所有公開方法簽名與原版完全相同，不做任何更動。

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
        """批次下載所有港口的 48h + 7d 預報（邏輯與原版完全相同）"""
        total  = len(self.port_list)
        s_48h: Dict[str, int] = {"success": 0, "skipped": 0, "failed": 0}
        s_7d:  Dict[str, int] = {"success": 0, "skipped": 0, "failed": 0}

        logger.info("開始批次下載 %d 個港口（48h + 7d）", total)

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

        logger.info("批次下載完成 — 48h：%s / 7d：%s", s_48h, s_7d)
        return {"48h": s_48h, "7d": s_7d}

    def test_api_connection(self) -> None:
        """
        測試 AWT API 連線與認證狀態（除錯用）。

        變更說明：
          - 原版：測試 _AEDYN_USER_API 與 _AEDYN_BASE_URL
          - 改寫：測試 AWT /auth/login（透過 get_token()）
                  與 /ports/SGSIN/forecasts（用新加坡港口驗證資料存取）
        """
        logger.info("測試 AWT API 連線...")

        # 測試 Token 取得
        try:
            token = self.login_manager.get_token()
            logger.info("✅ AWT Token 取得成功（長度：%d）", len(token))
        except RuntimeError:
            logger.error("❌ AWT Token 取得失敗", exc_info=True)
            return

        # 測試實際資料請求（以新加坡港口為例）
        try:
            records = self.fetcher.fetch_port_weather("SGSIN", days=1)
            if records:
                logger.info(
                    "✅ AWT 資料請求成功（SGSIN，共 %d 筆預報）", len(records)
                )
            else:
                logger.warning("⚠️ AWT 資料請求回傳空資料（SGSIN）")
        except Exception:
            logger.error("❌ AWT 資料請求失敗", exc_info=True)


# ================= 使用範例 =================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("初始化爬蟲系統（WHL 港口清單 × AWT API）")
    logger.info("=" * 60)

    # ── 改寫：直接使用模組層級常數（已由 _get_env / load_dotenv 載入）──────
    # 原版：os.getenv("AEDYN_USERNAME") / os.getenv("AEDYN_PASSWORD")
    # 改寫：AWT_USERNAME / AWT_PASSWORD（模組層級，避免重複讀取環境變數）
    crawler = PortWeatherCrawler(
        username   = AWT_USERNAME,
        password   = AWT_PASSWORD,
        auto_login = False,
    )
    crawler.test_api_connection()

    # ── 範例 1: 下載單一港口 48h 預報
    ok, msg = crawler.fetch_port_data("TWKHH")
    logger.info("TWKHH 48h 結果：%s", msg)

    # ── 範例 2: 下載單一港口 7d 預報
    ok, msg = crawler.fetch_port_data_7d("TWKHH")
    logger.info("TWKHH 7d 結果：%s", msg)

    # ── 範例 3: 從 DB 讀取 48h 預報
    data = crawler.get_data_from_db("TWKHH")
    if data:
        content, issued_time, name_en, name_zh = data
        logger.info("港口：%s（%s），發布時間：%s", name_en, name_zh, issued_time)
        # ── 改寫：content 為 JSON 字串，需反序列化後使用 ──────────────────
        records = json.loads(content)
        logger.info(
            "預報筆數：%d，第一筆預覽：%s",
            len(records), str(records[0])[:200],
        )

    # ── 範例 4: 取得港口資訊（與原版完全相同）
    port_info = crawler.get_port_info("TWKHH")
    if port_info:
        logger.info("港口資訊：%s", json.dumps(port_info, ensure_ascii=False))
