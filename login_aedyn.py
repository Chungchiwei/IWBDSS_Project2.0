import requests
import sqlite3
import pandas as pd
import os
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import urllib3
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
from dotenv import load_dotenv

# ── 初始化 ──────────────────────────────────────────────
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── 日誌設定 ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ================= 設定區 =================
@dataclass
class Config:
    DB_FILE: str = "port_weather.db"
    EXCEL_FILE: str = "all_ports_list.xlsx"
    TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    LOGIN_URL: str = (
        "https://idp.aedyn.wni.com/auth/realms/aedyn/protocol/openid-connect/auth"
        "?response_type=id_token%20token&scope=openid&client_id=aedyn"
        "&state=cZr_CP7VqEq2p8j6D_a_YrL2ucA"
        "&redirect_uri=https%3A%2F%2Faedyn.weathernews.com%2Fhttpd-auth%2Fredirect_uri"
        "&nonce=cwGprMflnWRdzaLvLMkCMI2az5vjS79XdTW0gtUulwo"
    )
    # ✅ 從環境變數讀取，不硬寫在程式碼中
    USERNAME: str = field(default_factory=lambda: os.getenv("AEDYN_USERNAME", ""))
    PASSWORD: str = field(default_factory=lambda: os.getenv("AEDYN_PASSWORD", ""))

    def validate(self):
        if not self.USERNAME or not self.PASSWORD:
            raise EnvironmentError(
                "❌ 找不到帳號或密碼，請在 .env 檔案中設定 AEDYN_USERNAME 與 AEDYN_PASSWORD"
            )


CONFIG = Config()


# ================= 資料結構 =================
@dataclass
class PortInfo:
    """港口資訊的型別安全容器"""
    code: str
    name: str
    station_id: str
    country: str
    latitude: float = 0.0
    longitude: float = 0.0


@dataclass
class LoginResult:
    """登入結果容器"""
    cookies: dict
    jwt_token: str = ""

    @property
    def is_valid(self) -> bool:
        return bool(self.cookies)


# ================= 登入管理 =================
class AedynLoginManager:
    """負責自動登入 Aedyn 並取得最新 Cookie 和 JWT Token"""

    BASE_URL = "https://aedyn.weathernews.com"

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self._login_result: Optional[LoginResult] = None

    # ── 公開介面 ─────────────────────────────────────────

    def login_and_get_cookies(self, headless: bool = True) -> LoginResult:
        """
        使用 Selenium 登入 Aedyn，回傳 LoginResult。
        若登入失敗會拋出例外。
        """
        driver = self._build_driver(headless)
        try:
            self._perform_login(driver)
            cookies = self._collect_cookies(driver)
            jwt_token = self._extract_jwt(driver, cookies)
            self._verify_session(cookies, jwt_token)

            self._login_result = LoginResult(cookies=cookies, jwt_token=jwt_token)
            return self._login_result

        except Exception as exc:
            self._save_error_screenshot(driver)
            raise RuntimeError(f"登入失敗: {exc}") from exc

        finally:
            driver.quit()

    def get_headers(self) -> dict:
        """回傳完整 HTTP Headers（含 Cookie 與 JWT）"""
        if not self._login_result or not self._login_result.is_valid:
            raise RuntimeError("尚未登入，請先呼叫 login_and_get_cookies()")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/143.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-TW,zh-CN;q=0.9,zh;q=0.8,en-US;q=0.7,en;q=0.6",
            "Referer": f"{self.BASE_URL}/",
            "sec-ch-ua": '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "Cookie": self._cookie_string(),
        }
        if self._login_result.jwt_token:
            headers["json_web_token"] = self._login_result.jwt_token
        return headers

    # ── 私有輔助方法 ──────────────────────────────────────

    @staticmethod
    def _build_driver(headless: bool) -> webdriver.Chrome:
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        if headless:
            options.add_argument("--headless=new")
        return webdriver.Chrome(options=options)

    def _perform_login(self, driver: webdriver.Chrome):
        """填寫帳密並等待跳轉完成"""
        wait = WebDriverWait(driver, 30)
        logger.info("🔐 正在登入 Aedyn...")
        driver.get(CONFIG.LOGIN_URL)

        try:
            user_el = wait.until(EC.visibility_of_element_located((By.ID, "username")))
            pwd_el = wait.until(EC.visibility_of_element_located((By.ID, "password")))
            user_el.clear()
            user_el.send_keys(self.username)
            pwd_el.clear()
            pwd_el.send_keys(self.password)
            pwd_el.send_keys(Keys.ENTER)

            wait.until(
                lambda d: self.BASE_URL in d.current_url
                and "redirect_uri" not in d.current_url
            )
            logger.info("✅ 登入成功")

        except TimeoutException:
            if self.BASE_URL in driver.current_url:
                logger.info("✅ 偵測到已登入狀態")
            else:
                raise TimeoutException("登入流程超時，請確認帳密是否正確")

    def _collect_cookies(self, driver: webdriver.Chrome) -> dict:
        """收集瀏覽器 Cookie，並訪問 API 端點觸發完整 session"""
        time.sleep(2)
        cookies = {c["name"]: c["value"] for c in driver.get_cookies()}

        driver.get(f"{self.BASE_URL}/")
        time.sleep(1)
        driver.get(f"{self.BASE_URL}/api/account/user")
        time.sleep(1)

        # 合併更新後的 Cookie
        cookies.update({c["name"]: c["value"] for c in driver.get_cookies()})
        logger.info("✅ 已取得 %d 個 Cookie", len(cookies))
        return cookies

    @staticmethod
    def _extract_jwt(driver: webdriver.Chrome, cookies: dict) -> str:
        """依序嘗試從 localStorage / Cookie 取得 JWT"""
        try:
            token = driver.execute_script(
                "return localStorage.getItem('jwt') || sessionStorage.getItem('jwt');"
            )
            if token:
                logger.info("✅ 已從 localStorage 取得 JWT Token (長度: %d)", len(token))
                return token
        except Exception:
            logger.warning("⚠️ 無法從 localStorage 取得 JWT Token")

        if "jwt" in cookies:
            token = cookies["jwt"]
            logger.info("✅ 已從 Cookie 取得 JWT Token (長度: %d)", len(token))
            return token

        return ""

    def _verify_session(self, cookies: dict, jwt_token: str):
        """用 requests 驗證 Cookie 有效性"""
        logger.info("🔍 正在驗證 Cookie 有效性...")
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Cookie": "; ".join(f"{k}={v}" for k, v in cookies.items()),
            "Accept": "application/json",
            "Referer": f"{self.BASE_URL}/",
        }
        if jwt_token:
            headers["json_web_token"] = jwt_token

        try:
            resp = requests.get(
                f"{self.BASE_URL}/api/account/user",
                headers=headers,
                timeout=10,
                verify=False,
            )
            if resp.status_code == 200:
                name = resp.json().get("user_disp_name", "Unknown")
                logger.info("✅ Cookie 驗證成功！使用者: %s", name)
            else:
                logger.warning("⚠️ Cookie 驗證失敗 (HTTP %d)", resp.status_code)
        except Exception as exc:
            logger.warning("⚠️ Cookie 驗證時發生錯誤: %s", exc)

    def _cookie_string(self) -> str:
        if not self._login_result:
            return ""
        return "; ".join(f"{k}={v}" for k, v in self._login_result.cookies.items())

    @staticmethod
    def _save_error_screenshot(driver: Optional[webdriver.Chrome]):
        if driver:
            path = "login_error.png"
            driver.save_screenshot(path)
            logger.error("已儲存錯誤截圖: %s | 當前網址: %s", path, driver.current_url)


# ================= 資料庫 =================
class WeatherDatabase:
    """SQLite 資料庫操作封裝，使用 context manager 確保連線安全"""

    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS weather_data (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            port_code     TEXT NOT NULL,
            port_name     TEXT NOT NULL,
            port_id       TEXT NOT NULL,
            country       TEXT,
            issued_time   TEXT NOT NULL,
            content       TEXT NOT NULL,
            download_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(port_code, issued_time)
        )
    """

    def __init__(self, db_file: str = CONFIG.DB_FILE):
        self.db_file = db_file
        self._init_database()

    def _init_database(self):
        with self._connect() as conn:
            conn.execute(self.CREATE_TABLE_SQL)

    def _connect(self):
        """回傳帶有 WAL 模式的連線（提升並發讀取效能）"""
        conn = sqlite3.connect(self.db_file)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def get_latest_content(self, port_code: str) -> Optional[tuple]:
        """回傳 (content, issued_time, port_name) 或 None"""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT content, issued_time, port_name
                FROM weather_data
                WHERE port_code = ?
                ORDER BY issued_time DESC
                LIMIT 1
                """,
                (port_code,),
            ).fetchone()
        return row

    def get_latest_time(self, port_code: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT issued_time FROM weather_data WHERE port_code = ? ORDER BY issued_time DESC LIMIT 1",
                (port_code,),
            ).fetchone()
        return row[0] if row else None

    def save_weather(
        self,
        port: PortInfo,
        issued_time: str,
        content: str,
    ) -> bool:
        """
        儲存氣象資料，改用 PortInfo dataclass 傳入，減少參數數量。
        回傳 True 表示成功。
        """
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO weather_data
                        (port_code, port_name, port_id, country, issued_time, content, download_time)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (port.code, port.name, port.station_id, port.country, issued_time, content),
                )
            return True
        except sqlite3.Error as exc:
            logger.error("DB 寫入失敗 (%s): %s", port.code, exc)
            return False


# ================= 爬蟲主體 =================
class PortWeatherCrawler:
    """港口氣象資料爬蟲，整合登入、下載與資料庫操作"""

    API_BASE = "https://aedyn.weathernews.com/api/business/sea/portstatus/content/48h"

    def __init__(self, excel_path: str = CONFIG.EXCEL_FILE, auto_login: bool = True):
        CONFIG.validate()  # ✅ 啟動時即驗證帳密是否存在
        self.excel_path = excel_path
        self.db = WeatherDatabase()
        self.session = self._create_session()
        self.port_map: dict[str, PortInfo] = {}
        self.login_manager = AedynLoginManager(CONFIG.USERNAME, CONFIG.PASSWORD)
        self.headers: dict = {}

        self._load_port_map()
        if auto_login:
            self.refresh_cookies()

    # ── Session ──────────────────────────────────────────

    @staticmethod
    def _create_session() -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=CONFIG.MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    # ── 登入 ─────────────────────────────────────────────

    def refresh_cookies(self, headless: bool = True) -> bool:
        """重新登入並更新 Headers"""
        try:
            logger.info("🔄 正在更新 Cookie 和 JWT Token...")
            result = self.login_manager.login_and_get_cookies(headless=headless)
            self.headers = self.login_manager.get_headers()
            logger.info(
                "✅ Headers 已更新 | Cookie 數量: %d | JWT: %s",
                len(result.cookies),
                "✅ 已取得" if result.jwt_token else "❌ 未取得",
            )
            return True
        except Exception as exc:
            logger.error("❌ Cookie 更新失敗: %s", exc)
            return False

    # ── 港口資料載入 ──────────────────────────────────────

    def _load_port_map(self):
        """從 Excel 載入港口清單，轉為 PortInfo dataclass"""
        if not os.path.exists(self.excel_path):
            logger.warning("⚠️ 找不到 %s", self.excel_path)
            return

        try:
            logger.info("⏳ 正在載入港口資料...")
            df = pd.read_excel(self.excel_path, sheet_name="all_ports_list")
            df.columns = df.columns.str.strip()

            for _, row in df.iterrows():
                code = str(row["Port Code"]).strip()
                obj_id = str(row["Station ID (Object_ID)"]).strip()
                if not code or not obj_id or obj_id == "nan":
                    continue

                self.port_map[code] = PortInfo(
                    code=code,
                    name=str(row["Port Name"]).strip(),
                    station_id=obj_id,
                    country=str(row.get("Country", "N/A")),
                    latitude=self._safe_float(row.get("Lat")),
                    longitude=self._safe_float(row.get("Lon")),
                )

            logger.info("✅ 已載入 %d 個港口資料", len(self.port_map))

        except Exception as exc:
            logger.exception("❌ 讀取 Excel 失敗: %s", exc)

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        """安全轉換為 float，處理 NaN 與例外"""
        try:
            result = float(value)
            return default if np.isnan(result) else result
        except (TypeError, ValueError):
            return default

    # ── 公開 API ──────────────────────────────────────────

    def get_all_ports_display(self) -> list[str]:
        """回傳 UI 下拉選單用的清單"""
        return [f"{code} - {info.name}" for code, info in self.port_map.items()]

    def get_port_info(self, port_code: str) -> Optional[PortInfo]:
        """取得港口完整資訊"""
        info = self.port_map.get(port_code)
        if not info:
            logger.warning("❌ 港口代碼 %s 不在 port_map 中", port_code)
        return info

    def get_data_from_db(self, port_code: str) -> Optional[tuple]:
        """從資料庫讀取最新內容 (content, issued_time, port_name)"""
        return self.db.get_latest_content(port_code)

    def fetch_port_data(self, port_code: str, retry_login: bool = True) -> tuple[bool, str]:
        """
        下載單一港口氣象資料。
        
        Returns:
            (success: bool, message: str)
        """
        port = self.port_map.get(port_code)
        if not port:
            return False, f"找不到港口代碼: {port_code}"

        url = f"{self.API_BASE}/{port.station_id}.txt"
        logger.info("📡 正在下載 %s (%s)...", port_code, port.name)

        try:
            response = self.session.get(url, headers=self.headers, verify=False, timeout=CONFIG.TIMEOUT)

            if response.status_code == 200:
                return self._handle_success(response.text, port)

            if response.status_code in (401, 403) and retry_login:
                logger.warning("⚠️ Cookie 可能已過期，正在重新登入...")
                if self.refresh_cookies():
                    return self.fetch_port_data(port_code, retry_login=False)
                return False, "重新登入失敗"

            return False, f"下載失敗 (HTTP {response.status_code})"

        except requests.RequestException as exc:
            return False, f"連線錯誤: {exc}"

    def fetch_all_ports(self):
        """批次下載所有港口資料，並輸出統計摘要"""
        total = len(self.port_map)
        logger.info("🚀 開始批次下載 %d 個港口資料...", total)

        success_count = skip_count = fail_count = 0

        for i, port_code in enumerate(self.port_map, 1):
            logger.info("[%d/%d] %s", i, total, port_code)
            success, message = self.fetch_port_data(port_code)

            if success:
                if "已是最新" in message:
                    skip_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1

            logger.info("   %s", message)

        logger.info(
            "📊 下載完成！成功: %d | 略過: %d | 失敗: %d",
            success_count, skip_count, fail_count,
        )

    def test_api_connection(self):
        """測試 API 連線與認證狀態"""
        test_urls = [
            "https://aedyn.weathernews.com/api/account/user",
            "https://aedyn.weathernews.com/",
        ]
        logger.info("🧪 測試 API 連線...")
        for url in test_urls:
            try:
                resp = self.session.get(url, headers=self.headers, verify=False, timeout=10)
                logger.info("測試 %s → HTTP %d", url, resp.status_code)
                if resp.status_code == 200:
                    if "application/json" in resp.headers.get("Content-Type", ""):
                        preview = json.dumps(resp.json(), ensure_ascii=False)[:200]
                        logger.info("   回應: %s...", preview)
                    else:
                        logger.info("   回應長度: %d bytes", len(resp.text))
            except Exception as exc:
                logger.error("   ❌ 錯誤: %s", exc)

    # ── 私有輔助 ──────────────────────────────────────────

    def _handle_success(self, content: str, port: PortInfo) -> tuple[bool, str]:
        """處理 HTTP 200 回應：比對時間戳並決定是否寫入 DB"""
        issued_time = self._parse_issued_time(content)
        if self.db.get_latest_time(port.code) == issued_time:
            return True, f"資料已是最新 ({issued_time})"
        if self.db.save_weather(port, issued_time, content):
            return True, f"更新成功 ({issued_time})"
        return False, "資料庫寫入失敗"

    @staticmethod
    def _parse_issued_time(content: str) -> str:
        """從氣象文字中解析 ISSUED AT 時間戳"""
        for line in content.splitlines():
            if line.strip().startswith("ISSUED AT:"):
                return (
                    line.split(":", 1)[1]
                    .strip()
                    .replace(" UTC", "")
                    .replace(" ", "_")
                )
        return datetime.now().strftime("%Y%m%d%H%M")


# ================= 使用範例 =================
if __name__ == "__main__":
    crawler = PortWeatherCrawler(auto_login=True)

    crawler.test_api_connection()

    print("\n" + "=" * 50)
    print("範例 1: 下載單一港口資料")
    print("=" * 50)
    success, message = crawler.fetch_port_data("KAO")
    print(f"結果: {message}")

    print("\n" + "=" * 50)
    print("範例 2: 從資料庫讀取資料")
    print("=" * 50)
    data = crawler.get_data_from_db("KAO")
    if data:
        content, issued_time, port_name = data
        print(f"港口: {port_name}")
        print(f"發布時間: {issued_time}")
        print(f"內容預覽: {content[:200]}...")
