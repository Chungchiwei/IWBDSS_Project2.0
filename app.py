# app.py
"""
IWBDSS Pro — 船舶靠泊決策輔助系統

架構：
  app.py          → Streamlit 進入點，僅負責頁面流程編排
  analysis.py     → 核心氣象解析與風險計算
  models.py       → 資料結構（VesselInfo、AnalysisResult 等）
  ui_components.py → 所有 Streamlit UI 渲染
  plotting.py     → Matplotlib / Plotly 圖表
  weather_crawler.py      → 港口氣象資料爬蟲
  app_config.py   → 常數與設定
  app_helpers.py  → 共用輔助函式
  font_loader.py  → Matplotlib 中文字體設定
"""
from __future__ import annotations

import logging
import math
import traceback
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# ==================== 模組匯入 ====================

def _import_modules():
    modules = {
        "analysis":      ["WeatherParser", "WeatherAnalyzer"],
        "models":        ["VesselInfo", "AnalysisResult"],
        "font_loader":   ["ensure_chinese_font"],
        "weather_crawler":       ["PortWeatherCrawler"],
        "AI_Analyzer":   ["get_cached_ai_analyzer"],
        "app_config":    [
            "PHYSICS",
            "MOORING",
            "THRESHOLDS",
            "RISK_LEVEL_SPECS",
            "RISK_LEVEL_COLORS",
            "RISK_LEVEL_LABELS",
            "COMPASS",
            "BEAUFORT",
            "FIELD_MAPPING",
            "AppConfig",
            "score_to_risk_level",
            "risk_level_to_zh",
        ],
        "app_helpers":   ["normalize_dataframe", "AIR_DENSITY", "knots_to_ms", "compass_to_degrees"],
        "ui_components": [
            "render_sidebar", "render_port_info", "render_kpi_metrics",
            "render_detail_report", "render_chart_analysis",
            "render_ai_analysis", "render_data_list", "render_welcome_page",
            "render_berthing_advisory",
        ],
    }

    missing = []
    imported: dict = {}
    for module_name, names in modules.items():
        try:
            mod = __import__(module_name, fromlist=names)
            for name in names:
                try:
                    imported[name] = getattr(mod, name)
                except AttributeError:
                    missing.append(f"  - `{module_name}.{name}` 不存在")
        except ImportError as exc:
            missing.append(f"  - 無法匯入 `{module_name}`: {exc}")

    if missing:
        st.error(
            "❌ 以下名稱匯入失敗，請確認模組內容：\n\n"
            + "\n".join(missing)
        )
        st.stop()

    return imported


_mods = _import_modules()

# ── 繫結至本地名稱 ─────────────────────────────────────────
WeatherParser          = _mods["WeatherParser"]
WeatherAnalyzer        = _mods["WeatherAnalyzer"]
VesselInfo             = _mods["VesselInfo"]
AnalysisResult         = _mods["AnalysisResult"]
ensure_chinese_font    = _mods["ensure_chinese_font"]
PortWeatherCrawler     = _mods["PortWeatherCrawler"]
get_cached_ai_analyzer = _mods["get_cached_ai_analyzer"]

# app_config
PHYSICS                = _mods["PHYSICS"]
MOORING                = _mods["MOORING"]
THRESHOLDS             = _mods["THRESHOLDS"]
RISK_LEVEL_SPECS       = _mods["RISK_LEVEL_SPECS"]
RISK_LEVEL_COLORS      = _mods["RISK_LEVEL_COLORS"]
RISK_LEVEL_LABELS      = _mods["RISK_LEVEL_LABELS"]
COMPASS                = _mods["COMPASS"]
BEAUFORT               = _mods["BEAUFORT"]
FIELD_MAPPING          = _mods["FIELD_MAPPING"]
AppConfig              = _mods["AppConfig"]
score_to_risk_level    = _mods["score_to_risk_level"]
risk_level_to_zh       = _mods["risk_level_to_zh"]

FIXED_SAFETY_FACTOR    = MOORING.fixed_safety_factor
PERPLEXITY_API_KEY     = AppConfig.perplexity_api_key()

# app_helpers
normalize_dataframe    = _mods["normalize_dataframe"]
AIR_DENSITY            = _mods["AIR_DENSITY"]
knots_to_ms            = _mods["knots_to_ms"]
compass_to_degrees     = _mods["compass_to_degrees"]

# ui_components
render_sidebar         = _mods["render_sidebar"]
render_port_info       = _mods["render_port_info"]
render_kpi_metrics     = _mods["render_kpi_metrics"]
render_detail_report   = _mods["render_detail_report"]
render_chart_analysis  = _mods["render_chart_analysis"]
render_ai_analysis     = _mods["render_ai_analysis"]
render_data_list       = _mods["render_data_list"]
render_welcome_page        = _mods["render_welcome_page"]
render_berthing_advisory   = _mods["render_berthing_advisory"]


# ==================== 頁面配置 ====================

st.set_page_config(
    page_title="IWBDSS 靠泊決策系統",
    layout="wide",
    page_icon="🚢",
    initial_sidebar_state="expanded",
)

ensure_chinese_font()


# ==================== Session State ====================

_SESSION_DEFAULTS: dict = {
    "analyzer":    None,
    "result":      None,
    "vessel":      None,
    "df_analysis": None,
    "ai_analyzer": None,
    "port_info":   None,
}

for _key, _default in _SESSION_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ==================== 服務初始化 ====================

@st.cache_resource
def _get_crawler() -> PortWeatherCrawler:
    return PortWeatherCrawler()


@st.cache_resource
def _get_ai_analyzer():
    return get_cached_ai_analyzer(PERPLEXITY_API_KEY)


crawler = _get_crawler()

if st.session_state.ai_analyzer is None:
    st.session_state.ai_analyzer = _get_ai_analyzer()


# ==================== 物理計算輔助函式 ====================
# 從 analysis.py 的私有邏輯提取為 app.py 層級的獨立函式，
# 供 _build_detail_dataframe 逐行計算使用。

# 纜繩效率係數（與 analysis.py MooringEfficiency 一致）
_HEAD_TRANS   = 0.95
_HEAD_LONG    = 0.15
_SPRING_TRANS = 0.25
_SPRING_LONG  = 0.95

# 拖船換算係數（與 analysis.py 一致）
_HP_TO_BOLLARD_PER_100HP = 1.1
_GRAVITY                 = 9.81


def _compute_vessel_heading(vessel: VesselInfo) -> float:
    """
    依靠泊舷側計算船艏向。

    與 WeatherAnalyzer._vessel_heading 邏輯完全一致：
      - 右靠 (starboard)：heading = berth_direction + 180°
      - 左靠 (port)：      heading = berth_direction
    """
    side = str(vessel.berthing_side).lower().strip()
    if any(k in side for k in ("starboard", "右", "s", "stbd")):
        return (vessel.berth_direction + 180) % 360
    return float(vessel.berth_direction)


def _compute_wind_force_row(
    wind_gust_kts: float,
    wind_dir_deg:  float,
    vessel:        VesselInfo,
    heading:       float,
) -> tuple[float, float, str]:
    """
    計算單筆記錄的風力。

    Returns:
        (total_force_N, transverse_force_N, wind_type)

    與 WeatherAnalyzer._calc_wind_force 邏輯完全一致。
    """
    wind_ms  = knots_to_ms(wind_gust_kts)
    relative = (wind_dir_deg - heading + 180) % 360 - 180
    abs_rel  = abs(relative)

    if 45 <= abs_rel <= 135:
        wind_type = "offshore" if relative > 0 else "onshore"
    else:
        wind_type = "parallel"

    drag_coef = getattr(vessel, "wind_drag_coef", 1.0)
    total_N   = 0.5 * AIR_DENSITY * drag_coef * vessel.wind_area * wind_ms ** 2
    rad       = math.radians(abs_rel)

    trans_N   = total_N * abs(math.sin(rad))
    return total_N, trans_N, wind_type


def _compute_mooring_restraint_kN(vessel: VesselInfo) -> float:
    """
    計算纜繩橫向抗力 (kN)。

    與 WeatherAnalyzer._calc_mooring_restraint 邏輯完全一致。
    """
    wll_N        = vessel.mbl * vessel.safety_factor  # WLL = MBL × 0.33 → 等效 SF=3.0
    head_count   = vessel.bow_lines + vessel.stern_lines
    spring_count = vessel.bow_spring_lines + vessel.stern_spring_lines
    trans_N      = (
        head_count   * wll_N * _HEAD_TRANS
        + spring_count * wll_N * _SPRING_TRANS
    )
    return trans_N / 1000.0


def _compute_tug_restraint_kN(tug_count: int, tug_hp: float) -> float:
    """
    計算拖船助力 (kN)。

    與 WeatherAnalyzer._calc_tug_force 邏輯完全一致。
    """
    if tug_count <= 0:
        return 0.0
    bollard_ton = (tug_hp / 100.0) * _HP_TO_BOLLARD_PER_100HP
    return tug_count * bollard_ton * _GRAVITY


# ==================== 輔助函式 ====================

def _format_stay_duration(td: timedelta) -> str:
    """
    將 timedelta 格式化為「X天 Y小時 Z分」。

    Examples:
        >>> _format_stay_duration(timedelta(hours=25, minutes=30))
        '1天 1小時 30分'
    """
    total            = int(td.total_seconds())
    days, remainder  = divmod(total, 86400)
    hours, seconds   = divmod(remainder, 3600)
    minutes          = seconds // 60
    return f"{days}天 {hours}小時 {minutes}分"


# ── 靠泊側標準化對照表 ────────────────────────────────────
_SIDE_NORMALISE: dict[str, str] = {
    "port":         "port",
    "starboard":    "starboard",
    "左靠 (port)":  "port",
    "右靠 (stbd)":  "starboard",
    "左靠 (Port)":  "port",
    "右靠 (Stbd)":  "starboard",
}


def _normalise_side(raw: str) -> str:
    """將任意格式靠泊側字串統一轉為 'port' 或 'starboard'"""
    normalised = _SIDE_NORMALISE.get(raw.strip())
    if normalised is None:
        logger.warning("berthing_side 值 '%s' 不在標準清單中，預設使用 'port'", raw)
        return "port"
    return normalised


def _build_vessel(sidebar_data: dict) -> VesselInfo:
    """
    從側邊欄輸入資料建立 VesselInfo。

    MBL 單位轉換：sidebar 輸入為 kN，VesselInfo 儲存為 N。
    """
    arrival   = sidebar_data["arrival"]
    departure = sidebar_data["departure"]

    return VesselInfo(
        berth_direction    = sidebar_data["berth_dir"],
        berthing_side      = _normalise_side(sidebar_data["side"]),
        arrival_time       = arrival,
        departure_time     = departure,
        stay_duration      = _format_stay_duration(departure - arrival),
        draft_bow          = sidebar_data["draft_b"],
        draft_stern        = sidebar_data["draft_s"],
        mbl                = sidebar_data["mbs"] * 1000.0,
        total_lines        = (
            sidebar_data["bh"] + sidebar_data["bs"]
            + sidebar_data["sh"] + sidebar_data["ss"]
        ),
        safety_factor      = FIXED_SAFETY_FACTOR,
        tug_hp             = sidebar_data["tug_hp"],
        bow_lines          = sidebar_data["bh"],
        bow_spring_lines   = sidebar_data["bs"],
        stern_lines        = sidebar_data["sh"],
        stern_spring_lines = sidebar_data["ss"],
        wind_area          = sidebar_data["area"],
        wind_drag_coef     = sidebar_data.get("cd", 1.0),
    )


def _build_detail_dataframe(
    analyzer: WeatherAnalyzer,
    vessel:   VesselInfo,
    result:   AnalysisResult,
) -> pd.DataFrame:
    """
    建立詳細氣象 + 風力計算 DataFrame，供圖表與列表使用。
    """
    # Step 1：標準化 DataFrame
    df = normalize_dataframe(analyzer, vessel)

    # Step 2：固定參數
    heading            = _compute_vessel_heading(vessel)
    tug_final_count    = _get_tug_final_count(result)
    mooring_kN         = _compute_mooring_restraint_kN(vessel)
    tug_kN             = _compute_tug_restraint_kN(tug_final_count, vessel.tug_hp)
    total_restraint_kN = mooring_kN + tug_kN

    # Step 3：逐行計算風力，確保每個值都是純量
    gust_force_list:  list[float] = []
    is_offshore_list: list[int]   = []
    sf_list:          list[float] = []

    for _, row in df.iterrows():
        # 用 .item() 或 float() 強制從 Series/ndarray 取出純量
        gust_kts = float(row["wind_gust_kts"]) if "wind_gust_kts" in df.columns else 0.0
        dir_deg  = float(row["wind_dir_deg"])  if "wind_dir_deg"  in df.columns else 0.0

        total_N, trans_N, wind_type = _compute_wind_force_row(
            wind_gust_kts = gust_kts,
            wind_dir_deg  = dir_deg,
            vessel        = vessel,
            heading       = heading,
        )

        req_kN = trans_N / 1000.0
        sf     = total_restraint_kN / req_kN if req_kN > 0.1 else 99.9

        gust_force_list.append(float(total_N))
        is_offshore_list.append(1 if wind_type == "offshore" else 0)
        sf_list.append(round(float(sf), 3))

    # Step 4：直接指派欄位，避免 concat 索引錯位
    df = df.reset_index(drop=True)
    df["gust_force_N"]  = gust_force_list
    df["is_offshore"]   = is_offshore_list
    df["safety_factor"] = sf_list

    return df


def _get_tug_final_count(result: AnalysisResult) -> int:
    """相容 TugRecommendation dataclass 與舊版 dict，取得最終拖船數"""
    tug = result.tug_recommendation
    if hasattr(tug, "final_tug_count"):
        return int(tug.final_tug_count)
    if isinstance(tug, dict):
        return int(tug.get("final_tug_count", 0))
    return 0


def _get_attr_or_key(obj: object, attr: str, default):
    """通用相容器：優先取 dataclass 屬性，退回 dict key"""
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return default


# ==================== 分析邏輯 ====================

def _run_analysis(sidebar_data: dict) -> None:
    """
    執行完整分析流程並將結果寫入 session_state。

    步驟：
      1. 建立 VesselInfo
      2. 呼叫 analyzer.analyze(vessel) 取得 AnalysisResult
      3. 建立詳細 DataFrame（含風力與安全係數）
      4. 將結果寫入 session_state
    """
    analyzer = st.session_state.analyzer

    with st.spinner("🔄 正在進行多維度風險分析..."):
        vessel    = _build_vessel(sidebar_data)
        result    = analyzer.analyze(vessel)
        df_detail = _build_detail_dataframe(analyzer, vessel, result)

        st.session_state.result      = result
        st.session_state.vessel      = vessel
        st.session_state.df_analysis = df_detail

    st.success("✅ 分析完成！")
    logger.info(
        "分析完成 — 港口：%s，風險：%s（%.1f）",
        analyzer.port_name, result.risk_level, result.risk_score,
    )


# ==================== 側邊欄 ====================

sidebar_data = render_sidebar(crawler)


# ==================== 分析觸發 ====================

if sidebar_data["btn_analyze"] and st.session_state.analyzer:
    try:
        _run_analysis(sidebar_data)
    except ValueError as exc:
        st.error(f"❌ 船舶參數驗證失敗：{exc}")
        logger.warning("VesselInfo 驗證失敗", exc_info=True)
    except Exception:
        st.error("❌ 分析過程發生未預期錯誤，請查看詳細訊息")
        with st.expander("🔍 詳細錯誤訊息"):
            st.code(traceback.format_exc())
        logger.exception("分析流程發生未預期錯誤")


# ==================== 結果顯示 ====================

if st.session_state.result:
    res       = st.session_state.result
    ves       = st.session_state.vessel
    analyzer  = st.session_state.analyzer
    df_detail = st.session_state.df_analysis

    render_port_info(st.session_state.port_info, analyzer)
    render_kpi_metrics(res)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 詳細報告",
        "📊 圖表分析",
        "🤖 AI 輔助",
        "💾 數據列表",
    ])

    with tab1:
        render_berthing_advisory(res, ves, analyzer)
        render_detail_report(res, sidebar_data, df_detail, analyzer)

    with tab2:
        render_chart_analysis(analyzer, ves, res, df_detail, sidebar_data)

    with tab3:
        render_ai_analysis(
            sidebar_data["enable_ai"],
            sidebar_data.get("ai_mode"),
            res, sidebar_data, df_detail, analyzer,
        )

    with tab4:
        render_data_list(df_detail, analyzer, sidebar_data)

elif not st.session_state.analyzer:
    render_welcome_page()


# ==================== 頁腳 ====================

st.markdown("---")
st.caption("IWBDSS Pro v2.1 | © 2025 Maritime Safety Systems")
