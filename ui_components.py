# ui_components.py
"""
UI 組件模組

負責所有 Streamlit UI 渲染，包含：
  - 側邊欄輸入（港口選擇、船舶參數）
  - 港口資訊卡片
  - KPI 指標
  - 詳細報告（風險警示、纜繩、拖船、受力）
  - 圖表分析（Matplotlib + Plotly）
  - AI 分析
  - 數據列表

設計原則：
  - 所有 result 存取透過 dataclass 屬性，不使用裸露 dict
  - port_info 使用 PortDisplayInfo dataclass
  - 風險顏色統一由 RISK_LEVEL_SPECS 提供
  - 重複邏輯抽出為私有輔助函式
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from datetime import datetime

from AI_Analyzer import AnalysisResults, HourlyRiskEntry, VesselParams
from analysis import WeatherAnalyzer, WeatherParser
from app_config import RISK_LEVEL_SPECS
from app_helpers import (
    PortDisplayInfo,
    calculate_mooring_capacity,
    calculate_tug_capacity,
    get_port_full_info,
)
from models import AnalysisResult, VesselInfo
from plotting import PlotService, plot_enhanced_timeline

# ── 新增：船型資料庫 ──────────────────────────────────────────
from vessel_windage_db import (
    VESSEL_TYPE_DISPLAY,
    VESSEL_TYPE_KEY_MAP,
    lookup_windage_area,
    get_windage_stats,
)

logger = logging.getLogger(__name__)


# ================= 私有輔助函式 =================

def _risk_row_style(risk_level_zh: str, num_cols: int) -> list[str]:
    """
    依中文風險等級回傳 DataFrame 列背景色。
    使用 RISK_LEVEL_SPECS 的 color_bg，不硬寫顏色碼。
    """
    # key must match RISK_LEVEL_SPECS.name_zh exactly
    _ZH_TO_KEY = {spec.name_zh: key for key, spec in RISK_LEVEL_SPECS.items()}
    key   = _ZH_TO_KEY.get(risk_level_zh, "low")
    spec  = RISK_LEVEL_SPECS.get(key)
    color = spec.color_bg if spec else ""
    return [f"background-color: {color}" if color else ""] * num_cols


def _classify_op_risk(gust_kts: float, wind_kts: float, wave_m: float) -> str:
    """依作業閾值（航海標準）分類風險等級 key"""
    def _tier(val: float, thr: dict) -> int:
        if val >= thr["danger"]:  return 3
        if val >= thr["warning"]: return 2
        if val >= thr["caution"]: return 1
        return 0
    level = max(_tier(gust_kts, _GUST_THR),
                _tier(wind_kts, _WIND_THR),
                _tier(wave_m,   _WAVE_THR))
    return ["low", "medium", "high", "extreme"][level]


def _render_recommendation(rec: str) -> None:
    """依建議文字前綴選擇適當的 Streamlit 顯示元件"""
    _PREFIX_MAP = {
        "🚨": st.error,
        "⚠️": st.warning,
        "🌙": st.info,
        "✅": st.success,
        "🚤": st.info,
    }
    for prefix, fn in _PREFIX_MAP.items():
        if prefix in rec:
            fn(rec)
            return
    st.write(f"- {rec}")


def _get_wfs_value(wfs: Any, attr: str, dict_key: str, default: float = 0.0) -> float:
    """
    相容新版 WindForceSummary dataclass 與舊版 dict 的通用存取器。
    """
    if hasattr(wfs, attr):
        return float(getattr(wfs, attr, default))
    if isinstance(wfs, dict):
        return float(wfs.get(dict_key, default))
    return default


def _get_tug_value(tug: Any, attr: str, dict_key: str, default: Any = None) -> Any:
    """相容 TugRecommendation dataclass 與舊版 dict 的通用存取器"""
    if hasattr(tug, attr):
        return getattr(tug, attr, default)
    if isinstance(tug, dict):
        return tug.get(dict_key, default)
    return default


def _build_hourly_risk_entries(df_detail: pd.DataFrame) -> List[HourlyRiskEntry]:
    """
    將 df_detail 轉換為 HourlyRiskEntry 列表，供 AI 分析使用。
    只取 safety_factor < 1.5 的高風險時段（最多 20 筆，避免 prompt 過長）。
    """
    required = {"wind_gust_kts", "safety_factor"}
    if not required.issubset(df_detail.columns):
        return []

    entries: List[HourlyRiskEntry] = []
    for _, row in df_detail.iterrows():
        sf = float(row.get("safety_factor", 99.9))
        if sf >= 1.5:
            continue

        t_raw = row.get("time", "")
        if hasattr(t_raw, "strftime"):
            t_str = t_raw.strftime("%Y-%m-%d %H:%M")
        else:
            t_str = str(t_raw)

        entries.append(
            HourlyRiskEntry(
                time          = t_str,
                wind_gust_kts = float(row.get("wind_gust_kts", 0.0)),
                safety_factor = sf,
            )
        )

    return entries[:20]


# ================= 側邊欄 =================

def render_sidebar(crawler: Any) -> Dict[str, Any]:
    """
    渲染側邊欄並回傳所有輸入參數。
    """
    with st.sidebar:
        st.title("🚢 IWBDSS Pro")
        st.caption("Integrated Weather & Berthing Decision Support System")

        # ── 1. 港口選擇 ──────────────────────────────────────
        st.header("1. 選擇港口")
        content_to_parse   = None
        selected_port_code = None
        p_name             = None

        ports = crawler.get_all_ports_display()
        if not ports:
            st.error("❌ 無法載入港口清單")
        else:
            selected_port_str  = st.selectbox("選擇港口", ports, label_visibility="collapsed")
            p_code, p_name     = selected_port_str.split(" - ", 1)
            selected_port_code = p_code

            db_data = crawler.get_data_from_db(p_code)
            col_status, col_update = st.columns([3, 1])

            if db_data:
                content, t_str, _name_en, _name_zh = db_data
                col_status.success(f"📅 {t_str}")
                content_to_parse = content
            else:
                col_status.warning("⚠️ 無本地資料")

            if col_update.button("🔄", help="更新氣象資料"):
                with st.spinner(f"下載 {p_name} 資料中..."):
                    ok, msg = crawler.fetch_port_data(p_code)
                    if ok:
                        st.toast(f"✅ {msg}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

        # ── 氣象資料解析 ─────────────────────────────────────
        if content_to_parse:
            _parse_weather_content(
                content_to_parse, p_name, selected_port_code, crawler
            )

        st.markdown("---")

        # ── 2. 參數設定 ──────────────────────────────────────
        st.header("2. 參數設定")
        min_t = max_t = datetime.now()
        if st.session_state.get("analyzer"):
            min_t, max_t = st.session_state.analyzer.time_range()

        c1, c2 = st.columns(2)
        arr_dt = c1.date_input("靠泊日期", min_t)
        arr_tm = c1.time_input("靠泊時間", min_t.time())
        dep_dt = c2.date_input("離泊日期", max_t)
        dep_tm = c2.time_input("離泊時間", max_t.time())

        arrival   = datetime.combine(arr_dt, arr_tm)
        departure = datetime.combine(dep_dt, dep_tm)

        berth_dir = st.number_input("泊位方位 (0–360°)", 0, 360, 0)
        side = st.selectbox(
            "靠泊側",
            options=["port", "starboard"],
            format_func=lambda x: "左靠 (Port)" if x == "port" else "右靠 (Stbd)",
        )

        # ── ⚓ 船舶細節（含船型自動帶入受風面積）────────────
        with st.expander("⚓ 船舶細節", expanded=True):

            # 船型選擇
            type_options = list(VESSEL_TYPE_DISPLAY.values()) + ["手動輸入"]
            vessel_type_display = st.selectbox(
                "船型",
                options=type_options,
                help="選擇船型後，系統將依吃水自動帶入受風面積",
            )
            vessel_type_key = VESSEL_TYPE_KEY_MAP.get(vessel_type_display)  # None → 手動

            draft_b = st.number_input("船艏吃水 (m)", 0.0, 20.0, 11.0, 0.1)
            draft_s = st.number_input("船艉吃水 (m)", 0.0, 20.0, 10.0, 0.1)

            # 自動查找受風面積
            auto_area: Optional[float] = None
            if vessel_type_key:
                lookup = lookup_windage_area(vessel_type_key, draft_b, draft_s)
                if lookup:
                    auto_area = lookup.windage_area

                    # 顯示查找結果
                    st.success(
                        f"📐 自動帶入受風面積：**{auto_area:,.0f} m²**\n\n"
                        f"🔍 {lookup.method}"
                    )

                    # 顯示前 3 近候選紀錄
                    with st.expander("🔎 查看前 3 近候選紀錄", expanded=False):
                        for i, cand in enumerate(lookup.candidates, 1):
                            st.caption(
                                f"**{i}.** {cand.vessel_name} @ {cand.port} ｜ "
                                f"艏 {cand.draft_fwd} m / 艉 {cand.draft_aft} m "
                                f"→ **{cand.windage_area:,.0f} m²**"
                            )

                    # 顯示該船型受風面積統計範圍
                    stats = get_windage_stats(vessel_type_key)
                    if stats:
                        st.caption(
                            f"📊 {vessel_type_display} 資料範圍："
                            f"{stats['min']:,.0f} ~ {stats['max']:,.0f} m²"
                            f"（平均 {stats['mean']:,.0f} m²，共 {int(stats['count'])} 筆）"
                        )

            # 受風面積輸入：自動帶入或手動輸入
            area = st.number_input(
                "受風面積 (m²)",
                min_value  = 100.0,
                max_value  = 20000.0,
                value      = float(auto_area) if auto_area is not None else 9000.0,
                step       = 100.0,
                help       = "已根據船型與吃水自動帶入，可直接手動覆蓋",
            )

            tug_hp = st.number_input("拖船馬力 (HP)", 0, 10000, 4000, 100)
            cd     = st.slider("風阻係數", 0.5, 1.5, 1.0, 0.05)

        with st.expander("🔗 纜繩配置", expanded=True):
            mbs = st.number_input("MBL (kN)", 100.0, 2000.0, 1000.0, 50.0)
            st.markdown("**纜繩數量配置**")
            c1, c2 = st.columns(2)
            bh = c1.number_input("艏-頭纜", 0, 10, 4)
            bs = c2.number_input("艏-倒纜", 0, 10, 2)
            sh = c1.number_input("艉-尾纜", 0, 10, 4)
            ss = c2.number_input("艉-倒纜", 0, 10, 2)
            st.info(f"📊 總纜繩數: **{bh + bs + sh + ss}** 條")

        st.markdown("---")

        # ── AI 設定 ───────────────────────────────────────────
        st.subheader("🤖 AI 決策輔助")
        enable_ai = st.checkbox("啟用 AI 分析", value=True)
        ai_mode   = st.radio("分析模式", ["快速摘要", "完整分析"]) if enable_ai else None

        st.markdown("---")

        btn_analyze = st.button(
            "🚀 開始分析",
            type="primary",
            disabled=not st.session_state.get("analyzer"),
            use_container_width=True,
        )

        if not st.session_state.get("analyzer"):
            st.info("💡 請先選擇港口並載入氣象資料")

    return {
        "arrival":    arrival,    "departure":  departure,
        "berth_dir":  berth_dir,  "side":       side,
        "draft_b":    draft_b,    "draft_s":    draft_s,
        "area":       area,       "tug_hp":     tug_hp,
        "cd":         cd,         "mbs":        mbs,
        "bh":         bh,         "bs":         bs,
        "sh":         sh,         "ss":         ss,
        "enable_ai":  enable_ai,  "ai_mode":    ai_mode,
        "btn_analyze": btn_analyze,
        # 額外回傳船型資訊，供後續模組使用
        "vessel_type_display": vessel_type_display,
        "vessel_type_key":     vessel_type_key,
    }


def _parse_weather_content(
    content: str,
    p_name: Optional[str],
    port_code: Optional[str],
    crawler: Any,
) -> None:
    """解析氣象內容並寫入 session_state"""
    parser = WeatherParser()
    try:
        p_parsed_name, data, conditions, warns = parser.parse_content(content)
        final_name = p_name or p_parsed_name

        if port_code:
            st.session_state.port_info = get_port_full_info(port_code, crawler)

        st.session_state.analyzer = WeatherAnalyzer(final_name, data, conditions=conditions)
        st.success(f"✅ 已載入 {len(data)} 筆氣象資料")

        if warns:
            with st.expander(f"⚠️ 解析警告 ({len(warns)})"):
                for w in warns:
                    st.warning(w)

    except Exception:
        logger.exception("氣象資料解析失敗（port_code=%s）", port_code)
        st.error("❌ 解析失敗，請確認資料格式")
        st.session_state.analyzer = None


# ================= 港口資訊卡片 =================

def render_port_info(
    port_info: Optional[PortDisplayInfo],
    analyzer: Optional[WeatherAnalyzer],
) -> None:
    """渲染港口資訊卡片（強化版）"""
    has_valid_info = (
        port_info is not None
        and (
            getattr(port_info, "port_name", None) not in (None, "N/A")
            or (isinstance(port_info, dict) and port_info.get("port_name") not in (None, "N/A"))
        )
    )

    if has_valid_info:
        name       = getattr(port_info, "port_name",  None) or (port_info.get("port_name",  "") if isinstance(port_info, dict) else "")
        code       = getattr(port_info, "port_code",  None) or (port_info.get("port_code",  "") if isinstance(port_info, dict) else "")
        country    = getattr(port_info, "country",    None) or (port_info.get("country",    "") if isinstance(port_info, dict) else "")
        station_id = getattr(port_info, "station_id", None) or (port_info.get("station_id", "") if isinstance(port_info, dict) else "")
        lat_ns     = getattr(port_info, "lat_ns",     None) or (port_info.get("lat_ns",     "") if isinstance(port_info, dict) else "")
        lon_ew     = getattr(port_info, "lon_ew",     None) or (port_info.get("lon_ew",     "") if isinstance(port_info, dict) else "")
    else:
        name       = getattr(analyzer, "port_name", "Unknown Port") if analyzer else "Unknown Port"
        code = country = station_id = lat_ns = lon_ew = ""

    # ── 港口名稱大標題 ────────────────────────────────────────
    st.markdown(
        f"<div style='text-align:center;margin:8px 0 4px'>"
        f"<span style='font-size:2.4em;font-weight:800;letter-spacing:2px;"
        f"background:linear-gradient(90deg,#1E40AF,#7C3AED);-webkit-background-clip:text;"
        f"-webkit-text-fill-color:transparent;'>⚓ {name.upper()}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not has_valid_info:
        return

    # ── 主資訊列 ─────────────────────────────────────────────
    items = [
        ("Port Code",    code,       country),
        ("Station ID",   station_id, "WNI Weather Station"),
        ("Coordinates",  f"{lat_ns}", f"{lon_ew}"),
    ]
    cols_html = "".join(
        f"<div style='text-align:center;padding:8px 0;border-right:1px solid rgba(255,255,255,0.2)'>"
        f"<div style='font-size:0.72em;letter-spacing:1.5px;opacity:0.75;text-transform:uppercase'>{label}</div>"
        f"<div style='font-size:1.15em;font-weight:700;margin:3px 0'>{val}</div>"
        f"<div style='font-size:0.75em;opacity:0.7'>{sub}</div>"
        f"</div>"
        for label, val, sub in items
    )
    st.markdown(
        f"<div style='background:linear-gradient(135deg,#1E3A8A 0%,#4C1D95 60%,#7C3AED 100%);"
        f"padding:18px 24px;border-radius:14px;margin:6px 0 14px;color:white;"
        f"box-shadow:0 4px 20px rgba(79,70,229,0.35);'>"
        f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:0'>"
        f"{cols_html}"
        f"</div></div>",
        unsafe_allow_html=True,
    )


# ================= KPI 指標 =================

# 各風險等級對應的具體舒緩措施
_MITIGATION_STEPS: dict = {
    "medium": [
        "增加艏艉頭纜各 1 條，強化橫向抗力",
        "確認拖船就位並處於隨時可動狀態",
        "每小時巡視全部纜繩張力與護舷器位置",
        "密切追蹤氣象預報，風況惡化時立即升級警戒",
    ],
    "high": [
        "立即增加頭纜 / 倒纜各 1–2 條（艏艉均需加強）",
        "安排額外拖船 1 艘於船舷待命，隨時提供側向抵抗力",
        "每 30 分鐘巡視纜繩，記錄張力計讀數",
        "通知港務局並回報公司輪管部，取得額外纜繩資源授權",
        "評估提前離泊或移泊至遮蔽錨地的可行性",
    ],
    "extreme": [
        "全數纜繩加倍（至少艏艉各 6 條）並確認纜樁負荷上限",
        "立即申請增派 2 艘拖船在船側頂推，防止船體偏移",
        "停止一切貨物作業，甲板人員縮減至最低必要值班人數",
        "備妥緊急離泊程序：主機備車、VHF CH16 持續值守",
        "聯繫引航站及港口主管機關，評估封港或強制離泊命令",
        "必要時立即執行緊急離泊，駛往避風錨地",
    ],
}


def render_kpi_metrics(result: AnalysisResult) -> None:
    """渲染 KPI 指標列與風險舒緩措施橫幅"""
    wfs         = result.wind_force_summary
    max_force_N = _get_wfs_value(wfs, "max_gust_force_N", "max_gust_force_N")
    sf          = _get_wfs_value(wfs, "safety_factor",    "safety_factor")

    bow_status  = result.mooring_split.bow.status
    delta_color = "normal" if bow_status == "OK" else "inverse"
    risk_delta  = result.risk_score - result.mitigated_risk_score

    risk_spec  = RISK_LEVEL_SPECS.get(result.risk_level, RISK_LEVEL_SPECS["low"])
    risk_label = risk_spec.label

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("綜合風險評分", f"{result.risk_score:.1f} / 100", risk_label,         delta_color="inverse")
    k2.metric("緩解後風險",   f"{result.mitigated_risk_score:.1f}", f"降幅 {risk_delta:.1f}")
    k3.metric("最大受力",     f"{max_force_N / 1000:.0f} kN",       f"SF {sf:.2f}")
    k4.metric("繫泊狀態",     bow_status,                            delta_color=delta_color)

    # ── 風險舒緩橫幅（medium 以上才顯示）────────────────────
    steps = _MITIGATION_STEPS.get(result.risk_level)
    if not steps:
        return

    if result.risk_level == "extreme":
        icon, title = "🚨", f"極度危險（評分 {result.risk_score:.0f}/100）— 立即採取以下緊急措施"
    elif result.risk_level == "high":
        icon, title = "⚠️", f"高度風險（評分 {result.risk_score:.0f}/100）— 建議執行以下舒緩措施"
    else:
        icon, title = "📋", f"中度風險（評分 {result.risk_score:.0f}/100）— 建議採取以下預防措施"

    steps_html = "".join(
        f"<div style='display:flex;align-items:flex-start;gap:8px;margin-bottom:5px'>"
        f"<span style='color:{risk_spec.color_hex};font-weight:700;flex-shrink:0'>▶</span>"
        f"<span style='color:#374151;font-size:0.9em'>{s}</span></div>"
        for s in steps
    )
    st.markdown(
        f"<div style='background:{risk_spec.color_bg};border-left:6px solid {risk_spec.color_hex};"
        f"padding:14px 18px;border-radius:8px;margin-top:10px'>"
        f"<div style='font-weight:700;color:{risk_spec.color_hex};font-size:1em;margin-bottom:10px'>"
        f"{icon} {title}</div>"
        f"{steps_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ================= 靠離泊作業建議 =================

def _wind_type_label(wind_type: str) -> str:
    """將 wind_type 英文代碼轉為中文標籤"""
    return {
        "offshore": "吹開風",
        "onshore":  "攏風",
        "parallel": "側風",    # 舊版相容
        "headwind": "逆風",
        "tailwind": "順風",
    }.get(wind_type, "未知")


def _wind_type_color(wind_type: str) -> str:
    """風型對應警示顏色"""
    return {
        "offshore": "green",
        "onshore":  "red",
        "parallel": "orange",  # 舊版相容
        "headwind": "blue",
        "tailwind": "blue",
    }.get(wind_type, "gray")


def _wave_period_type(period_s: float) -> str:
    """根據波浪週期判斷浪型"""
    if period_s <= 0:
        return "N/A"
    if period_s >= 10:
        return f"{period_s:.1f}s（長浪/湧浪）"
    if period_s >= 6:
        return f"{period_s:.1f}s（混合浪）"
    return f"{period_s:.1f}s（風浪）"


# ── 閾值常數 ──────────────────────────────────────────────────
_GUST_THR  = {"caution": 28.0, "warning": 34.0, "danger": 41.0}
_WIND_THR  = {"caution": 22.0, "warning": 28.0, "danger": 34.0}
_WAVE_THR  = {"caution": 2.5,  "warning": 3.5,  "danger": 4.0}


def _thr_color(val: float, thr: dict) -> str:
    """依閾值回傳 CSS 顏色（正常→灰、警戒→藍、警告→橙、危險→紅）"""
    if val >= thr["danger"]:
        return "#B91C1C"
    if val >= thr["warning"]:
        return "#B45309"
    if val >= thr["caution"]:
        return "#1D4ED8"
    return "#374151"


def _metric_html(label: str, value: str, color: str, note: str = "") -> str:
    """回傳一個有顏色的指標 HTML 區塊"""
    note_html = f"<div style='font-size:0.75em;color:#6B7280;margin-top:1px'>{note}</div>" if note else ""
    return (
        f"<div style='background:#F9FAFB;border-radius:6px;padding:8px 10px;"
        f"margin-bottom:6px;border-left:3px solid {color}'>"
        f"<div style='font-size:0.75em;color:#6B7280'>{label}</div>"
        f"<div style='font-size:1.1em;font-weight:bold;color:{color}'>{value}</div>"
        f"{note_html}</div>"
    )


def _wx_zh(codes: list) -> str:
    _MAP = {"CLR": "晴", "CLOUDY": "多雲", "OVERCAST": "陰", "FOG": "霧",
            "MIST": "薄霧", "RAIN": "雨", "DRIZZLE": "毛雨", "SNOW": "雪",
            "THUNDER": "雷暴", "SLEET": "雨夾雪", "HAZE": "霾"}
    return "、".join(dict.fromkeys(_MAP.get(c, c) for c in codes)) or "N/A"


def _render_op_col(col, window, label: str, op_time, header_color: str) -> None:
    """在 Streamlit column 內渲染單一作業時窗卡片"""
    with col:
        is_night = op_time.hour >= 20 or op_time.hour < 6
        night_tag = " 🌙" if is_night else ""
        st.markdown(
            f"<div style='background:{header_color};color:#fff;padding:8px 12px;"
            f"border-radius:8px 8px 0 0;font-weight:bold;font-size:1em'>"
            f"{label}{night_tag}"
            f"<span style='font-weight:normal;font-size:0.85em;margin-left:8px;opacity:0.9'>"
            f"{op_time.strftime('%m/%d %H:%M')}</span></div>",
            unsafe_allow_html=True,
        )

        if window is None or not getattr(window, "has_data", False):
            st.markdown(
                "<div style='background:#F3F4F6;border-radius:0 0 8px 8px;"
                "padding:16px;text-align:center;color:#9CA3AF'>無該時段氣象資料</div>",
                unsafe_allow_html=True,
            )
            return

        # 取值
        max_gust   = getattr(window, "max_wind_gust",      0.0) or 0.0
        max_dir    = getattr(window, "max_wind_direction",  "")
        hr_hours   = getattr(window, "high_risk_hours",     0)  or 0
        max_wave   = getattr(window, "max_wave_height",     0.0) or 0.0
        max_period = getattr(window, "max_wave_period",     0.0) or 0.0
        avg_temp   = getattr(window, "avg_temp",            None)
        min_vis_m  = getattr(window, "min_vis_m",           None)
        wx_codes   = getattr(window, "weather_codes",       [])
        ws         = getattr(window, "window_start",        None)
        we         = getattr(window, "window_end",          None)
        wt         = getattr(window, "dominant_wind_type",  "")
        all_risks  = list(getattr(window, "risks", [])) + list(getattr(window, "condition_risks", []))

        # 卡片主體
        body_parts = []

        # 風型 badge
        wt_label = _wind_type_label(wt)
        wt_color = _wind_type_color(wt)
        wt_icon  = {"offshore": "⬆", "onshore": "⬇", "headwind": "⬅", "tailwind": "➡", "parallel": "↔"}.get(wt, "↔")
        body_parts.append(
            f"<div style='background:{wt_color}18;border:1px solid {wt_color}44;"
            f"border-radius:4px;padding:4px 8px;margin-bottom:8px;display:inline-block;"
            f"color:{wt_color};font-weight:bold;font-size:0.9em'>"
            f"{wt_icon} {wt_label}</div>"
        )

        # 時窗
        if ws and we:
            body_parts.append(
                f"<div style='font-size:0.75em;color:#6B7280;margin-bottom:6px'>"
                f"⏱ 時窗 {ws.strftime('%H:%M')}–{we.strftime('%H:%M')}</div>"
            )

        # 最大陣風
        gust_color = _thr_color(max_gust, _GUST_THR)
        gust_note  = f"↑ {max_dir}" if max_dir else ""
        body_parts.append(_metric_html("最大陣風", f"{max_gust:.1f} kts", gust_color, gust_note))

        # 高風險時段
        risk_color = "#B91C1C" if hr_hours >= 6 else "#B45309" if hr_hours >= 3 else "#1D4ED8" if hr_hours >= 1 else "#374151"
        body_parts.append(_metric_html("高風險時段", f"{hr_hours} 小時", risk_color))

        # 最大浪高
        wave_color = _thr_color(max_wave, _WAVE_THR)
        wave_note  = _wave_period_type(max_period) if max_period > 0 else ""
        body_parts.append(_metric_html("最大浪高", f"{max_wave:.1f} m", wave_color, wave_note))

        # 氣溫
        if avg_temp is not None:
            temp_color = "#B45309" if avg_temp < 5 else "#374151"
            temp_note  = "⚠ 低溫警示" if avg_temp < 5 else ""
            body_parts.append(_metric_html("平均氣溫", f"{avg_temp:.1f}°C", temp_color, temp_note))

        # 能見度
        if min_vis_m is not None:
            vis_str = f"{min_vis_m/1000:.1f} km" if min_vis_m >= 1000 else f"{int(min_vis_m)} m"
            vis_color = "#B91C1C" if min_vis_m < 1000 else "#B45309" if min_vis_m < 3000 else "#374151"
            vis_note  = "⚠ 濃霧" if min_vis_m < 1000 else ("偏低" if min_vis_m < 3000 else "")
            body_parts.append(_metric_html("最低能見度", vis_str, vis_color, vis_note))

        # 天氣概況
        if wx_codes:
            body_parts.append(_metric_html("天氣概況", _wx_zh(wx_codes), "#374151"))

        # 風險警示
        for risk in all_risks:
            body_parts.append(
                f"<div style='background:#FEF2F2;border-left:3px solid #B91C1C;"
                f"border-radius:4px;padding:5px 8px;margin-top:4px;"
                f"font-size:0.82em;color:#B91C1C'>🚨 {risk}</div>"
            )

        st.markdown(
            f"<div style='background:#fff;border:1px solid #E5E7EB;border-top:none;"
            f"border-radius:0 0 8px 8px;padding:12px'>{''.join(body_parts)}</div>",
            unsafe_allow_html=True,
        )


def render_berthing_advisory(
    result: AnalysisResult,
    vessel: VesselInfo,
    analyzer: Optional[Any] = None,
) -> None:
    """渲染靠離泊作業建議面板（吹開/攏風、高風險時段、最大風力、浪型、溫度、能見度）"""
    arr = getattr(result, "arr_window_result", None)
    dep = getattr(result, "dep_window_result", None)

    if arr is None and dep is None:
        return

    st.subheader("⚓ 靠離泊作業建議")

    col_arr, col_dep, col_port = st.columns(3)

    # ── 靠泊天氣 ────────────────────────────────────────────────
    _render_op_col(col_arr, arr, "🚢 靠泊天氣", vessel.arrival_time, "#1E40AF")

    # ── 離泊天氣 ────────────────────────────────────────────────
    _render_op_col(col_dep, dep, "⚓ 離泊天氣", vessel.departure_time, "#065F46")

    # ── 在港天氣 ────────────────────────────────────────────────
    with col_port:
        st.markdown(
            "<div style='background:#7C3AED;color:#fff;padding:8px 12px;"
            "border-radius:8px 8px 0 0;font-weight:bold;font-size:1em'>📊 在港天氣</div>",
            unsafe_allow_html=True,
        )

        inport_body = []

        if analyzer is not None and getattr(analyzer, "conditions", []):
            summary = analyzer.inport_condition_summary(vessel)
            if summary:
                avg_t  = summary.get("avg_temp")
                min_t  = summary.get("min_temp")
                min_v  = summary.get("min_vis_m")
                avg_v  = summary.get("avg_vis_m")
                wx_c   = summary.get("weather_codes", [])
                risks  = summary.get("condition_risks", [])

                if avg_t is not None:
                    temp_color = "#B45309" if (min_t is not None and min_t < 5) else "#374151"
                    temp_note  = f"最低 {min_t:.1f}°C{'  ⚠ 低溫' if min_t < 5 else ''}" if min_t is not None else ""
                    inport_body.append(_metric_html("平均氣溫", f"{avg_t:.1f}°C", temp_color, temp_note))

                if min_v is not None:
                    vis_str   = f"{min_v/1000:.1f} km" if min_v >= 1000 else f"{int(min_v)} m"
                    vis_color = "#B91C1C" if min_v < 1000 else "#B45309" if min_v < 3000 else "#374151"
                    vis_note  = ""
                    if avg_v is not None:
                        avg_v_str = f"{avg_v/1000:.1f} km" if avg_v >= 1000 else f"{int(avg_v)} m"
                        vis_note  = f"平均 {avg_v_str}"
                    inport_body.append(_metric_html("最低能見度", vis_str, vis_color, vis_note))

                if wx_c:
                    inport_body.append(_metric_html("天氣概況", _wx_zh(wx_c), "#374151"))

                inport_body.append(_metric_html("氣象資料筆數", f"{len(analyzer.conditions)} 筆", "#374151"))

                for risk in risks:
                    inport_body.append(
                        f"<div style='background:#FFFBEB;border-left:3px solid #B45309;"
                        f"border-radius:4px;padding:5px 8px;margin-top:4px;"
                        f"font-size:0.82em;color:#B45309'>⚠️ {risk}</div>"
                    )
            else:
                inport_body.append(
                    "<div style='color:#9CA3AF;font-size:0.9em;text-align:center;padding:8px'>無在港天氣摘要</div>"
                )
        else:
            inport_body.append(
                "<div style='color:#9CA3AF;font-size:0.9em;text-align:center;padding:8px'>無在港天氣資料</div>"
            )

        st.markdown(
            f"<div style='background:#fff;border:1px solid #E5E7EB;border-top:none;"
            f"border-radius:0 0 8px 8px;padding:12px'>{''.join(inport_body)}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")


# ================= 詳細報告 =================

def _gust_bft_label(gust_kts: float) -> str:
    """陣風 kts → 蒲福等級標籤"""
    if gust_kts >= 64: return "Bft 12 颶風"
    if gust_kts >= 56: return "Bft 11 暴風"
    if gust_kts >= 48: return "Bft 10 狂風"
    if gust_kts >= 41: return "Bft 9 烈風"
    if gust_kts >= 34: return "Bft 8 大風"
    if gust_kts >= 28: return "Bft 7 疾風"
    if gust_kts >= 22: return "Bft 6 強風"
    if gust_kts >= 17: return "Bft 5 清勁風"
    return "Bft ≤4 微風"


def _sf_status_label(sf: float) -> tuple[str, str]:
    """SF → (中文狀態, CSS 顏色)"""
    if sf >= 3.0:  return "優良", "#059669"
    if sf >= 2.0:  return "良好", "#16A34A"
    if sf >= 1.7:  return "合格", "#CA8A04"
    if sf >= 1.0:  return "偏低⚠️", "#DC2626"
    return "嚴重不足🚨", "#7F1D1D"


def render_detail_report(
    result: AnalysisResult,
    sidebar_data: Dict[str, Any],
    df_detail: Optional[pd.DataFrame] = None,
    _analyzer: Optional[Any] = None,   # reserved for future port-level data
) -> None:
    """渲染詳細報告（從船長 & 公司管理層雙視角）"""
    from datetime import datetime as _dt

    wfs       = result.wind_force_summary
    sf        = _get_wfs_value(wfs, "safety_factor",    "safety_factor")
    trans_N   = _get_wfs_value(wfs, "max_trans_force_N","max_trans_force_N")
    long_N    = _get_wfs_value(wfs, "max_long_force_N", "max_long_force_N")
    total_kN  = _get_wfs_value(wfs, "total_restraint_kN","total_restraint_kN")
    req_kN    = _get_wfs_value(wfs, "required_force_kN","required_force_kN")
    w_type    = (getattr(wfs, "wind_type", None)
                 or (wfs.get("wind_type","N/A") if isinstance(wfs,dict) else "N/A"))

    wfs_rec   = (wfs.max_gust_record if hasattr(wfs,"max_gust_record")
                 else wfs.get("max_gust_record",{}) if isinstance(wfs,dict) else {})
    rec_gust  = (wfs_rec.get("wind_gust") if isinstance(wfs_rec,dict)
                 else getattr(wfs_rec,"wind_gust", 0.0)) or 0.0
    rec_wave  = (wfs_rec.get("wave_height") if isinstance(wfs_rec,dict)
                 else getattr(wfs_rec,"wave_height", 0.0)) or 0.0
    rec_time  = (wfs_rec.get("time") if isinstance(wfs_rec,dict)
                 else getattr(wfs_rec,"time", None))

    risk_spec   = RISK_LEVEL_SPECS.get(result.risk_level, RISK_LEVEL_SPECS["low"])
    sf_label, sf_color = _sf_status_label(sf)
    bft_label   = _gust_bft_label(rec_gust)
    arrival     = sidebar_data.get("arrival")
    departure   = sidebar_data.get("departure")
    stay_hrs    = ((departure - arrival).total_seconds() / 3600) if (arrival and departure) else 0

    tug         = result.tug_recommendation
    final_tugs  = _get_tug_value(tug,"final_tug_count","final_tug_count", 0)
    tug_ok      = _get_tug_value(tug,"adequacy","adequacy", False)
    mooring_ok  = result.mooring_split.bow.status == "OK" and result.mooring_split.stern.status == "OK"

    # ── 決策建議判定 ──────────────────────────────────────────
    if result.risk_level == "low":
        decision, dec_color, dec_bg = "✅ GO — 可按計劃執行", "#065F46", "#ECFDF5"
    elif result.risk_level == "medium":
        decision, dec_color, dec_bg = "🟡 CONDITIONAL GO — 謹慎評估後執行", "#92400E", "#FFFBEB"
    elif result.risk_level == "high":
        decision, dec_color, dec_bg = "⚠️ CAUTION — 建議強化措施或考慮延後", "#B45309", "#FFF7ED"
    else:
        decision, dec_color, dec_bg = "🚨 NO-GO — 強烈建議停止或延後作業", "#7F1D1D", "#FEF2F2"

    # ══════════════════════════════════════════════════════════
    # A. 執行決策摘要
    # ══════════════════════════════════════════════════════════
    st.markdown(
        f"<div style='background:{dec_bg};border-left:6px solid {dec_color};"
        f"border-radius:8px;padding:16px 20px;margin-bottom:16px'>"
        f"<div style='font-size:1.3em;font-weight:700;color:{dec_color}'>{decision}</div>"
        f"<div style='color:#374151;margin-top:6px;font-size:0.9em'>"
        f"綜合風險評分 <b>{result.risk_score:.0f}/100</b> ({risk_spec.name_zh}) ｜ "
        f"安全係數 <b>SF {sf:.2f}</b> ({sf_label}) ｜ "
        f"最大陣風 <b>{rec_gust:.0f} kts</b> ({bft_label})"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════
    # B. 關鍵指標一覽表（管理層視角）
    # ══════════════════════════════════════════════════════════
    with st.expander("📋 關鍵指標一覽（管理層）", expanded=True):
        st.caption("以下指標供公司主管快速掌握作業風險與合規狀態。")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 風險評分",    f"{result.risk_score:.0f} / 100",
                    risk_spec.label,            delta_color="inverse")
        col2.metric("⚖️ 安全係數 SF", f"{sf:.2f}",
                    sf_label,                   delta_color="off")
        col3.metric("💨 最大陣風",    f"{rec_gust:.0f} kts",
                    bft_label,                  delta_color="off")
        col4.metric("⏱️ 在港時長",    f"{stay_hrs:.1f} h",
                    f"ETA {arrival.strftime('%m/%d %H:%M') if arrival else 'N/A'}",
                    delta_color="off")

        st.markdown("---")

        # 合規狀態表
        compliance_rows = [
            ("OCIMF MEG4 最低 SF ≥ 1.7",
             "✅ 符合" if sf >= 1.7 else "❌ 未達標", "#059669" if sf >= 1.7 else "#DC2626"),
            ("纜繩配置充足",
             "✅ 足夠" if mooring_ok else "⚠️ 需增強", "#059669" if mooring_ok else "#B45309"),
            ("拖船抗力充足",
             "✅ 充足" if tug_ok else "⚠️ 需增援", "#059669" if tug_ok else "#B45309"),
            ("風力等級（OCIMF 作業界限）",
             f"{'✅ Bft≤7 安全' if rec_gust < 34 else '⚠️ Bft 8+ 高度警戒' if rec_gust < 41 else '🚨 Bft 9+ 超界限'}",
             "#059669" if rec_gust < 34 else "#B45309" if rec_gust < 41 else "#DC2626"),
        ]

        for label, status, color in compliance_rows:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:6px 10px;border-bottom:1px solid #F3F4F6'>"
                f"<span style='color:#374151;font-size:0.9em'>{label}</span>"
                f"<b style='color:{color};font-size:0.9em'>{status}</b></div>",
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════
    # C. 氣象威脅評估（船長視角）
    # ══════════════════════════════════════════════════════════
    with st.expander("🌬️ 氣象威脅評估（船長）", expanded=True):
        st.caption("以下數據供船長評估在港期間的天氣風險及操作時機。")

        cw1, cw2, cw3 = st.columns(3)
        cw1.metric("最大陣風",   f"{rec_gust:.1f} kts",  bft_label)
        cw2.metric("最大波高",   f"{rec_wave:.2f} m",    "浪高警戒" if rec_wave >= 2.5 else "正常")
        cw3.metric("最大受力時刻", rec_time.strftime('%m/%d %H:%M') if rec_time else "N/A")

        # 在港高風險時段
        if df_detail is not None and "wind_gust_kts" in df_detail.columns:
            try:
                gust_col = pd.to_numeric(df_detail["wind_gust_kts"], errors="coerce")
                haz_df   = df_detail[gust_col >= 28.0].copy()   # gust_caution 以上
                if not haz_df.empty:
                    st.markdown("**⚠️ 在港高警戒時段（陣風 ≥28 kts）**")
                    # 建立顯示 DataFrame
                    show_cols = {}
                    if "time" in haz_df.columns:
                        show_cols["時間"] = pd.to_datetime(haz_df["time"]).dt.strftime("%m/%d %H:%M")
                    if "wind_speed_kts" in haz_df.columns:
                        show_cols["風速(kts)"] = pd.to_numeric(haz_df["wind_speed_kts"], errors="coerce").round(0)
                    if "wind_gust_kts" in haz_df.columns:
                        show_cols["陣風(kts)"] = pd.to_numeric(haz_df["wind_gust_kts"], errors="coerce").round(0)
                    if "wave_sig_m" in haz_df.columns:
                        show_cols["浪高(m)"] = pd.to_numeric(haz_df["wave_sig_m"], errors="coerce").round(2)
                    if "safety_factor" in haz_df.columns:
                        show_cols["SF"] = pd.to_numeric(haz_df["safety_factor"], errors="coerce").round(2)

                    disp = pd.DataFrame(show_cols).reset_index(drop=True)

                    def _style_row(row):
                        g = row.get("陣風(kts)", 0) or 0
                        if g >= 41:   bg = "#FECACA"
                        elif g >= 34: bg = "#FED7AA"
                        elif g >= 28: bg = "#FEF9C3"
                        else:         bg = ""
                        return [f"background-color:{bg}" if bg else "" for _ in row]

                    st.dataframe(
                        disp.style.apply(_style_row, axis=1),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.success("在港期間陣風均低於警戒值（28 kts），氣象狀況良好。")
            except Exception:
                pass

        # 風型說明
        wt_zh    = _wind_type_label(w_type)
        wt_color = _wind_type_color(w_type)
        wt_note  = {
            "offshore": "吹開風對靠泊安全有利（船舶被風推離碼頭），但對纜繩要求較高。",
            "onshore":  "攏風（船被風推向碼頭），靠泊易但離泊困難，需注意護舷器壓力。",
            "headwind": "逆風有助於操船減速，但橫向擾動較大。",
            "tailwind": "順風助速，靠泊時制動距離增加，注意速度控制。",
        }.get(w_type, "")
        if wt_note:
            st.markdown(
                f"<div style='background:#EFF6FF;border-left:4px solid {wt_color};"
                f"border-radius:4px;padding:8px 12px;margin-top:8px;font-size:0.88em'>"
                f"<b>主要風型：{wt_zh}</b> — {wt_note}</div>",
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════
    # D. 風險警示與操作建議
    # ══════════════════════════════════════════════════════════
    with st.expander("⚠️ 風險警示與操作建議", expanded=True):
        if result.recommendations:
            for rec in result.recommendations:
                _render_recommendation(rec)
        else:
            st.success("✅ 目前評估無特殊風險警示")

    # ══════════════════════════════════════════════════════════
    # E. 安全力學詳細計算
    # ══════════════════════════════════════════════════════════
    with st.expander("⚙️ 安全力學詳細計算（OCIMF 方法）"):
        st.caption("依據 OCIMF 風力計算方法（F = ½ρCdAV²）與 MEG4 繫泊標準。")

        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("橫向受風力",  f"{trans_N/1000:.0f} kN",
                      "垂直碼頭方向")
        col_f2.metric("縱向受風力",  f"{long_N/1000:.0f} kN",
                      "沿碼頭方向")
        col_f3.metric("纜繩+拖船抗力", f"{total_kN:.0f} kN",
                      f"需求 {req_kN:.0f} kN")

        margin = total_kN - req_kN
        margin_pct = (margin / req_kN * 100) if req_kN > 0 else 0
        m_color = "#059669" if margin > 0 else "#DC2626"
        st.markdown(
            f"<div style='background:#F9FAFB;border-radius:6px;padding:12px;margin-top:8px'>"
            f"<b>安全餘裕：</b> "
            f"<span style='color:{m_color};font-weight:700'>"
            f"{'+' if margin>=0 else ''}{margin:.0f} kN ({margin_pct:+.1f}%)</span><br>"
            f"<small style='color:#6B7280'>"
            f"安全係數 SF = 抗力÷需求 = {total_kN:.0f}÷{req_kN:.0f} = <b>{sf:.2f}</b>（OCIMF MEG4 要求 ≥1.7）"
            f"</small></div>",
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════
    # F. 纜繩與拖船配置
    # ══════════════════════════════════════════════════════════
    with st.expander("⚓ 纜繩 & 拖船配置詳情", expanded=True):
        mc1, mc2 = st.columns(2)

        with mc1:
            st.markdown("**🪢 纜繩配置**")
            mbl_kn = sidebar_data["mbs"]
            wll_kn = mbl_kn * 0.33   # WLL = MBL × 0.33 (SF=3.0)
            st.markdown(f"""
| 位置 | 頭纜/尾纜 | 倒纜 |
|------|-----------|------|
| 艏 | {sidebar_data['bh']} 條 | {sidebar_data['bs']} 條 |
| 艉 | {sidebar_data['sh']} 條 | {sidebar_data['ss']} 條 |
| **合計** | **{sidebar_data['bh']+sidebar_data['sh']} 條** | **{sidebar_data['bs']+sidebar_data['ss']} 條** |
""")
            st.caption(
                f"MBL: {mbl_kn:.0f} kN／條 ｜ WLL: {wll_kn:.0f} kN／條（MEG4 WLL=MBL×0.33）"
            )
            m = result.mooring_split
            if m.bow.status == "OK" and m.stern.status == "OK":
                st.success("✅ 纜繩配置足夠")
            else:
                st.warning(
                    f"⚠️ 建議：艏 [{m.bow.recommendation_text}] / 艉 [{m.stern.recommendation_text}]"
                )

        with mc2:
            st.markdown("**🚤 拖船支援**")
            tug_hp   = sidebar_data.get("tug_hp", 0)
            bp_ton   = (tug_hp / 100.0) * 1.1
            bp_kn    = bp_ton * 9.81
            st.markdown(f"""
| 項目 | 數值 |
|------|------|
| 拖船數量 | {final_tugs} 艘 |
| 單艘馬力 | {tug_hp:.0f} HP |
| 單艘推力 | {bp_ton:.1f} ton ({bp_kn:.0f} kN) |
| 合計推力 | {final_tugs * bp_kn:.0f} kN |
""")
            st.caption("推力係數 1.1 ton/100HP（港口作業拖船，含螺旋槳效率）")
            if tug_ok:
                st.success("✅ 拖船推力充足")
            else:
                st.warning("⚠️ 建議增加拖船支援")

    # ══════════════════════════════════════════════════════════
    # G. 應變觸發條件清單
    # ══════════════════════════════════════════════════════════
    with st.expander("🚨 應變觸發條件清單（船長 & 值班官）"):
        st.caption("以下觸發條件應預先告知全體值班人員，並確認應變程序。")
        triggers = [
            ("🟡 一級警戒",  "陣風 28–33 kts", "加強纜繩巡視（每 30 分鐘）；確認拖船隨時備妥"),
            ("🟠 二級警戒",  "陣風 34–40 kts（Bft 8）", "啟動額外纜繩；通知港務；拖船移至船旁待命"),
            ("🔴 三級警戒",  "陣風 41–47 kts（Bft 9）", "考慮提前離泊；橋樑團隊備妥；聯繫公司主管"),
            ("⛔ 緊急離泊",  "陣風 ≥48 kts（Bft 10+）或 SF < 1.0", "立即啟動緊急離泊程序；VHF 頻道 16 通報"),
            ("🌊 浪高警戒",  "浪高 ≥2.5 m", "評估護舷器狀況；加強艏艉纜繩；通知貨物操作暫停"),
            ("👁️ 能見度", "能見度 < 1,000 m", "霧笛備妥；VHF 持續監聽；延遲靠/離泊作業"),
        ]
        for level, trigger, action in triggers:
            st.markdown(
                f"<div style='display:grid;grid-template-columns:130px 220px 1fr;"
                f"gap:8px;align-items:start;padding:6px 4px;"
                f"border-bottom:1px solid #F3F4F6;font-size:0.87em'>"
                f"<span style='font-weight:700'>{level}</span>"
                f"<span style='color:#374151'>{trigger}</span>"
                f"<span style='color:#6B7280'>{action}</span></div>",
                unsafe_allow_html=True,
            )


# ================= 圖表分析 =================

def render_chart_analysis(
    analyzer:     WeatherAnalyzer,
    vessel:       VesselInfo,
    result:       AnalysisResult,
    df_detail:    Optional[pd.DataFrame],
    sidebar_data: Dict[str, Any],
) -> None:
    """渲染圖表分析（Matplotlib 靜態圖 + Plotly 互動式時間軸）"""
    ps = PlotService(analyzer)

    tab_c1, tab_c2 = st.tabs(["綜合趨勢", "時間軸分析"])

    with tab_c1:
        st.subheader("📊 風速與浪高趨勢")
        st.pyplot(ps.plot_wind_trend(vessel, result))
        st.pyplot(ps.plot_wave_trend(vessel, result))
        st.pyplot(ps.plot_force(vessel, result))

    with tab_c2:
        st.subheader("📈 增強型時間軸分析")
        if df_detail is None or len(df_detail) == 0:
            st.info("無可用的詳細氣象資料")
            return

        try:
            mbl_kN      = vessel.mbl / 1000.0
            total_lines = vessel.total_mooring_lines
            cap_kN      = mbl_kN * vessel.safety_factor * total_lines

            max_force_N = _get_wfs_value(
                result.wind_force_summary,
                "max_gust_force_N", "max_gust_force_N",
            )

            fig_enhanced = plot_enhanced_timeline(
                df_detail,
                sidebar_data["arrival"],
                sidebar_data["departure"],
                sidebar_data["berth_dir"],
                vessel_info={
                    "total_force_N":       max_force_N,
                    "mooring_capacity_kN": cap_kN,
                },
            )
            if fig_enhanced:
                st.plotly_chart(fig_enhanced, use_container_width=True)

        except Exception:
            logger.exception("render_chart_analysis：Plotly 繪圖失敗")
            st.error("❌ 互動式圖表繪製失敗，請查看日誌")


# ================= 靠泊風險分析書 =================

def render_risk_analysis_report(
    result:       AnalysisResult,
    sidebar_data: Dict[str, Any],
    df_detail:    Optional[pd.DataFrame],
    analyzer:     Any,
    sf:           float,
    mooring_cap:  Any = None,
    tug_cap:      Any = None,
    max_force_N:  float = 0.0,
) -> None:
    """
    靠泊風險分析書 — 完整本地報告（不耗用 API）。
    同時滿足：
      ‧ 公司管理層  → 決策結論、合規狀態、責任依據
      ‧ 船長/值班員 → 逐時天氣、受力計算、應變條件
    """
    from datetime import datetime as _dt

    # ── 基本資料抽取 ───────────────────────────────────────────
    wfs      = result.wind_force_summary
    trans_N  = _get_wfs_value(wfs, "max_trans_force_N", "max_trans_force_N")
    long_N   = _get_wfs_value(wfs, "max_long_force_N",  "max_long_force_N")
    total_kN = _get_wfs_value(wfs, "total_restraint_kN","total_restraint_kN")
    req_kN   = _get_wfs_value(wfs, "required_force_kN", "required_force_kN")
    w_type   = (getattr(wfs,"wind_type",None)
                or (wfs.get("wind_type","N/A") if isinstance(wfs,dict) else "N/A"))

    wfs_rec  = (wfs.max_gust_record if hasattr(wfs,"max_gust_record")
                else wfs.get("max_gust_record",{}) if isinstance(wfs,dict) else {})
    rec_gust = (wfs_rec.get("wind_gust") if isinstance(wfs_rec,dict)
                else getattr(wfs_rec,"wind_gust",0.0)) or 0.0
    rec_wave = (wfs_rec.get("wave_height") if isinstance(wfs_rec,dict)
                else getattr(wfs_rec,"wave_height",0.0)) or 0.0
    rec_time = (wfs_rec.get("time") if isinstance(wfs_rec,dict)
                else getattr(wfs_rec,"time",None))

    arrival    = sidebar_data.get("arrival")
    departure  = sidebar_data.get("departure")
    stay_hrs   = ((departure - arrival).total_seconds()/3600) if (arrival and departure) else 0
    port_name  = getattr(analyzer,"port_name","—") if analyzer else "—"
    risk_spec  = RISK_LEVEL_SPECS.get(result.risk_level, RISK_LEVEL_SPECS["low"])
    bft_label  = _gust_bft_label(rec_gust)
    sf_label, sf_color = _sf_status_label(sf)

    tug = result.tug_recommendation
    final_tugs = _get_tug_value(tug,"final_tug_count","final_tug_count",0)
    tug_ok     = _get_tug_value(tug,"adequacy","adequacy",False)
    mooring_ok = (result.mooring_split.bow.status == "OK"
                  and result.mooring_split.stern.status == "OK")

    moor_cap_kN = (mooring_cap.total_capacity_kN if mooring_cap else total_kN)
    tug_cap_kN  = (tug_cap.total_push_kN if tug_cap else 0.0)

    gen_time = _dt.now().strftime("%Y-%m-%d %H:%M")

    # ── 決策判定 ─────────────────────────────────────────────
    if result.risk_level == "low":
        decision_en, decision_zh = "GO", "可按計劃執行"
        dec_color, dec_bg = "#065F46", "#ECFDF5"
    elif result.risk_level == "medium":
        decision_en, decision_zh = "CONDITIONAL GO", "採取加強措施後可執行"
        dec_color, dec_bg = "#92400E", "#FFFBEB"
    elif result.risk_level == "high":
        decision_en, decision_zh = "CAUTION", "建議強化措施 / 評估延後"
        dec_color, dec_bg = "#B45309", "#FFF7ED"
    else:
        decision_en, decision_zh = "NO-GO", "強烈建議停止或顯著延後作業"
        dec_color, dec_bg = "#7F1D1D", "#FEF2F2"

    # ══════════════════════════════════════════════════════════
    # 報告標題
    # ══════════════════════════════════════════════════════════
    st.markdown(
        f"<div style='background:linear-gradient(135deg,#1E3A5F,#2563EB);color:#fff;"
        f"border-radius:10px;padding:20px 24px;margin-bottom:20px'>"
        f"<div style='font-size:0.8em;letter-spacing:2px;opacity:0.8'>BERTHING RISK ANALYSIS REPORT</div>"
        f"<div style='font-size:1.6em;font-weight:800;margin:4px 0'>⚓ 靠泊風險分析書</div>"
        f"<div style='font-size:0.88em;opacity:0.85'>"
        f"港口：<b>{port_name}</b> ｜ "
        f"ETA：{arrival.strftime('%Y-%m-%d %H:%M') if arrival else 'N/A'} ｜ "
        f"ETD：{departure.strftime('%Y-%m-%d %H:%M') if departure else 'N/A'} ｜ "
        f"報告產生時間：{gen_time}"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════
    # A. 決策結論（管理層首要關注）
    # ══════════════════════════════════════════════════════════
    st.markdown("### A｜決策結論")
    st.markdown(
        f"<div style='background:{dec_bg};border:2px solid {dec_color};"
        f"border-radius:10px;padding:20px 24px;text-align:center'>"
        f"<div style='font-size:2em;font-weight:900;color:{dec_color}'>"
        f"{decision_en}</div>"
        f"<div style='font-size:1.1em;color:{dec_color};margin-top:4px'>{decision_zh}</div>"
        f"<div style='margin-top:12px;display:flex;justify-content:center;gap:32px;"
        f"font-size:0.9em;color:#374151'>"
        f"<span>風險評分 <b>{result.risk_score:.0f}/100</b></span>"
        f"<span>等級 <b>{risk_spec.label} {risk_spec.name_zh}</b></span>"
        f"<span>SF <b>{sf:.2f}</b> ({sf_label})</span>"
        f"<span>最大陣風 <b>{rec_gust:.0f} kts</b> ({bft_label})</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ══════════════════════════════════════════════════════════
    # B. 關鍵指標儀表板（管理層視角）
    # ══════════════════════════════════════════════════════════
    st.markdown("### B｜關鍵指標儀表板")
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("綜合風險", f"{result.risk_score:.0f}/100",
              risk_spec.label, delta_color="inverse")
    b2.metric("安全係數 SF", f"{sf:.2f}", sf_label, delta_color="off")
    b3.metric("最大陣風", f"{rec_gust:.0f} kts", bft_label, delta_color="off")
    b4.metric("最大波高", f"{rec_wave:.2f} m",
              "正常" if rec_wave < 2.5 else "⚠️警戒", delta_color="off")
    b5.metric("在港時長", f"{stay_hrs:.1f} h",
              f"{int(stay_hrs)}h {int((stay_hrs%1)*60)}m", delta_color="off")

    # 合規快覽
    comp_items = [
        ("OCIMF MEG4 安全係數 ≥1.7", sf >= 1.7),
        ("纜繩配置充足", mooring_ok),
        ("拖船推力充足", tug_ok),
        ("陣風未超 Bft 8 作業上限（34 kts）", rec_gust < 34),
    ]
    cols_c = st.columns(len(comp_items))
    for col, (label, ok) in zip(cols_c, comp_items):
        col.markdown(
            f"<div style='background:{'#ECFDF5' if ok else '#FEF2F2'};"
            f"border-radius:6px;padding:8px 10px;text-align:center;"
            f"border:1px solid {'#BBF7D0' if ok else '#FECACA'}'>"
            f"<div style='font-size:1.3em'>{'✅' if ok else '❌'}</div>"
            f"<div style='font-size:0.75em;color:#374151;margin-top:2px'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # C. 作業資訊摘要
    # ══════════════════════════════════════════════════════════
    st.markdown("### C｜作業資訊摘要")
    ci1, ci2 = st.columns(2)
    with ci1:
        st.markdown("**🏢 公司 / 港口**")
        vt = sidebar_data.get("vessel_type_display","手動輸入")
        st.markdown(f"""
| 項目 | 內容 |
|------|------|
| 港口 | {port_name} |
| 船型 | {vt} |
| 靠泊舷側 | {sidebar_data.get('side','N/A')} |
| 泊位方向 | {sidebar_data.get('berth_dir',0):.0f}° |
| 吃水（艏/艉） | {sidebar_data.get('draft_b',0):.1f} m / {sidebar_data.get('draft_s',0):.1f} m |
| 受風面積 | {sidebar_data.get('area',0):,.0f} m² |
""")
    with ci2:
        st.markdown("**⏱️ 作業時間**")
        st.markdown(f"""
| 項目 | 時間 |
|------|------|
| ETA（預定靠港） | {arrival.strftime('%Y-%m-%d %H:%M') if arrival else 'N/A'} |
| ETD（預定離港） | {departure.strftime('%Y-%m-%d %H:%M') if departure else 'N/A'} |
| 在港時長 | {stay_hrs:.1f} 小時 |
| 報告生成 | {gen_time} |
""")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # D. 氣象威脅分析（船長視角）
    # ══════════════════════════════════════════════════════════
    st.markdown("### D｜氣象威脅分析")
    di1, di2 = st.columns([1, 1])

    with di1:
        st.markdown("**🌬️ 最惡劣時刻**")
        wt_zh = _wind_type_label(w_type)
        st.markdown(f"""
| 指標 | 數值 | 評估 |
|------|------|------|
| 最大陣風 | **{rec_gust:.1f} kts** | {bft_label} |
| 最大波高 | **{rec_wave:.2f} m** | {'⚠️中度警戒' if rec_wave >= 2.5 else '正常'} |
| 最大受力時刻 | {rec_time.strftime('%m/%d %H:%M') if rec_time else 'N/A'} | — |
| 主要風型 | {wt_zh} | {'對靠泊有利' if w_type=='offshore' else '⚠️注意攏風' if w_type=='onshore' else '—'} |
""")

    with di2:
        st.markdown("**📊 在港高風險時段**")
        if df_detail is not None and "wind_gust_kts" in df_detail.columns:
            try:
                gust_s  = pd.to_numeric(df_detail["wind_gust_kts"], errors="coerce")
                bft8_h  = int((gust_s >= 34).sum())
                bft9_h  = int((gust_s >= 41).sum())
                bft10_h = int((gust_s >= 48).sum())
                caution_h = int((gust_s >= 28).sum())
                st.markdown(f"""
| 等級 | 小時數 | 影響 |
|------|--------|------|
| 陣風 ≥28 kts（Bft 7+，警戒） | **{caution_h}h** | {'⚠️' if caution_h > 0 else '✅'} |
| 陣風 ≥34 kts（Bft 8，高度警戒） | **{bft8_h}h** | {'🔴' if bft8_h > 0 else '✅'} |
| 陣風 ≥41 kts（Bft 9，暴風） | **{bft9_h}h** | {'🚨' if bft9_h > 0 else '✅'} |
| 陣風 ≥48 kts（Bft 10，極危） | **{bft10_h}h** | {'⛔' if bft10_h > 0 else '✅'} |
""")
            except Exception:
                st.caption("無法統計高風險時段。")
        else:
            st.caption("無逐時數據。")

    # 高風險時段時間表
    if df_detail is not None and "wind_gust_kts" in df_detail.columns:
        try:
            gust_col = pd.to_numeric(df_detail["wind_gust_kts"], errors="coerce")
            haz_mask = gust_col >= 28.0
            if haz_mask.any():
                with st.expander(f"🕐 逐時高風險時段明細（陣風≥28 kts，共 {haz_mask.sum()} 筆）"):
                    haz_df = df_detail[haz_mask].copy()
                    show = {}
                    if "time" in haz_df.columns:
                        show["時間"] = pd.to_datetime(haz_df["time"]).dt.strftime("%m/%d %H:%M")
                    for src, name, div in [
                        ("wind_speed_kts","風速(kts)",1),
                        ("wind_gust_kts","陣風(kts)",1),
                        ("wave_sig_m","浪高(m)",1),
                        ("gust_force_N","陣風力(kN)",1000),
                        ("safety_factor","SF",1),
                    ]:
                        if src in haz_df.columns:
                            col_data = pd.to_numeric(haz_df[src], errors="coerce")
                            show[name] = (col_data/div).round(2 if div==1 else 0)
                    disp = pd.DataFrame(show).reset_index(drop=True)

                    def _row_color(row):
                        g = row.get("陣風(kts)", 0) or 0
                        bg = ("#FECACA" if g >= 41 else "#FED7AA" if g >= 34 else "#FEF9C3")
                        return [f"background-color:{bg}"] * len(row)

                    st.dataframe(
                        disp.style.apply(_row_color, axis=1),
                        use_container_width=True, hide_index=True,
                    )
        except Exception:
            pass

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # E. 安全力學評估
    # ══════════════════════════════════════════════════════════
    st.markdown("### E｜安全力學評估（OCIMF 方法）")
    ea1, ea2 = st.columns(2)
    with ea1:
        st.markdown("**💨 受力分解**")
        total_force_kN = max_force_N / 1000.0
        st.markdown(f"""
| 項目 | 數值 |
|------|------|
| 最大總受風力 | **{total_force_kN:.0f} kN** |
| 橫向受力（垂直碼頭） | **{trans_N/1000:.0f} kN** |
| 縱向受力（沿碼頭） | **{long_N/1000:.0f} kN** |
| 纜繩抗力 | {moor_cap_kN:.0f} kN |
| 拖船推力 | {tug_cap_kN:.0f} kN |
| **合計抗力** | **{total_kN:.0f} kN** |
""")
    with ea2:
        st.markdown("**⚖️ 安全係數評估**")
        margin = total_kN - req_kN
        margin_pct = (margin / req_kN * 100) if req_kN > 0 else 0
        sf_rows = [
            ("安全係數 SF", f"{sf:.2f}", sf_color),
            ("安全餘裕", f"{'+' if margin>=0 else ''}{margin:.0f} kN ({margin_pct:+.0f}%)",
             "#059669" if margin >= 0 else "#DC2626"),
            ("OCIMF MEG4 最低要求", "SF ≥ 1.7", "#374151"),
            ("本次評估結果",
             "✅ 符合" if sf >= 1.7 else "❌ 未達標",
             "#059669" if sf >= 1.7 else "#DC2626"),
        ]
        for label, val, color in sf_rows:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:5px 8px;border-bottom:1px solid #F3F4F6'>"
                f"<span style='font-size:0.88em;color:#374151'>{label}</span>"
                f"<b style='color:{color};font-size:0.88em'>{val}</b></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # F. 繫泊與拖船配置要求
    # ══════════════════════════════════════════════════════════
    st.markdown("### F｜繫泊 & 拖船配置要求")
    fa1, fa2 = st.columns(2)
    with fa1:
        st.markdown("**🪢 纜繩配置**")
        mbl_kn = sidebar_data.get("mbs", 0)
        wll_kn = mbl_kn * 0.33
        bh = sidebar_data.get("bh", 0)
        bs = sidebar_data.get("bs", 0)
        sh = sidebar_data.get("sh", 0)
        ss = sidebar_data.get("ss", 0)
        st.markdown(f"""
| 位置 | 頭纜/尾纜 | 倒纜 | 小計 |
|------|-----------|------|------|
| 艏 | {bh} 條 | {bs} 條 | {bh+bs} 條 |
| 艉 | {sh} 條 | {ss} 條 | {sh+ss} 條 |
| **合計** | **{bh+sh} 條** | **{bs+ss} 條** | **{bh+bs+sh+ss} 條** |

- MBL：{mbl_kn:.0f} kN／條
- WLL（MEG4 WLL=MBL×0.33）：{wll_kn:.0f} kN／條
- 狀態：{"✅ 配置足夠" if mooring_ok else "⚠️ 建議增加纜繩"}
""")
    with fa2:
        st.markdown("**🚤 拖船要求**")
        tug_hp = sidebar_data.get("tug_hp", 0)
        bp_ton = (tug_hp / 100.0) * 1.1
        bp_kn  = bp_ton * 9.81
        st.markdown(f"""
| 項目 | 數值 |
|------|------|
| 建議拖船數量 | **{final_tugs} 艘** |
| 單艘馬力 | {tug_hp:.0f} HP |
| 單艘推力 | {bp_ton:.1f} ton（{bp_kn:.0f} kN） |
| 合計推力 | {final_tugs*bp_kn:.0f} kN |
| 推力充足性 | {"✅ 充足" if tug_ok else "⚠️ 需增援"} |

> 推力係數 1.1 ton/100HP（一般港口作業拖船）
""")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # G. 靠泊 / 離泊時窗建議
    # ══════════════════════════════════════════════════════════
    st.markdown("### G｜靠泊 / 離泊時窗建議")
    arr_win = getattr(result, "arr_window_result", None)
    dep_win = getattr(result, "dep_window_result", None)
    ga1, ga2 = st.columns(2)

    def _window_card(col, win, label, op_time, color):
        with col:
            if win is None or not getattr(win,"has_data",False):
                st.info(f"{label}：無氣象時窗資料")
                return
            g  = getattr(win,"max_wind_gust",0.0) or 0.0
            wv = getattr(win,"max_wave_height",0.0) or 0.0
            hr = getattr(win,"high_risk_hours",0) or 0
            risks = (list(getattr(win,"risks",[])) +
                     list(getattr(win,"condition_risks",[])))
            ok = g < 34 and wv < 2.5
            status = "✅ 條件良好" if ok else "⚠️ 需注意"
            st.markdown(
                f"<div style='background:{'#ECFDF5' if ok else '#FFF7ED'};"
                f"border-left:4px solid {color};border-radius:6px;padding:12px'>"
                f"<b style='color:{color}'>{label}｜{op_time.strftime('%m/%d %H:%M')}</b>"
                f"<div style='margin-top:6px;font-size:0.88em;color:#374151'>"
                f"陣風：{g:.1f} kts ｜ 浪高：{wv:.2f} m ｜ 高風險時段：{hr}h</div>"
                f"<div style='margin-top:4px;font-weight:600;color:{'#059669' if ok else '#B45309'}'>"
                f"{status}</div></div>",
                unsafe_allow_html=True,
            )
            if risks:
                for r in risks:
                    st.caption(f"⚠️ {r}")

    _window_card(ga1, arr_win, "🚢 靠泊時窗", arrival, "#1E40AF")
    _window_card(ga2, dep_win, "⚓ 離泊時窗", departure, "#065F46")

    # 最佳時窗建議
    if df_detail is not None and "wind_gust_kts" in df_detail.columns and "time" in df_detail.columns:
        try:
            tmp = df_detail.copy()
            tmp["_gust"] = pd.to_numeric(tmp["wind_gust_kts"], errors="coerce")
            tmp["_time"] = pd.to_datetime(tmp["time"])
            best = tmp.nsmallest(3, "_gust")[["_time","_gust"]]
            if not best.empty:
                best_strs = "、".join(
                    f"{row['_time'].strftime('%m/%d %H:%M')}（{row['_gust']:.0f} kts）"
                    for _, row in best.iterrows()
                )
                st.info(f"💡 **最佳作業時窗**（陣風最小前 3 筆）：{best_strs}")
        except Exception:
            pass

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # H. 值班監控計畫（船長下達給值班員）
    # ══════════════════════════════════════════════════════════
    st.markdown("### H｜值班監控計畫")
    watch_rows = [
        ("每小時", "巡視全部纜繩張力；記錄缆繩計數器讀數；確認護舷器位置"),
        ("陣風≥28 kts", "每 30 分鐘巡視；通知輪機長；拖船確認備妥"),
        ("陣風≥34 kts", "每 15 分鐘巡視；啟動備用纜繩；通報港務當局"),
        ("陣風≥41 kts", "持續值守甲板；考慮請求拖船移至船旁；評估提前離泊"),
        ("浪高≥2.5 m", "檢查船體縱橫搖動；加強艏艉纜繩；暫停貨物作業"),
        ("能見度<1 km", "霧笛值守；VHF CH16 持續監聽；推遲靠離泊"),
    ]
    for condition, action in watch_rows:
        st.markdown(
            f"<div style='display:grid;grid-template-columns:160px 1fr;"
            f"gap:12px;padding:7px 8px;border-bottom:1px solid #F3F4F6;"
            f"align-items:start;font-size:0.87em'>"
            f"<span style='font-weight:700;color:#1E40AF'>{condition}</span>"
            f"<span style='color:#374151'>{action}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # I. 應變觸發條件
    # ══════════════════════════════════════════════════════════
    st.markdown("### I｜應變觸發條件 & 行動矩陣")
    trigger_rows = [
        ("🟡 一級警戒", "陣風 28–33 kts", "通知大副；每 30 分鐘巡視；確認拖船 ETA"),
        ("🟠 二級警戒", "陣風 34–40 kts（Bft 8）", "加固纜繩；拖船就位；通報港務；評估貨物作業是否暫停"),
        ("🔴 三級警戒", "陣風 41–47 kts（Bft 9）", "啟動緊急預案；考慮提前離泊；聯繫公司主管"),
        ("⛔ 緊急離泊", "陣風≥48 kts 或 SF<1.0", "立即啟動緊急離泊；VHF CH16 通報；填寫緊急事件報告"),
        ("🌊 浪湧警戒", "浪高≥2.5 m 或 Swell≥3 m", "評估船體搖擺；加強繫泊；暫停所有貨物吊具操作"),
        ("🔥 纜繩斷裂", "任何纜繩斷裂", "立即緊急備車；VHF CH16；拖船協助；視情況離泊"),
    ]
    for level, trigger, action in trigger_rows:
        st.markdown(
            f"<div style='display:grid;grid-template-columns:130px 220px 1fr;"
            f"gap:8px;padding:7px 4px;border-bottom:1px solid #F3F4F6;"
            f"align-items:start;font-size:0.86em'>"
            f"<b>{level}</b>"
            f"<span style='color:#374151'>{trigger}</span>"
            f"<span style='color:#6B7280'>{action}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════
    # J. 合規確認清單（管理層 / 船長雙簽）
    # ══════════════════════════════════════════════════════════
    st.markdown("### J｜合規確認清單")
    compliance_items = [
        ("OCIMF MEG4 安全係數 SF ≥ 1.7", sf >= 1.7,
         f"實際 SF = {sf:.2f}" if sf >= 1.7 else f"⚠️ 實際 SF = {sf:.2f}，需增強繫泊"),
        ("纜繩配置符合 OCIMF 最低要求", mooring_ok,
         "艏艉纜繩配置已驗證" if mooring_ok else "⚠️ 建議增加頭纜或倒纜"),
        ("拖船推力滿足操船需求", tug_ok,
         f"共 {final_tugs} 艘拖船" if tug_ok else "⚠️ 建議增加拖船"),
        ("陣風未超 OCIMF 作業上限（Bft 8 / 34 kts）", rec_gust < 34,
         f"最大陣風 {rec_gust:.0f} kts" if rec_gust < 34
         else f"⚠️ 最大陣風 {rec_gust:.0f} kts，已超 Bft 8 界限"),
        ("浪高未超 OCIMF 警戒值（2.5 m）", rec_wave < 2.5,
         f"最大浪高 {rec_wave:.2f} m" if rec_wave < 2.5
         else f"⚠️ 最大浪高 {rec_wave:.2f} m，需評估護舷器"),
        ("應變計畫已告知全體值班人員", True, "請於作業前確認 Section H/I"),
    ]
    for label, ok, note in compliance_items:
        icon = "✅" if ok else "❌"
        bg   = "#F0FDF4" if ok else "#FFF1F2"
        bc   = "#BBF7D0" if ok else "#FECACA"
        st.markdown(
            f"<div style='background:{bg};border:1px solid {bc};"
            f"border-radius:6px;padding:8px 14px;margin-bottom:6px;"
            f"display:flex;justify-content:space-between;align-items:center'>"
            f"<span style='font-size:0.88em;color:#374151'>{icon} {label}</span>"
            f"<span style='font-size:0.8em;color:#6B7280'>{note}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════
    # 免責聲明
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:6px;"
        f"padding:12px 16px;font-size:0.8em;color:#64748B'>"
        f"<b>⚠️ 免責聲明</b>：本分析書由 IWBDSS Pro 系統依據 OCIMF 方法自動生成，"
        f"僅供決策輔助參考，不構成操作指令。最終靠泊決策權在船長（Master），"
        f"並應遵守港口國主管機關之規定。報告生成時間：{gen_time}。"
        f"</div>",
        unsafe_allow_html=True,
    )


# ================= AI 分析 =================

def render_ai_analysis(
    enable_ai:    bool,
    ai_mode:      Optional[str],
    result:       AnalysisResult,
    sidebar_data: Dict[str, Any],
    df_detail:    pd.DataFrame,
    analyzer:     WeatherAnalyzer,
) -> None:
    """渲染 AI 決策輔助分析"""
    if not enable_ai:
        st.info("AI 分析已停用")
        return

    st.subheader("🤖 AI 決策輔助分析")

    # ── 共用計算 ──────────────────────────────────────────────
    mooring_cap = calculate_mooring_capacity(
        num_bow_lines          = sidebar_data["bh"],
        num_bow_spring_lines   = sidebar_data["bs"],
        num_stern_lines        = sidebar_data["sh"],
        num_stern_spring_lines = sidebar_data["ss"],
        mbl_per_line           = sidebar_data["mbs"],
        safety_factor          = sidebar_data.get("safety_factor", 0.5),
    )
    tug_final_count = _get_tug_value(
        result.tug_recommendation, "final_tug_count", "final_tug_count", 0
    )
    tug_cap = calculate_tug_capacity(tug_final_count, sidebar_data["tug_hp"])

    max_force_N = _get_wfs_value(
        result.wind_force_summary, "max_gust_force_N", "max_gust_force_N"
    )
    sf = _get_wfs_value(
        result.wind_force_summary, "safety_factor", "safety_factor"
    )

    # ── 快速摘要（靠泊風險分析書，不呼叫 API）─────────────────
    if ai_mode == "快速摘要":
        render_risk_analysis_report(
            result, sidebar_data, df_detail, analyzer, sf,
            mooring_cap, tug_cap, max_force_N,
        )
        return

    # ── 完整分析（呼叫 Perplexity API）──────────────────────
    with st.spinner("AI 正在進行深度分析（Perplexity）..."):

        total_records  = len(df_detail)
        offshore_count = (
            int(pd.to_numeric(df_detail["is_offshore"], errors="coerce").fillna(0).sum())
            if "is_offshore" in df_detail.columns
            else 0
        )
        # ── 建立 VesselParams dataclass ──────────────────────
        vessel_params = VesselParams(
            area               = float(sidebar_data["area"]),
            cd                 = float(sidebar_data["cd"]),
            draft_bow          = float(sidebar_data["draft_b"]),
            draft_stern        = float(sidebar_data["draft_s"]),
            bow_lines          = int(sidebar_data["bh"]),
            bow_spring_lines   = int(sidebar_data["bs"]),
            stern_lines        = int(sidebar_data["sh"]),
            stern_spring_lines = int(sidebar_data["ss"]),
            line_mbl           = float(sidebar_data["mbs"]),
            num_tugs           = int(tug_final_count),
            tug_hp             = float(sidebar_data["tug_hp"]),
        )

        # ── dominant_wind_dir ─────────────────────────────────
        dominant_wind_dir = "N/A"
        if "wind_gust_kts" in df_detail.columns and "wind_dir_deg" in df_detail.columns:
            idx = df_detail["wind_gust_kts"].idxmax()
            dominant_wind_dir = str(df_detail.loc[idx, "wind_dir_deg"])

        # ── 建立 AnalysisResults dataclass ───────────────────
        analysis_results = AnalysisResults(
            risk_level                = result.risk_level,
            risk_score                = float(result.risk_score),
            max_wind_force_kN         = max_force_N / 1000.0,
            max_gust_kts              = (
                float(pd.to_numeric(df_detail["wind_gust_kts"], errors="coerce").max())
                if "wind_gust_kts" in df_detail.columns
                else 0.0
            ),
            dominant_wind_dir         = dominant_wind_dir,
            offshore_wind_ratio       = (
                offshore_count / total_records * 100
                if total_records > 0
                else 0.0
            ),
            mooring_capacity_total_kN = mooring_cap.total_capacity_kN,
            tug_capacity_total_kN     = tug_cap.total_push_kN,
        )

        # ── 建立 HourlyRiskEntry 列表 ─────────────────────────
        hourly_data = _build_hourly_risk_entries(df_detail)

        # ── 呼叫 AI ───────────────────────────────────────────
        ai_content = st.session_state.ai_analyzer.generate_analysis(
            port_name        = analyzer.port_name,
            vessel_params    = vessel_params,
            analysis_results = analysis_results,
            berthing_time    = sidebar_data["arrival"],
            departure_time   = sidebar_data["departure"],
            hourly_data      = hourly_data if hourly_data else None,
        )
        st.markdown(ai_content)


# ================= 數據列表 =================

def render_data_list(
    df_detail:    Optional[pd.DataFrame],
    _analyzer:    Any = None,       # reserved
    _sidebar_data: Dict[str, Any] = None,  # reserved
) -> None:
    """渲染氣象數據列表（含風險等級背景色）"""
    if df_detail is None:
        return

    st.subheader("📋 氣象數據列表")

    legend_cols = st.columns(len(RISK_LEVEL_SPECS))
    for col, (key, spec) in zip(legend_cols, RISK_LEVEL_SPECS.items()):
        col.caption(f"● {spec.name_zh}")

    _COL_MAP = {
        "時間":       ("time",          "time"),
        "風速(kts)":  ("wind_speed_kts", "kts"),
        "陣風(kts)":  ("wind_gust_kts",  "kts"),
        "風向(°)":    ("wind_dir_deg",   "deg"),
        "浪高(m)":    ("wave_sig_m",     "m"),
        "最大浪(m)":  ("wave_max_m",     "m"),
        "平均力(kN)": ("avg_force_N",    "force"),
        "陣風力(kN)": ("gust_force_N",   "force"),
        "風險":       ("risk_level",     "risk"),
        "安全係數":   ("safety_factor",  "factor"),
    }

    _RISK_ZH = {key: spec.name_zh for key, spec in RISK_LEVEL_SPECS.items()}

    display_df = df_detail.copy()
    show_df    = pd.DataFrame()

    for display_name, (src_col, fmt) in _COL_MAP.items():
        if src_col not in display_df.columns:
            continue
        try:
            raw = display_df[src_col]
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[:, 0]

            if fmt == "time":
                show_df[display_name] = pd.to_datetime(raw).dt.strftime("%Y-%m-%d %H:%M")
            elif fmt == "force":
                show_df[display_name] = (pd.to_numeric(raw, errors="coerce") / 1000).round(1)
            elif fmt in ("kts", "deg", "m"):
                show_df[display_name] = pd.to_numeric(raw, errors="coerce").round(1)
            elif fmt == "factor":
                show_df[display_name] = pd.to_numeric(raw, errors="coerce").round(2)
            elif fmt == "risk":
                # 使用作業閾值重新分類，讓超標數據正確顯示高風險
                gust_col = display_df.get("wind_gust_kts", pd.Series(0.0, index=display_df.index))
                wind_col = display_df.get("wind_speed_kts", pd.Series(0.0, index=display_df.index))
                wave_col = display_df.get("wave_sig_m",    pd.Series(0.0, index=display_df.index))
                op_risk  = [
                    _classify_op_risk(
                        float(g) if pd.notna(g) else 0.0,
                        float(w) if pd.notna(w) else 0.0,
                        float(v) if pd.notna(v) else 0.0,
                    )
                    for g, w, v in zip(gust_col, wind_col, wave_col)
                ]
                show_df[display_name] = [_RISK_ZH.get(k, k) for k in op_risk]
            else:
                show_df[display_name] = raw

        except Exception:
            logger.warning("render_data_list：欄位 '%s' 處理失敗，已跳過", src_col)
            continue

    def _highlight_risk(row: pd.Series) -> list[str]:
        risk_zh = str(row.get("風險", ""))
        return _risk_row_style(risk_zh, len(row))

    def _cell_style(series: pd.Series, thr: dict) -> pd.Series:
        """對單一欄位各儲存格依閾值套用文字顏色 + 粗體"""
        def _fmt(val):
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            color = _thr_color(v, thr)
            if color == "#374151":   # 正常值，不特別標示
                return ""
            return f"color: {color}; font-weight: bold"
        return series.map(_fmt)

    styler = show_df.style.apply(_highlight_risk, axis=1)

    _COL_THR = {
        "陣風(kts)": _GUST_THR,
        "風速(kts)": _WIND_THR,
        "浪高(m)":   _WAVE_THR,
        "最大浪(m)": _WAVE_THR,
    }
    for col_name, thr in _COL_THR.items():
        if col_name in show_df.columns:
            styler = styler.apply(_cell_style, thr=thr, subset=[col_name])

    st.dataframe(
        styler,
        use_container_width=True,
        height=500,
    )


# ================= 歡迎頁面 =================

def render_welcome_page() -> None:
    """渲染初始歡迎頁面"""
    st.info("👈 請在左側選擇氣象來源（上傳檔案或自動抓取）")
    st.markdown(
        """
        ## 🚢 IWBDSS 船舶靠泊決策輔助系統

        ### 新版功能 (v2.0)
        - 🌙 **夜間作業偵測**：自動識別靠/離泊時段是否為夜間，並給予風險加成。
        - 🌪️ **大風大浪警示**：自動掃描在港期間是否有超過閾值的氣象狀況。
        - 🚨 **時窗風險檢查**：特別針對靠泊與離泊前後 2 小時進行高風險掃描。
        - 🤖 **AI 深度分析**：整合 Perplexity AI 進行專業航海諮詢。
        """
    )
