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
    """渲染港口資訊卡片"""
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

        st.markdown(
            f"<h1 style='text-align:center;color:#1f77b4;font-size:36px;'>🏝️ {name}</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style='background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                        padding:20px;border-radius:15px;margin:10px 0;color:white;'>
              <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:10px;text-align:center;'>
                <div><small>Port Code</small><br><b>{code}</b><br><small>{country}</small></div>
                <div><small>Station ID</small><br><b>{station_id}</b></div>
                <div><small>Coordinates</small><br><b>{lat_ns} / {lon_ew}</b></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        port_name = getattr(analyzer, "port_name", "Unknown Port") if analyzer else "Unknown Port"
        st.markdown(
            f"<h1 style='text-align:center;font-size:36px;'>⚓ {port_name}</h1>",
            unsafe_allow_html=True,
        )


# ================= KPI 指標 =================

def render_kpi_metrics(result: AnalysisResult) -> None:
    """渲染四格 KPI 指標列"""
    wfs         = result.wind_force_summary
    max_force_N = _get_wfs_value(wfs, "max_gust_force_N", "max_gust_force_N")
    sf          = _get_wfs_value(wfs, "safety_factor",    "safety_factor")

    bow_status  = result.mooring_split.bow.status
    delta_color = "normal" if bow_status == "OK" else "inverse"
    risk_delta  = result.risk_score - result.mitigated_risk_score

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("原始風險",   f"{result.risk_score:.1f}",           result.risk_level,  delta_color="inverse")
    k2.metric("緩解後風險", f"{result.mitigated_risk_score:.1f}", f"降幅 {risk_delta:.1f}")
    k3.metric("最大受力",   f"{max_force_N / 1000:.0f} kN",       f"SF {sf:.2f}")
    k4.metric("繫泊狀態",   bow_status,                            delta_color=delta_color)


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

def render_detail_report(result: AnalysisResult, sidebar_data: Dict[str, Any]) -> None:
    """渲染詳細報告（風險警示、纜繩配置、拖船建議、受力分析）"""

    # ── 1. 風險警示 ───────────────────────────────────────────
    st.subheader("⚠️ 風險警示與操作建議")
    if result.recommendations:
        for rec in result.recommendations:
            _render_recommendation(rec)
    else:
        st.success("✅ 目前評估無特殊風險警示", icon="✅")

    st.markdown("---")

    # ── 2. 纜繩配置 ───────────────────────────────────────────
    with st.expander("⚓ 船體與纜繩配置總覽", expanded=True):
        c1, c2 = st.columns(2)
        c1.write(f"**吃水：** 艏 {sidebar_data['draft_b']} m / 艉 {sidebar_data['draft_s']} m")
        c2.write(f"**受風面積：** {sidebar_data['area']} m²")

        # 顯示船型來源
        vt_display = sidebar_data.get("vessel_type_display", "手動輸入")
        if vt_display != "手動輸入":
            c1.caption(f"🚢 船型：{vt_display}")

        st.write(
            f"**纜繩配置：** "
            f"艏（{sidebar_data['bh']} 頭 / {sidebar_data['bs']} 倒）| "
            f"艉（{sidebar_data['sh']} 尾 / {sidebar_data['ss']} 倒）"
        )

        m = result.mooring_split
        if m.bow.status != "OK" or m.stern.status != "OK":
            st.warning(
                f"⚠️ 建議：船艏 [{m.bow.recommendation_text}] / "
                f"船艉 [{m.stern.recommendation_text}]"
            )
        else:
            st.success("✅ 纜繩配置足夠")

    # ── 3. 拖船建議 ───────────────────────────────────────────
    with st.expander("🚤 拖船配置建議", expanded=True):
        tug = result.tug_recommendation

        final_count = _get_tug_value(tug, "final_tug_count",    "final_tug_count",    0)
        adequacy    = _get_tug_value(tug, "adequacy",            "adequacy",           False)
        reasons     = _get_tug_value(tug, "enforcement_reasons", "enforcement_reasons", [])

        c1, c2 = st.columns(2)
        c1.metric("需求數量", f"{final_count} 艘")
        c2.metric("推力狀態", "充足" if adequacy else "不足")

        if reasons:
            st.write("**增派理由：**")
            for r in reasons:
                st.caption(f"- {r}")

    # ── 4. 受力分析 ───────────────────────────────────────────
    with st.expander("💨 受力分析詳情"):
        wfs = result.wind_force_summary

        if hasattr(wfs, "max_gust_record"):
            rec = wfs.max_gust_record
        else:
            rec = wfs.get("max_gust_record", {}) if isinstance(wfs, dict) else {}

        rec_time = rec.get("time") if isinstance(rec, dict) else getattr(rec, "time", None)
        rec_gust = rec.get("wind_gust") if isinstance(rec, dict) else getattr(rec, "wind_gust", "N/A")

        if rec_time:
            st.write(f"**最大受力時刻：** {rec_time.strftime('%Y-%m-%d %H:%M')}")

        trans_N = _get_wfs_value(wfs, "max_trans_force_N", "max_trans_force_N")
        long_N  = _get_wfs_value(wfs, "max_long_force_N",  "max_long_force_N")
        sf      = _get_wfs_value(wfs, "safety_factor",     "safety_factor")
        w_type  = (
            getattr(wfs, "wind_type", None)
            or (wfs.get("wind_type", "N/A") if isinstance(wfs, dict) else "N/A")
        )

        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("最大陣風",  f"{rec_gust} kts")
        col_f2.metric("橫向受力",  f"{trans_N / 1000:.0f} kN")
        col_f3.metric("縱向受力",  f"{long_N  / 1000:.0f} kN")
        st.caption(f"風向性質：{w_type}（安全係數：{sf:.2f}）")


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
    safety_margin_kN = (
        mooring_cap.total_capacity_kN
        + tug_cap.total_push_kN
        - max_force_N / 1000.0
    )

    # ── 快速摘要（不呼叫 API）────────────────────────────────
    if ai_mode == "快速摘要":
        summary = st.session_state.ai_analyzer.generate_quick_summary(
            result.risk_score,
            result.risk_level,
            safety_margin_kN,
            safety_factor=sf,
        )
        st.markdown(summary)
        return

    # ── 完整分析（呼叫 Perplexity API）──────────────────────
    with st.spinner("AI 正在進行深度分析（Perplexity）..."):

        total_records  = len(df_detail)
        offshore_count = (
            int(pd.to_numeric(df_detail["is_offshore"], errors="coerce").fillna(0).sum())
            if "is_offshore" in df_detail.columns
            else 0
        )
        max_trans_N = _get_wfs_value(
            result.wind_force_summary, "max_trans_force_N", "max_trans_force_N"
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
    analyzer:     Any,
    sidebar_data: Dict[str, Any],
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
