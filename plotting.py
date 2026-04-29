# plotting.py
"""
整合繪圖服務模組 (Integrated Plotting Service)

包含：
  1. PlotService 類別：Matplotlib 靜態圖表（用於詳細報告）
  2. 獨立函式：Plotly 互動式圖表（增強型時間軸、風玫瑰、熱圖等）

設計原則：
  - 所有物理常數引用 app_config.PHYSICS，不硬寫數值
  - 風險閾值引用 app_config.THRESHOLDS，不重複定義備援字典
  - 繪圖函式不修改傳入的 DataFrame（防禦性複製）
  - 錯誤一律透過 logging 記錄，不裸露 except: pass
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app_config import PHYSICS, RISK_LEVEL_SPECS, THRESHOLDS
from app_helpers import classify_wind_risk, get_risk_color

logger = logging.getLogger(__name__)


# ================= 圖表配色常數 =================

CHART_COLORS: Dict[str, str] = {
    "wind_speed":        "#1f77b4",
    "wind_gust":         "#ff7f0e",
    "wave_height":       "#2ca02c",
    "wind_force":        "#d62728",
    "safety_factor":     "#9467bd",
    "berthing":          "#17becf",
    "departure":         "#e377c2",
    "threshold_low":     "#90EE90",
    "threshold_medium":  "#FFD700",
    "threshold_high":    "#FF8C00",
    "threshold_extreme": "#FF4500",
    "capacity_line":     "#2E8B57",
}

RISK_LEVEL_COLORS: Dict[str, str] = {
    key: spec.color_bg
    for key, spec in RISK_LEVEL_SPECS.items()
}

_RISK_ORDER: List[str] = ["low", "medium", "high", "extreme"]


# ================= 私有輔助函式 =================

def _classify_wave_risk(wave_m: float) -> str:
    """波高 → 風險等級"""
    if wave_m >= 3.5:
        return "extreme"
    if wave_m >= 2.5:
        return "high"
    if wave_m >= 1.5:
        return "medium"
    return "low"


def _add_vessel_markers(
    ax: plt.Axes,
    arrival_time: Any,
    departure_time: Any,
) -> None:
    """在 Matplotlib 圖表上加入靠/離港時間標記線"""
    if arrival_time:
        ax.axvline(
            arrival_time, color="green", linestyle="--",
            label="靠港時間", linewidth=1.5,
        )
    if departure_time:
        ax.axvline(
            departure_time, color="red", linestyle="--",
            label="離港時間", linewidth=1.5,
        )


def _format_time_axis(ax: plt.Axes) -> None:
    """統一格式化 Matplotlib 時間軸"""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.xticks(rotation=45)


def _apply_common_ax_style(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """套用共用的 Axes 樣式"""
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    _format_time_axis(ax)


def _get_rec_val(rec: Any, *candidates: str, default: Any = 0.0) -> Any:
    """
    依序嘗試多個屬性名稱，回傳第一個存在且非 None 的值。

    修正說明：
      PlotService 原版直接存取 rec.wind_speed / rec.wind_gust，
      這些屬性在 AwtWeatherRecord 中不存在（AWT 版欄位為 wind_speed_kts）。
      改用此函式依序嘗試新舊欄位名稱，確保兩種記錄格式都能正確讀取。
    """
    for attr in candidates:
        val = getattr(rec, attr, None)
        if val is not None:
            return val
    return default


def _compute_mooring_capacity_kN(vessel: Any, result: Any) -> float:
    """
    計算纜繩總抓力 (kN)。

    優先從 result.wind_force_summary 讀取，
    若無則退回由 vessel 參數直接估算。
    """
    if hasattr(result, "wind_force_summary"):
        wfs = result.wind_force_summary
        if hasattr(wfs, "mooring_capacity_N") and wfs.mooring_capacity_N > 0:
            return wfs.mooring_capacity_N / 1000.0
        if isinstance(wfs, dict):
            cap = wfs.get("mooring_capacity_N", 0)
            if cap > 0:
                return cap / 1000.0

    if vessel is None:
        return 0.0

    mbl_kN      = vessel.mbl / 1000.0
    total_lines = (
        vessel.bow_lines + vessel.bow_spring_lines
        + vessel.stern_lines + vessel.stern_spring_lines
    )
    capacity = mbl_kN * vessel.safety_factor * total_lines
    logger.debug("纜繩抓力由 vessel 估算：%.1f kN", capacity)
    return capacity


# ================= 1. PlotService（Matplotlib 靜態圖表）=================

class PlotService:
    """
    Matplotlib 靜態繪圖服務。
    每個方法回傳獨立的 matplotlib Figure 物件。
    """

    def __init__(self, analyzer: Any) -> None:
        self.analyzer   = analyzer
        self.data       = analyzer.data
        self._port_name = getattr(analyzer, "port_name", "港口")

    # ── 風速趨勢 ──────────────────────────────────────────────

    def plot_wind_trend(
        self, vessel: Any, result: Any, figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """繪製風速趨勢圖（含靠/離港標記與風險閾值線）"""
        # ── 修正 Bug 1：AwtWeatherRecord 無 wind_speed / wind_gust 屬性 ────────
        # 原版：w.wind_speed / w.wind_gust → AttributeError（AWT 版欄位為 _kts 後綴）
        # 改寫：_get_rec_val() 依序嘗試新舊欄位名稱，同時相容兩種記錄格式
        times  = [w.time for w in self.data]
        speeds = [_get_rec_val(w, "wind_speed_kts", "wind_speed") for w in self.data]
        gusts  = [_get_rec_val(w, "wind_gust_kts",  "wind_gust")  for w in self.data]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times, speeds, "b-",  label="平均風速", linewidth=2)
        ax.plot(times, gusts,  "r--", label="最大陣風", linewidth=2)

        _add_vessel_markers(ax, vessel.arrival_time, vessel.departure_time)

        ax.axhline(
            THRESHOLDS.wind.medium, color="orange", linestyle=":", alpha=0.5,
            label=f"中風險 ({THRESHOLDS.wind.medium:.0f} kts)",
        )
        ax.axhline(
            THRESHOLDS.wind.high, color="red", linestyle=":", alpha=0.5,
            label=f"高風險 ({THRESHOLDS.wind.high:.0f} kts)",
        )

        _apply_common_ax_style(
            ax,
            xlabel="時間",
            ylabel="風速 (knots)",
            title=f"{self._port_name} - 風速趨勢",
        )
        plt.tight_layout()
        return fig

    # ── 浪高趨勢 ──────────────────────────────────────────────

    def plot_wave_trend(
        self, vessel: Any, result: Any, figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """繪製浪高趨勢圖（含顯著浪高與最大浪高）"""
        # ── AwtWeatherRecord 有 wave_height / wave_max 屬性（名稱相同），直接相容 ──
        # 但 wave_max 在 AWT 版可能為 0.0（非 None），仍用 _get_rec_val 保險處理
        times        = [w.time for w in self.data]
        wave_heights = [_get_rec_val(w, "wave_height", "wave_sig_m")  for w in self.data]
        wave_maxs    = [_get_rec_val(w, "wave_max",    "wave_max_m")  for w in self.data]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times, wave_heights, "c-",  label="顯著浪高 (Hs)",  linewidth=2)
        ax.plot(times, wave_maxs,    "m--", label="最大浪高 (Hmax)", linewidth=2)
        ax.fill_between(times, wave_heights, alpha=0.3, color="cyan")

        _add_vessel_markers(ax, vessel.arrival_time, vessel.departure_time)

        ax.axhline(2.0, color="orange", linestyle=":", alpha=0.5, label="警戒 (2.0 m)")
        ax.axhline(3.0, color="red",    linestyle=":", alpha=0.5, label="危險 (3.0 m)")

        _apply_common_ax_style(
            ax,
            xlabel="時間",
            ylabel="浪高 (m)",
            title=f"{self._port_name} - 浪高趨勢",
        )
        plt.tight_layout()
        return fig

    # ── 風力 vs 纜繩抓力 ──────────────────────────────────────

    def plot_force(
        self, vessel: Any, result: Any, figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        繪製風力趨勢圖，並疊加纜繩總抓力參考線。
        風力公式：F = 0.5 × ρ × Cd × A × V²
        """
        rho  = PHYSICS.air_density
        cd   = getattr(vessel, "wind_drag_coef", 1.0)
        area = vessel.wind_area

        times       = [w.time for w in self.data]
        avg_forces  = []
        gust_forces = []

        for w in self.data:
            # ── 修正：AwtWeatherRecord.wind_speed_ms / wind_gust_ms 為 property ──
            # 兩種格式均有此 property（AwtWeatherRecord 由 kts 換算，WeatherRecord 同理），
            # 直接存取即可，_get_rec_val 作為 fallback 保護。
            v_avg  = _get_rec_val(w, "wind_speed_ms", default=0.0)
            v_gust = _get_rec_val(w, "wind_gust_ms",  default=0.0)
            avg_forces.append( 0.5 * rho * cd * area * v_avg  ** 2 / 1000)
            gust_forces.append(0.5 * rho * cd * area * v_gust ** 2 / 1000)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times, avg_forces,  "g-",  label="平均風力", linewidth=2)
        ax.plot(times, gust_forces, "r--", label="陣風力",   linewidth=2)
        ax.fill_between(times, avg_forces, alpha=0.3, color="green")

        _add_vessel_markers(ax, vessel.arrival_time, vessel.departure_time)

        cap_kN = _compute_mooring_capacity_kN(vessel, result)
        if cap_kN > 0:
            ax.axhline(
                cap_kN, color="blue", linestyle="-.", linewidth=2.5,
                label=f"纜繩總抓力 ({cap_kN:.0f} kN)",
            )

        max_force_kN = max(gust_forces, default=0.0)
        if max_force_kN > 0:
            ax.axhline(
                max_force_kN, color="red", linestyle=":", alpha=0.5,
                label=f"最大風力 ({max_force_kN:.0f} kN)",
            )

        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_xlabel("時間", fontsize=12)
        ax.set_ylabel("風力 (kN)", fontsize=12)
        ax.set_title(
            f"{self._port_name} - 風力 vs 纜繩抓力分析",
            fontsize=14, fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        _format_time_axis(ax)
        plt.tight_layout()
        return fig


# ================= 2. Plotly 互動式繪圖函式 =================

def plot_enhanced_timeline(
    df: pd.DataFrame,
    berthing_time:  Optional[Any] = None,
    departure_time: Optional[Any] = None,
    berth_angle:    Optional[float] = None,
    vessel_info:    Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """
    增強型氣象時間軸（Plotly 互動式）。

    子圖組成（動態）：
      Row 1：風速與陣風（必有）
      Row 2：浪高（必有）
      Row 3：風力 vs 纜繩抓力（有風力欄位或 vessel_info 時顯示）
      Row 4：安全係數（有 safety_factor 欄位時顯示）
    """
    try:
        df_plot = _prepare_dataframe(df)
        if df_plot is None:
            return go.Figure()

        subplot_specs: List[Tuple[str, bool]] = [
            ("風速與陣風 (Wind)",             True),
            ("浪高 (Wave)",                   True),
            ("風力與抓力 (Force vs Capacity)",
             "gust_force_N" in df_plot.columns
             or "total_force_N" in df_plot.columns
             or vessel_info is not None),
            ("安全係數 (SF)", "safety_factor" in df_plot.columns),
        ]
        active_specs = [(title, _) for title, show in subplot_specs for _ in [show] if show]
        num_rows     = len(active_specs)
        titles       = [t for t, _ in active_specs]

        _ROW_HEIGHT_MAP = {
            2: [0.50, 0.50],
            3: [0.40, 0.30, 0.30],
            4: [0.30, 0.25, 0.25, 0.20],
        }
        row_heights = _ROW_HEIGHT_MAP.get(num_rows, [1.0 / num_rows] * num_rows)

        fig = make_subplots(
            rows=num_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=titles,
            row_heights=row_heights,
        )

        row = 1

        _add_wind_traces(fig, df_plot, row);  row += 1
        _add_wave_traces(fig, df_plot, row);  row += 1

        if subplot_specs[2][1]:
            _add_force_traces(fig, df_plot, vessel_info, row); row += 1

        if subplot_specs[3][1]:
            _add_safety_factor_traces(fig, df_plot, row)

        _add_plotly_time_markers(fig, berthing_time, departure_time)

        title_text = "增強型氣象時間軸分析"
        if berth_angle is not None:
            title_text += f" — {berth_angle:.0f}° 泊位"

        fig.update_layout(
            height=300 * num_rows,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            template="plotly_white",
            title=dict(text=title_text, x=0.5),
            margin=dict(t=80),
        )
        return fig

    except Exception:
        logger.exception("plot_enhanced_timeline 發生錯誤")
        return go.Figure()


# ── 子圖繪製輔助函式 ──────────────────────────────────────────────

def _prepare_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    驗證並標準化 DataFrame，回傳副本或 None。

    修正 Bug 3：原版呼叫 normalize_dataframe(df_plot)，
    但 app_helpers.normalize_dataframe 簽名為 (analyzer, vessel_info)，
    傳入 DataFrame 會導致 TypeError。

    改寫：此函式的職責是「確保 DataFrame 有必要欄位」，
    若已有 wind_speed_kts 則直接使用，不再嘗試呼叫 normalize_dataframe。
    normalize_dataframe 應在上游（app.py / ui_components.py）呼叫完成後
    再傳入此函式，plotting.py 不應持有 analyzer 參考。
    """
    df_plot = df.copy()

    # ── 確保有 time 欄位 ─────────────────────────────────────────────────────
    if "time" not in df_plot.columns:
        if isinstance(df_plot.index, pd.DatetimeIndex):
            df_plot["time"] = df_plot.index
        else:
            logger.error("DataFrame 缺少 'time' 欄位且索引非 DatetimeIndex")
            return None

    if not pd.api.types.is_datetime64_any_dtype(df_plot["time"]):
        df_plot["time"] = pd.to_datetime(df_plot["time"], errors="coerce")

    # ── 確保有風速欄位（向下相容舊版欄位名）────────────────────────────────
    # 若已有 wind_speed_kts（AWT 版 normalize_dataframe 輸出），直接使用。
    # 若只有舊版 wind_speed，補上 _kts 別名，確保後續繪圖函式能找到欄位。
    if "wind_speed_kts" not in df_plot.columns:
        if "wind_speed" in df_plot.columns:
            df_plot["wind_speed_kts"] = df_plot["wind_speed"]
            logger.debug("_prepare_dataframe：wind_speed → wind_speed_kts 別名補上")
        else:
            logger.warning("DataFrame 缺少 wind_speed_kts 欄位，圖表可能不完整")

    if "wind_gust_kts" not in df_plot.columns:
        if "wind_gust" in df_plot.columns:
            df_plot["wind_gust_kts"] = df_plot["wind_gust"]
            logger.debug("_prepare_dataframe：wind_gust → wind_gust_kts 別名補上")

    # ── 確保有浪高欄位 ───────────────────────────────────────────────────────
    if "wave_sig_m" not in df_plot.columns:
        if "wave_height" in df_plot.columns:
            df_plot["wave_sig_m"] = df_plot["wave_height"]
        elif "wave_height_m" in df_plot.columns:
            df_plot["wave_sig_m"] = df_plot["wave_height_m"]

    if "wave_max_m" not in df_plot.columns and "wave_max" in df_plot.columns:
        df_plot["wave_max_m"] = df_plot["wave_max"]

    return df_plot


def _add_wind_traces(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    """Row：風速與陣風"""
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["wind_speed_kts"],
            name="風速",
            line=dict(color=CHART_COLORS["wind_speed"], width=2),
            mode="lines",
            hovertemplate="<b>風速</b>: %{y:.1f} kts<extra></extra>",
        ),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["wind_gust_kts"],
            name="陣風",
            line=dict(color=CHART_COLORS["wind_gust"], width=2, dash="dash"),
            mode="lines",
            hovertemplate="<b>陣風</b>: %{y:.1f} kts<extra></extra>",
        ),
        row=row, col=1,
    )

    for val, label, color_key in [
        (THRESHOLDS.wind.medium, "中風險",    "threshold_medium"),
        (THRESHOLDS.wind.high,   "高風險",    "threshold_high"),
        (THRESHOLDS.gust.high,   "陣風高風險", "threshold_extreme"),
    ]:
        fig.add_hline(
            y=val, line_dash="dot",
            line_color=CHART_COLORS[color_key],
            annotation_text=label,
            row=row, col=1,
        )

    _add_risk_background(fig, df, "wind_gust_kts", row)
    fig.update_yaxes(title_text="風速 (節)", row=row, col=1)


def _add_wave_traces(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    """Row：浪高"""
    # ── wave_sig_m 由 _prepare_dataframe 確保存在 ────────────────────────────
    if "wave_sig_m" not in df.columns:
        logger.warning("_add_wave_traces：缺少 wave_sig_m 欄位，跳過浪高繪製")
        return

    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["wave_sig_m"],
            name="顯著浪高",
            line=dict(color=CHART_COLORS["wave_height"], width=2),
            fill="tozeroy", fillcolor="rgba(44, 160, 44, 0.2)",
            mode="lines",
            hovertemplate="<b>浪高</b>: %{y:.2f} m<extra></extra>",
        ),
        row=row, col=1,
    )
    if "wave_max_m" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time"], y=df["wave_max_m"],
                name="最大浪高",
                line=dict(color="darkgreen", width=1, dash="dot"),
                mode="lines",
                hovertemplate="<b>最大浪</b>: %{y:.2f} m<extra></extra>",
            ),
            row=row, col=1,
        )
    fig.update_yaxes(title_text="浪高 (m)", row=row, col=1)


def _add_force_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    vessel_info: Optional[Dict[str, Any]],
    row: int,
) -> None:
    """Row：風力 vs 纜繩抓力"""
    force_col = next(
        (c for c in ["gust_force_N", "total_force_N"] if c in df.columns),
        None,
    )
    if force_col:
        force_kN = pd.to_numeric(df[force_col], errors="coerce") / 1000.0
        fig.add_trace(
            go.Scatter(
                x=df["time"], y=force_kN,
                name="總風力",
                line=dict(color=CHART_COLORS["wind_force"], width=2),
                mode="lines",
                hovertemplate="<b>風力</b>: %{y:.1f} kN<extra></extra>",
            ),
            row=row, col=1,
        )

    if vessel_info and "mooring_capacity_kN" in vessel_info:
        cap_kN = vessel_info["mooring_capacity_kN"]
        fig.add_hline(
            y=cap_kN,
            line_dash="dashdot",
            line_color=CHART_COLORS["capacity_line"],
            line_width=3,
            annotation_text=f"纜繩抓力 ({cap_kN:.0f} kN)",
            annotation_position="top left",
            row=row, col=1,
        )

    fig.update_yaxes(title_text="風力 (kN)", row=row, col=1)


def _add_safety_factor_traces(
    fig: go.Figure, df: pd.DataFrame, row: int
) -> None:
    """Row：安全係數"""
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["safety_factor"],
            name="安全係數",
            line=dict(color=CHART_COLORS["safety_factor"], width=2),
            mode="lines+markers",
            marker=dict(size=4),
            hovertemplate="<b>SF</b>: %{y:.2f}<extra></extra>",
        ),
        row=row, col=1,
    )
    fig.add_hline(y=1.5, line_dash="dash", line_color="green",
                  annotation_text="合格 (1.5)", row=row, col=1)
    fig.add_hline(y=1.2, line_dash="dash", line_color="red",
                  annotation_text="危險 (1.2)", row=row, col=1)
    fig.update_yaxes(title_text="係數 (SF)", row=row, col=1)


def _add_plotly_time_markers(
    fig: go.Figure,
    berthing_time:  Optional[Any],
    departure_time: Optional[Any],
) -> None:
    """在 Plotly 圖表上加入靠/離泊時間標記"""
    markers = [
        (berthing_time,  CHART_COLORS["berthing"],  "靠泊"),
        (departure_time, CHART_COLORS["departure"], "離泊"),
    ]
    for t, color, label in markers:
        if t is None:
            continue
        fig.add_shape(
            type="line", x0=t, y0=0, x1=t, y1=1,
            xref="x", yref="paper",
            line=dict(color=color, width=2, dash="solid"),
        )
        fig.add_annotation(
            x=t, y=1.02, xref="x", yref="paper",
            text=label, showarrow=False,
            font=dict(color=color, size=12),
        )


def _add_risk_background(
    fig: go.Figure, df: pd.DataFrame, column: str, row: int
) -> None:
    """
    在 Plotly 子圖中加入風險等級背景色帶。
    不修改傳入的 df（防禦性設計）。
    """
    try:
        if column == "wind_gust_kts" and "wind_speed_kts" in df.columns:
            risk_series = df.apply(
                lambda r: classify_wind_risk(
                    float(r.get("wind_speed_kts", 0.0)),
                    float(r.get(column, 0.0)),
                ),
                axis=1,
            )
        elif column == "wave_sig_m":
            risk_series = df[column].apply(_classify_wave_risk)
        else:
            return

        prev      = None
        start_idx = 0
        for i, level in enumerate(risk_series):
            if level != prev:
                if prev is not None and prev != "low":
                    fig.add_vrect(
                        x0=df.iloc[start_idx]["time"],
                        x1=df.iloc[i]["time"],
                        fillcolor=get_risk_color(prev),
                        opacity=0.15, layer="below", line_width=0,
                        row=row, col=1,
                    )
                prev      = level
                start_idx = i

        if prev and prev != "low" and start_idx < len(df) - 1:
            fig.add_vrect(
                x0=df.iloc[start_idx]["time"],
                x1=df.iloc[-1]["time"],
                fillcolor=get_risk_color(prev),
                opacity=0.15, layer="below", line_width=0,
                row=row, col=1,
            )

    except Exception:
        logger.warning("_add_risk_background 失敗，跳過背景色帶", exc_info=True)


# ================= 獨立 Plotly 圖表函式 =================

def plot_wind_rose(df: pd.DataFrame, title: str = "風玫瑰圖") -> go.Figure:
    """風玫瑰圖：顯示各方向風速分布"""
    try:
        if "wind_dir_deg" not in df.columns or "wind_speed_kts" not in df.columns:
            logger.warning("plot_wind_rose：缺少必要欄位 wind_dir_deg 或 wind_speed_kts")
            return go.Figure()

        df_plot = df.copy()
        direction_labels = [
            "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
        ]
        speed_bins   = [0, 10, 20, 30, 40, 100]
        speed_labels = ["0–10", "10–20", "20–30", "30–40", ">40"]

        df_plot["direction_bin"] = pd.cut(
            df_plot["wind_dir_deg"], bins=16, labels=direction_labels,
        )
        df_plot["speed_bin"] = pd.cut(
            df_plot["wind_speed_kts"], bins=speed_bins, labels=speed_labels,
        )

        wind_rose_data = (
            df_plot.groupby(["direction_bin", "speed_bin"], observed=False)
            .size()
            .unstack(fill_value=0)
        )

        colors = [
            CHART_COLORS["threshold_low"],
            CHART_COLORS["threshold_medium"],
            CHART_COLORS["threshold_high"],
            CHART_COLORS["threshold_extreme"],
            "#8B0000",
        ]

        fig = go.Figure()
        for i, col in enumerate(wind_rose_data.columns):
            fig.add_trace(
                go.Barpolar(
                    r=wind_rose_data[col].values,
                    theta=wind_rose_data.index.astype(str),
                    name=f"{col} kts",
                    marker_color=colors[min(i, len(colors) - 1)],
                )
            )

        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(visible=True),
                angularaxis=dict(direction="clockwise"),
            ),
            height=500,
        )
        return fig

    except Exception:
        logger.exception("plot_wind_rose 發生錯誤")
        return go.Figure()


def plot_risk_heatmap(df: pd.DataFrame, title: str = "風險熱圖") -> go.Figure:
    """風險等級熱圖：以色帶顯示時間序列風險變化"""
    try:
        if "risk_level" not in df.columns:
            logger.warning("plot_risk_heatmap：缺少 risk_level 欄位")
            return go.Figure()

        _RISK_NUM = {"low": 1, "medium": 2, "high": 3, "extreme": 4}
        z_vals    = df["risk_level"].map(_RISK_NUM).fillna(1)

        colorscale = [
            [0.00, RISK_LEVEL_COLORS["low"]],
            [0.33, RISK_LEVEL_COLORS["medium"]],
            [0.66, RISK_LEVEL_COLORS["high"]],
            [1.00, RISK_LEVEL_COLORS["extreme"]],
        ]

        fig = go.Figure(
            data=go.Heatmap(
                x=df["time"],
                y=["風險等級"],
                z=[z_vals],
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title="等級",
                    tickvals=[1, 2, 3, 4],
                    ticktext=["低", "中", "高", "極高"],
                ),
            )
        )
        fig.update_layout(title=title, height=250)
        return fig

    except Exception:
        logger.exception("plot_risk_heatmap 發生錯誤")
        return go.Figure()


def plot_safety_factor_timeline(
    df: pd.DataFrame,
    required_sf: float = 1.5,
    title: str = "安全係數時間軸",
) -> go.Figure:
    """安全係數時間軸：顯示 SF 變化並標記合格基準線"""
    try:
        if "safety_factor" not in df.columns:
            logger.warning("plot_safety_factor_timeline：缺少 safety_factor 欄位")
            return go.Figure()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["time"], y=df["safety_factor"],
                name="安全係數",
                line=dict(color=CHART_COLORS["safety_factor"], width=2),
                mode="lines+markers",
                marker=dict(size=4),
                hovertemplate="<b>SF</b>: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_hline(
            y=required_sf, line_dash="dash", line_color="red",
            annotation_text=f"要求值 ({required_sf})",
        )
        fig.update_layout(
            title=title,
            yaxis_title="安全係數 (SF)",
            xaxis_title="時間",
            height=400,
            template="plotly_white",
        )
        return fig

    except Exception:
        logger.exception("plot_safety_factor_timeline 發生錯誤")
        return go.Figure()
