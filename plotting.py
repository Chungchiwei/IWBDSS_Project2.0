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
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app_config import PHYSICS, RISK_LEVEL_SPECS, THRESHOLDS
from app_helpers import classify_wind_risk, get_risk_color, normalize_dataframe

logger = logging.getLogger(__name__)


# ================= 圖表配色常數 =================

CHART_COLORS: Dict[str, str] = {
    "wind_speed":        "#1f77b4",   # 藍色
    "wind_gust":         "#ff7f0e",   # 橙色
    "wave_height":       "#2ca02c",   # 綠色
    "wind_force":        "#d62728",   # 紅色
    "safety_factor":     "#9467bd",   # 紫色
    "berthing":          "#17becf",   # 青色
    "departure":         "#e377c2",   # 粉色
    "threshold_low":     "#90EE90",   # 淺綠
    "threshold_medium":  "#FFD700",   # 金黃
    "threshold_high":    "#FF8C00",   # 深橙
    "threshold_extreme": "#FF4500",   # 紅橙
    "capacity_line":     "#2E8B57",   # 海綠色
}

# 從 RISK_LEVEL_SPECS 自動生成，不重複定義
RISK_LEVEL_COLORS: Dict[str, str] = {
    key: spec.color_bg
    for key, spec in RISK_LEVEL_SPECS.items()
}

# 風險等級排序（用於比較嚴重程度）
_RISK_ORDER: List[str] = ["low", "medium", "high", "extreme"]


# ================= 私有輔助函式 =================

def _classify_wave_risk(wave_m: float) -> str:
    """
    波高 → 風險等級。

    使用 THRESHOLDS 中的波高閾值，與風速風險等級保持一致的設計。
    """
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
    """在 Matplotlib 圖表上加入靠/離港時間標記線（共用邏輯）"""
    if arrival_time:
        ax.axvline(
            arrival_time, color="green", linestyle="--",
            label="Berthing", linewidth=1.5,
        )
    if departure_time:
        ax.axvline(
            departure_time, color="red", linestyle="--",
            label="Departure", linewidth=1.5,
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


def _compute_mooring_capacity_kN(vessel: Any, result: Any) -> float:
    """
    計算纜繩總抓力 (kN)。

    優先從 result.wind_force_summary 讀取（已由分析模組計算），
    若無則退回由 vessel 參數直接估算。

    Returns:
        纜繩總抓力 (kN)，無法計算時回傳 0.0
    """
    # 優先：從分析結果讀取（型別化 WindForceSummary）
    if hasattr(result, "wind_force_summary"):
        wfs = result.wind_force_summary
        # 支援新版 WindForceSummary dataclass
        if hasattr(wfs, "mooring_capacity_N") and wfs.mooring_capacity_N > 0:
            return wfs.mooring_capacity_N / 1000.0
        # 相容舊版 dict
        if isinstance(wfs, dict):
            cap = wfs.get("mooring_capacity_N", 0)
            if cap > 0:
                return cap / 1000.0

    # 備援：從 vessel 直接估算
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

    主要用於生成報告中的靜態趨勢圖，
    每個方法回傳獨立的 matplotlib Figure 物件。
    """

    def __init__(self, analyzer: Any) -> None:
        self.analyzer = analyzer
        self.data      = analyzer.data
        raw_name = getattr(analyzer, "port_name", "Port") or "Port"
        # Strip non-ASCII characters (Chinese) so Matplotlib can render the title
        ascii_name = "".join(c for c in raw_name if ord(c) < 128).strip()
        self._port_name: str = ascii_name if ascii_name else "Port"

    # ── 風速趨勢 ──────────────────────────────────────────────

    def plot_wind_trend(
        self, vessel: Any, _result: Any = None, figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """繪製風速趨勢圖（含靠/離港標記與風險閾值線）"""
        times  = [w.time        for w in self.data]
        speeds = [w.wind_speed  for w in self.data]
        gusts  = [w.wind_gust   for w in self.data]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times, speeds, "b-",  label="Wind Speed", linewidth=2)
        ax.plot(times, gusts,  "r--", label="Wind Gust",  linewidth=2)

        _add_vessel_markers(ax, vessel.arrival_time, vessel.departure_time)

        ax.axhline(
            THRESHOLDS.wind.medium, color="orange", linestyle=":", alpha=0.5,
            label=f"Warning ({THRESHOLDS.wind.medium:.0f} kts)",
        )
        ax.axhline(
            THRESHOLDS.wind.high, color="red", linestyle=":", alpha=0.5,
            label=f"Danger ({THRESHOLDS.wind.high:.0f} kts)",
        )

        _apply_common_ax_style(
            ax,
            xlabel="Time",
            ylabel="Wind Speed (kts)",
            title=f"{self._port_name} - Wind Trend",
        )
        plt.tight_layout()
        return fig

    # ── 浪高趨勢 ──────────────────────────────────────────────

    def plot_wave_trend(
        self, vessel: Any, _result: Any = None, figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """繪製浪高趨勢圖（含顯著浪高與最大浪高）"""
        times        = [w.time        for w in self.data]
        wave_heights = [w.wave_height for w in self.data]
        wave_maxs    = [w.wave_max    for w in self.data]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times, wave_heights, "c-",  label="Sig. Wave Ht (Hs)",  linewidth=2)
        ax.plot(times, wave_maxs,    "m--", label="Max Wave Ht (Hmax)",  linewidth=2)
        ax.fill_between(times, wave_heights, alpha=0.3, color="cyan")

        _add_vessel_markers(ax, vessel.arrival_time, vessel.departure_time)

        ax.axhline(2.0, color="orange", linestyle=":", alpha=0.5, label="Caution (2.0 m)")
        ax.axhline(3.0, color="red",    linestyle=":", alpha=0.5, label="Danger (3.0 m)")

        _apply_common_ax_style(
            ax,
            xlabel="Time",
            ylabel="Wave Height (m)",
            title=f"{self._port_name} - Wave Trend",
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
            v_avg  = w.wind_speed_ms   # 使用 WeatherRecord property，不硬寫係數
            v_gust = w.wind_gust_ms
            avg_forces.append( 0.5 * rho * cd * area * v_avg  ** 2 / 1000)  # → kN
            gust_forces.append(0.5 * rho * cd * area * v_gust ** 2 / 1000)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times, avg_forces,  "g-",  label="Avg Wind Force", linewidth=2)
        ax.plot(times, gust_forces, "r--", label="Gust Force",     linewidth=2)
        ax.fill_between(times, avg_forces, alpha=0.3, color="green")

        _add_vessel_markers(ax, vessel.arrival_time, vessel.departure_time)

        cap_kN = _compute_mooring_capacity_kN(vessel, result)
        if cap_kN > 0:
            ax.axhline(
                cap_kN, color="blue", linestyle="-.", linewidth=2.5,
                label=f"Mooring Capacity ({cap_kN:.0f} kN)",
            )

        max_force_kN = max(gust_forces, default=0.0)
        if max_force_kN > 0:
            ax.axhline(
                max_force_kN, color="red", linestyle=":", alpha=0.5,
                label=f"Max Force ({max_force_kN:.0f} kN)",
            )

        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Force (kN)", fontsize=12)
        ax.set_title(
            f"{self._port_name} - Wind Force vs Mooring Capacity",
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

        # ── 決定子圖結構 ──────────────────────────────────────
        subplot_specs: List[Tuple[str, bool]] = [
            ("Wind Speed & Gust",            True),
            ("Wave Height",                  True),
            ("Force vs Mooring Capacity",
             "gust_force_N" in df_plot.columns
             or "total_force_N" in df_plot.columns
             or vessel_info is not None),
            ("Safety Factor (SF)",           "safety_factor" in df_plot.columns),
        ]
        active_specs = [(title, _) for title, show in subplot_specs for _ in [show] if show]
        num_rows     = len(active_specs)
        titles       = [t for t, _ in active_specs]

        _ROW_HEIGHT_MAP = {2: [0.50, 0.50], 3: [0.40, 0.30, 0.30], 4: [0.30, 0.25, 0.25, 0.20]}
        row_heights = _ROW_HEIGHT_MAP.get(num_rows, [1.0 / num_rows] * num_rows)

        fig = make_subplots(
            rows=num_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=titles,
            row_heights=row_heights,
        )

        row = 1

        # ── Row 1：風速 ────────────────────────────────────────
        _add_wind_traces(fig, df_plot, row)
        row += 1

        # ── Row 2：浪高 ────────────────────────────────────────
        _add_wave_traces(fig, df_plot, row)
        row += 1

        # ── Row 3：風力（條件性）──────────────────────────────
        if subplot_specs[2][1]:
            _add_force_traces(fig, df_plot, vessel_info, row)
            row += 1

        # ── Row 4：安全係數（條件性）─────────────────────────
        if subplot_specs[3][1]:
            _add_safety_factor_traces(fig, df_plot, row)

        # ── 靠/離泊標記 ───────────────────────────────────────
        _add_plotly_time_markers(fig, berthing_time, departure_time)

        title_text = "Enhanced Weather Timeline"
        if berth_angle is not None:
            title_text += f" — Berth {berth_angle:.0f}°"

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
    """驗證並標準化 DataFrame，回傳副本或 None"""
    df_plot = df.copy()

    if "wind_speed_kts" not in df_plot.columns:
        try:
            df_plot = normalize_dataframe(df_plot)
        except Exception:
            logger.warning("normalize_dataframe 失敗，嘗試繼續使用原始資料")

    if "time" not in df_plot.columns:
        if isinstance(df_plot.index, pd.DatetimeIndex):
            df_plot["time"] = df_plot.index
        else:
            logger.error("DataFrame 缺少 'time' 欄位且索引非 DatetimeIndex")
            return None

    if not pd.api.types.is_datetime64_any_dtype(df_plot["time"]):
        df_plot["time"] = pd.to_datetime(df_plot["time"])

    return df_plot


def _add_wind_traces(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    """Row: Wind Speed & Gust"""
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["wind_speed_kts"],
            name="Wind Speed",
            line=dict(color=CHART_COLORS["wind_speed"], width=2),
            mode="lines",
            hovertemplate="<b>Wind Speed</b>: %{y:.1f} kts<extra></extra>",
        ),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["wind_gust_kts"],
            name="Wind Gust",
            line=dict(color=CHART_COLORS["wind_gust"], width=2, dash="dash"),
            mode="lines",
            hovertemplate="<b>Wind Gust</b>: %{y:.1f} kts<extra></extra>",
        ),
        row=row, col=1,
    )

    for val, label, color_key in [
        (THRESHOLDS.wind.medium, f"Warning ({THRESHOLDS.wind.medium:.0f} kts)", "threshold_medium"),
        (THRESHOLDS.wind.high,   f"Danger ({THRESHOLDS.wind.high:.0f} kts)",    "threshold_high"),
        (THRESHOLDS.gust.high,   f"Gust Danger ({THRESHOLDS.gust.high:.0f} kts)", "threshold_extreme"),
    ]:
        fig.add_hline(
            y=val, line_dash="dot",
            line_color=CHART_COLORS[color_key],
            annotation_text=label,
            row=row, col=1,
        )

    _add_risk_background(fig, df, "wind_gust_kts", row)
    fig.update_yaxes(title_text="Wind Speed (kts)", row=row, col=1)


def _add_wave_traces(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    """Row: Wave Height"""
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["wave_sig_m"],
            name="Sig. Wave Ht",
            line=dict(color=CHART_COLORS["wave_height"], width=2),
            fill="tozeroy", fillcolor="rgba(44, 160, 44, 0.2)",
            mode="lines",
            hovertemplate="<b>Sig. Wave</b>: %{y:.2f} m<extra></extra>",
        ),
        row=row, col=1,
    )
    if "wave_max_m" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time"], y=df["wave_max_m"],
                name="Max Wave Ht",
                line=dict(color="darkgreen", width=1, dash="dot"),
                mode="lines",
                hovertemplate="<b>Max Wave</b>: %{y:.2f} m<extra></extra>",
            ),
            row=row, col=1,
        )
    fig.update_yaxes(title_text="Wave Height (m)", row=row, col=1)


def _add_force_traces(
    fig: go.Figure,
    df: pd.DataFrame,
    vessel_info: Optional[Dict[str, Any]],
    row: int,
) -> None:
    """Row: Wind Force vs Mooring Capacity"""
    force_col = next(
        (c for c in ["gust_force_N", "total_force_N"] if c in df.columns),
        None,
    )
    if force_col:
        force_kN = pd.to_numeric(df[force_col], errors="coerce") / 1000.0
        fig.add_trace(
            go.Scatter(
                x=df["time"], y=force_kN,
                name="Wind Force",
                line=dict(color=CHART_COLORS["wind_force"], width=2),
                mode="lines",
                hovertemplate="<b>Force</b>: %{y:.1f} kN<extra></extra>",
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
            annotation_text=f"Mooring Capacity ({cap_kN:.0f} kN)",
            annotation_position="top left",
            row=row, col=1,
        )

    fig.update_yaxes(title_text="Force (kN)", row=row, col=1)


def _add_safety_factor_traces(
    fig: go.Figure, df: pd.DataFrame, row: int
) -> None:
    """Row: Safety Factor"""
    fig.add_trace(
        go.Scatter(
            x=df["time"], y=df["safety_factor"],
            name="Safety Factor",
            line=dict(color=CHART_COLORS["safety_factor"], width=2),
            mode="lines+markers",
            marker=dict(size=4),
            hovertemplate="<b>SF</b>: %{y:.2f}<extra></extra>",
        ),
        row=row, col=1,
    )
    fig.add_hline(y=1.5, line_dash="dash", line_color="green",
                  annotation_text="Adequate (1.5)", row=row, col=1)
    fig.add_hline(y=1.2, line_dash="dash", line_color="red",
                  annotation_text="Danger (1.2)", row=row, col=1)
    fig.update_yaxes(title_text="Safety Factor (SF)", row=row, col=1)


def _add_plotly_time_markers(
    fig: go.Figure,
    berthing_time:  Optional[Any],
    departure_time: Optional[Any],
) -> None:
    """在 Plotly 圖表上加入靠/離泊時間標記"""
    markers = [
        (berthing_time,  CHART_COLORS["berthing"],  "Berthing"),
        (departure_time, CHART_COLORS["departure"], "Departure"),
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

    注意：不修改傳入的 df，在內部建立臨時 Series。
    """
    try:
        if column == "wind_gust_kts" and "wind_speed_kts" in df.columns:
            risk_series = df.apply(
                lambda r: classify_wind_risk(
                    r.get("wind_speed_kts", 0.0),
                    r.get(column, 0.0),
                ),
                axis=1,
            )
        elif column == "wave_sig_m":
            risk_series = df[column].apply(_classify_wave_risk)
        else:
            return

        # 找出風險等級變化點
        prev = None
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
                prev = level
                start_idx = i

        # 處理最後一段
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

def plot_wind_rose(df: pd.DataFrame, title: str = "Wind Rose") -> go.Figure:
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


def plot_risk_heatmap(df: pd.DataFrame, title: str = "Risk Heatmap") -> go.Figure:
    """Risk level heatmap: colour band showing risk level over time"""
    try:
        if "risk_level" not in df.columns:
            logger.warning("plot_risk_heatmap: missing risk_level column")
            return go.Figure()

        _RISK_NUM = {"low": 1, "medium": 2, "high": 3, "extreme": 4}
        z_vals = df["risk_level"].map(_RISK_NUM).fillna(1)

        colorscale = [
            [0.00, RISK_LEVEL_COLORS["low"]],
            [0.33, RISK_LEVEL_COLORS["medium"]],
            [0.66, RISK_LEVEL_COLORS["high"]],
            [1.00, RISK_LEVEL_COLORS["extreme"]],
        ]

        fig = go.Figure(
            data=go.Heatmap(
                x=df["time"],
                y=["Risk Level"],
                z=[z_vals],
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title="Level",
                    tickvals=[1, 2, 3, 4],
                    ticktext=["Low", "Med", "High", "Extreme"],
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
