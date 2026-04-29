# analysis.py
"""
船舶靠泊繫留風險評估系統 - 核心分析模組
修正纜繩拉力計算與風險等級評估
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from models import (
    AnalysisResult,
    MooringStatus,
    MooringSplit,
    VesselInfo,
    WeatherRecord,
)

try:
    from app_helpers import AIR_DENSITY, compass_to_degrees, knots_to_ms
except ImportError:
    AIR_DENSITY = 1.225

    def knots_to_ms(k: float) -> float:
        return k * 0.51444

    def compass_to_degrees(c: str) -> float:
        return 0.0


logger = logging.getLogger(__name__)


# ================= 常數定義 =================

@dataclass(frozen=True)
class WeatherThresholds:
    """氣象風險閾值（集中管理，方便調整）"""
    high_wind_speed_kts: float = 20.0
    high_gust_speed_kts: float = 35.0
    extreme_gust_kts:    float = 50.0
    high_wave_sig_m:     float = 2.5
    very_high_wave_sig_m: float = 4.0
    moderate_wave_sig_m: float = 1.5
    night_start_hour:    int = 20
    night_end_hour:      int = 6


@dataclass(frozen=True)
class MooringEfficiency:
    """纜繩效率係數（考慮纜繩角度與摩擦）"""
    head_transverse:   float = 0.95   # 頭纜橫向效率
    head_longitudinal: float = 0.15   # 頭纜縱向效率
    spring_transverse: float = 0.25   # 倒纜橫向效率
    spring_longitudinal: float = 0.95 # 倒纜縱向效率


@dataclass(frozen=True)
class RiskScoreWeights:
    """風險分數各項權重上限"""
    wind:        int = 40
    wave:        int = 30
    safety_factor: int = 20
    time_window: int = 10   # 靠/離泊時窗各 5 分
    night_ops:   int = 5
    port_level:  int = 5


# 模組層級共用常數實例
THRESHOLDS    = WeatherThresholds()
MOORING_EFF   = MooringEfficiency()
SCORE_WEIGHTS = RiskScoreWeights()

# 港口風險等級 → 所需安全係數
_PORT_RISK_SF_MAP: List[Tuple[int, float]] = [
    (3,  1.5),
    (6,  1.7),
    (10, 2.0),
]

# 風險分數 → 風險等級
_RISK_SCORE_LEVELS: List[Tuple[int, str]] = [
    (70, "extreme"),
    (50, "high"),
    (30, "medium"),
    (0,  "low"),
]

HP_TO_BOLLARD_PULL_PER_100HP = 1.1
GRAVITY = 9.81


# ================= 輔助資料結構 =================

@dataclass
class WindForceResult:
    """風力計算結果"""
    total_force_N:       float
    transverse_force_N:  float
    longitudinal_force_N: float
    wind_type:           str   # "offshore" | "onshore" | "parallel"


@dataclass
class MooringRestraintResult:
    """纜繩抗力計算結果"""
    transverse_restraint_kN:   float
    longitudinal_restraint_kN: float
    total_capacity_N:          float
    wll_single_N:              float


@dataclass
class RiskWindowResult:
    """時間窗口風險檢查結果（擴充版，供 UI 使用）"""
    risks:              List[str] = field(default_factory=list)
    max_wind_gust:      float     = 0.0
    max_wind_direction: str       = ""
    max_wave_height:    float     = 0.0
    max_wave_period:    float     = 0.0
    high_risk_hours:    int       = 0
    window_start:       Optional[datetime] = None
    window_end:         Optional[datetime] = None
    dominant_wind_type: str       = ""
    condition_risks:    List[str] = field(default_factory=list)

    @property
    def has_risk(self) -> bool:
        return bool(self.risks) and self.risks[0] != "無該時段氣象資料"


# analysis.py — 替換 _check_window_risk 方法

def _check_window_risk(
    self, target: datetime, window_hours: int = 2
) -> RiskWindowResult:
    """檢查目標時間前後 window_hours 小時內的風險，回傳豐富結果供 UI 使用"""
    start  = target - timedelta(hours=window_hours)
    end    = target + timedelta(hours=window_hours)
    window = [r for r in self.data if start <= r.time <= end]

    if not window:
        return RiskWindowResult(risks=["無該時段氣象資料"])

    max_gust_rec  = max(window, key=lambda r: r.wind_gust)
    max_wave_rec  = max(window, key=lambda r: r.wave_height)
    max_gust      = max_gust_rec.wind_gust
    max_wave      = max_wave_rec.wave_height
    max_period    = max(r.wave_period for r in window)
    high_risk_h   = sum(
        1 for r in window
        if r.wind_gust >= THRESHOLDS.high_gust_speed_kts
    )

    # 主要風型（取陣風最大那筆）
    heading   = self._vessel_heading_default()
    wind_deg  = compass_to_degrees(max_gust_rec.wind_direction)
    relative  = (wind_deg - heading + 180) % 360 - 180
    abs_rel   = abs(relative)
    if 45 <= abs_rel <= 135:
        dom_wind_type = "offshore" if relative > 0 else "onshore"
    else:
        dom_wind_type = "parallel"

    risks: List[str] = []
    if max_gust >= THRESHOLDS.high_gust_speed_kts:
        risks.append(f"前後{window_hours}H內有強陣風 ({max_gust:.1f} kts)")
    if max_wave >= THRESHOLDS.high_wave_sig_m:
        risks.append(f"前後{window_hours}H內有大浪 ({max_wave:.1f} m)")

    return RiskWindowResult(
        risks              = risks,
        max_wind_gust      = max_gust,
        max_wind_direction = max_gust_rec.wind_direction,
        max_wave_height    = max_wave,
        max_wave_period    = max_period,
        high_risk_hours    = high_risk_h,
        window_start       = window[0].time,
        window_end         = window[-1].time,
        dominant_wind_type = dom_wind_type,
    )


def _vessel_heading_default(self) -> float:
    """無 vessel 時的預設船艏向（用於時窗風型判斷）"""
    return 0.0


# ================= 氣象解析器 =================

class WeatherParser:
    """WNI 氣象資料解析器"""

    # 資料行格式：以 4 組 4 位數字開頭
    _LINE_PATTERN = re.compile(r"^\d{4}\s+\d{4}\s+\d{4}\s+\d{4}")
    _WIND_BLOCK_KEY = "WIND kts"

    def parse_content(
        self, content: str
    ) -> Tuple[str, List[WeatherRecord], List[str]]:
        """
        解析 WNI 氣象檔案內容。

        Returns:
            (port_name, records, warnings)

        Raises:
            ValueError: 找不到 WIND 資料區段或無法解析任何記錄。
        """
        lines = content.strip().split("\n")
        port_name = self._parse_port_name(lines)
        wind_start = self._find_wind_section(lines)
        records, warnings = self._parse_records(lines[wind_start:])

        if not records:
            raise ValueError("未成功解析任何氣象資料")

        logger.info("解析完成：港口=%s，記錄數=%d，警告數=%d", port_name, len(records), len(warnings))
        return port_name, records, warnings

    # ── 私有解析方法 ─────────────────────────────────────────

    @staticmethod
    def _parse_port_name(lines: List[str]) -> str:
        for line in lines:
            if "PORT NAME" in line.upper():
                return line.split(":", 1)[1].strip()
        return "Unknown Port"

    def _find_wind_section(self, lines: List[str]) -> int:
        for i, line in enumerate(lines):
            if self._WIND_BLOCK_KEY in line and "WAVE" in line:
                return i + 2
        raise ValueError("找不到 WIND 資料區段 (WIND kts)")

    def _parse_records(
        self, lines: List[str]
    ) -> Tuple[List[WeatherRecord], List[str]]:
        records: List[WeatherRecord] = []
        warnings: List[str] = []
        current_year = datetime.now().year
        prev_mmdd: Optional[str] = None

        for raw_line in lines:
            line = raw_line.strip()

            # 遇到分隔符號或空行即停止
            if not line or line[0] in ("*", "="):
                break

            if not self._LINE_PATTERN.match(line):
                continue

            try:
                record, current_year, prev_mmdd = self._parse_single_line(
                    line, current_year, prev_mmdd
                )
                records.append(record)
            except Exception as exc:
                warnings.append(f"解析失敗 [{line}]: {exc}")

        return records, warnings

    @staticmethod
    def _parse_single_line(
        line: str,
        current_year: int,
        prev_mmdd: Optional[str],
    ) -> Tuple[WeatherRecord, int, str]:
        parts = line.split()
        if len(parts) < 11:
            raise ValueError(f"欄位不足（需 ≥11，實際 {len(parts)}）")

        lct_date, lct_time = parts[2], parts[3]

        # 跨年偵測：前一筆是 12 月，當前是 1 月
        if (
            prev_mmdd
            and prev_mmdd > lct_date
            and prev_mmdd.startswith("12")
            and lct_date.startswith("01")
        ):
            current_year += 1

        dt = datetime.strptime(f"{current_year}{lct_date}{lct_time}", "%Y%m%d%H%M")

        def _safe_float(s: str) -> float:
            cleaned = s.replace("*", "")
            return float(cleaned) if cleaned else 0.0

        record = WeatherRecord(
            time=dt,
            wind_direction=parts[4],
            wind_speed=_safe_float(parts[5]),
            wind_gust=_safe_float(parts[6]),
            wave_direction=parts[7],
            wave_height=_safe_float(parts[8]),
            wave_max=_safe_float(parts[9]),
            wave_period=_safe_float(parts[10]),
        )
        return record, current_year, lct_date


# ================= 氣象分析器 =================

class WeatherAnalyzer:
    """
    氣象分析器 — 整合 OCIMF 計算與環境風險評估。

    主要流程：
        1. 物理計算（風力、纜繩抗力、拖船助力）
        2. 安全係數評估
        3. 風險分數計算
        4. 產生建議文字
        5. 回傳 AnalysisResult
    """

    def __init__(
        self,
        port_name: str,
        data: List[WeatherRecord],
        port_risk_level: int = 5,
    ):
        if not data:
            raise ValueError("氣象資料不可為空")
        if not hasattr(data[0], "wind_speed"):
            raise TypeError("❌ 資料格式錯誤：WeatherRecord 缺少 wind_speed 屬性")

        self.port_name = port_name
        self.data = sorted(data, key=lambda r: r.time)
        self.port_risk_level = max(1, min(10, port_risk_level))
        self.required_safety_factor = self._required_sf()

    # ── 公開查詢 ──────────────────────────────────────────────

    def time_range(self) -> Tuple[datetime, datetime]:
        return self.data[0].time, self.data[-1].time

    # ── 主分析入口 ────────────────────────────────────────────

    def analyze(self, vessel: VesselInfo) -> AnalysisResult:
        """
        執行完整靠泊風險分析。

        Raises:
            ValueError: 無氣象資料。
        """
        in_port = self._in_port_records(vessel)
        worst = max(in_port, key=lambda r: r.wind_gust)

        # ── 物理計算 ──
        heading    = self._vessel_heading(vessel)
        wind_res   = self._calc_wind_force(worst, vessel, heading)
        moor_res   = self._calc_mooring_restraint(vessel)
        tug_kN     = self._calc_tug_force(
            getattr(vessel, "tug_count", 2), vessel.tug_hp
        )

        # ── 安全係數 ──
        req_kN     = wind_res.transverse_force_N / 1000.0
        total_kN   = moor_res.transverse_restraint_kN + tug_kN
        sf, is_safe = self._evaluate_safety_factor(req_kN, total_kN)

        # ── 時間窗口風險 ──
        arr_window = self._check_window_risk(vessel.arrival_time)
        dep_window = self._check_window_risk(vessel.departure_time)

        # ── 建議文字 ──
        recommendations = self._build_recommendations(
            vessel, in_port, sf, is_safe, req_kN, total_kN,
            arr_window, dep_window,
        )

        # ── 風險分數與等級 ──
        risk_score = self._calc_risk_score(worst, sf, is_safe, arr_window, dep_window, vessel)
        risk_level = self._score_to_level(risk_score)

        # ── 組裝結果 ──
        return self._build_result(
            vessel, wind_res, moor_res, tug_kN,
            sf, is_safe, req_kN, total_kN,
            risk_score, risk_level,
            recommendations, worst,
        )

    # ── 私有：資料篩選 ────────────────────────────────────────

    def _in_port_records(self, vessel: VesselInfo) -> List[WeatherRecord]:
        records = [
            r for r in self.data
            if vessel.arrival_time <= r.time <= vessel.departure_time
        ]
        return records or self.data

    # ── 私有：物理計算 ────────────────────────────────────────

    def _vessel_heading(self, vessel: VesselInfo) -> float:
        """根據靠泊舷側計算船艏向"""
        side = str(vessel.berthing_side).lower().strip()
        if any(k in side for k in ("starboard", "右", "s", "stbd")):
            return (vessel.berth_direction + 180) % 360
        return vessel.berth_direction

    def _calc_wind_force(
        self,
        record: WeatherRecord,
        vessel: VesselInfo,
        heading: float,
    ) -> WindForceResult:
        """計算風力（OCIMF 方法）"""
        wind_ms  = knots_to_ms(record.wind_gust)
        wind_deg = compass_to_degrees(record.wind_direction)

        relative = (wind_deg - heading + 180) % 360 - 180
        abs_rel  = abs(relative)

        if 45 <= abs_rel <= 135:
            wind_type = "offshore" if relative > 0 else "onshore"
        else:
            wind_type = "parallel"

        drag_coef   = getattr(vessel, "wind_drag_coef", 1.0)
        total_N     = 0.5 * AIR_DENSITY * drag_coef * vessel.wind_area * wind_ms ** 2
        rad         = math.radians(abs_rel)

        return WindForceResult(
            total_force_N=total_N,
            transverse_force_N=total_N * abs(math.sin(rad)),
            longitudinal_force_N=total_N * abs(math.cos(rad)),
            wind_type=wind_type,
        )

    @staticmethod
    def _calc_mooring_restraint(vessel: VesselInfo) -> MooringRestraintResult:
        """
        計算纜繩抗力。
        WLL = MBL / safety_factor，再乘以各方向效率係數。
        """
        wll_N = vessel.mbl / vessel.safety_factor

        head_count   = vessel.bow_lines + vessel.stern_lines
        spring_count = vessel.bow_spring_lines + vessel.stern_spring_lines

        trans_N = (
            head_count   * wll_N * MOORING_EFF.head_transverse
            + spring_count * wll_N * MOORING_EFF.spring_transverse
        )
        long_N = (
            head_count   * wll_N * MOORING_EFF.head_longitudinal
            + spring_count * wll_N * MOORING_EFF.spring_longitudinal
        )

        return MooringRestraintResult(
            transverse_restraint_kN=trans_N / 1000.0,
            longitudinal_restraint_kN=long_N / 1000.0,
            total_capacity_N=trans_N,
            wll_single_N=wll_N,
        )

    @staticmethod
    def _calc_tug_force(tug_count: int, tug_hp: float) -> float:
        """計算拖船助力 (kN)"""
        if tug_count <= 0:
            return 0.0
        bollard_ton = (tug_hp / 100.0) * HP_TO_BOLLARD_PULL_PER_100HP
        return tug_count * bollard_ton * GRAVITY

    def _evaluate_safety_factor(
        self, req_kN: float, total_kN: float
    ) -> Tuple[float, bool]:
        """計算安全係數，回傳 (sf, is_safe)"""
        if req_kN < 0.01:
            return 99.9, True
        sf = total_kN / req_kN
        return sf, sf >= self.required_safety_factor

    # ── 私有：風險判斷 ────────────────────────────────────────

    def _required_sf(self) -> float:
        """依港口風險等級查表取得所需安全係數"""
        for threshold, sf in _PORT_RISK_SF_MAP:
            if self.port_risk_level <= threshold:
                return sf
        return 2.0

    @staticmethod
    def _score_to_level(score: float) -> str:
        """風險分數 → 風險等級字串"""
        for threshold, level in _RISK_SCORE_LEVELS:
            if score >= threshold:
                return level
        return "low"

    def _is_night(self, dt: datetime) -> bool:
        h = dt.hour
        s, e = THRESHOLDS.night_start_hour, THRESHOLDS.night_end_hour
        return h >= s or h < e if s > e else s <= h < e

    def _is_high_risk_weather(self, r: WeatherRecord) -> bool:
        return (
            r.wind_speed  >= THRESHOLDS.high_wind_speed_kts
            or r.wind_gust >= THRESHOLDS.high_gust_speed_kts
            or r.wave_height >= THRESHOLDS.high_wave_sig_m
        )

    def _check_window_risk(
        self, target: datetime, window_hours: int = 2
    ) -> RiskWindowResult:
        """檢查目標時間前後 window_hours 小時內的風險"""
        start = target - timedelta(hours=window_hours)
        end   = target + timedelta(hours=window_hours)
        window = [r for r in self.data if start <= r.time <= end]

        if not window:
            return RiskWindowResult(risks=["無該時段氣象資料"])

        risks: List[str] = []
        max_gust = max(r.wind_gust for r in window)
        max_wave = max(r.wave_height for r in window)

        if max_gust >= THRESHOLDS.high_gust_speed_kts:
            risks.append(f"前後{window_hours}H內有強陣風 ({max_gust:.1f} kts)")
        if max_wave >= THRESHOLDS.high_wave_sig_m:
            risks.append(f"前後{window_hours}H內有大浪 ({max_wave:.1f} m)")

        return RiskWindowResult(risks=risks)

    # ── 私有：風險分數計算 ────────────────────────────────────

    def _calc_risk_score(
        self,
        worst: WeatherRecord,
        sf: float,
        is_safe: bool,
        arr_window: RiskWindowResult,
        dep_window: RiskWindowResult,
        vessel: VesselInfo,
    ) -> float:
        score = 0.0

        # 風速風險（0–40 分）
        if worst.wind_gust >= THRESHOLDS.extreme_gust_kts:
            score += SCORE_WEIGHTS.wind
        elif worst.wind_gust >= THRESHOLDS.high_gust_speed_kts:
            score += 25
        elif worst.wind_gust >= THRESHOLDS.high_wind_speed_kts:
            score += 15

        # 浪高風險（0–30 分）
        if worst.wave_height >= THRESHOLDS.very_high_wave_sig_m:
            score += SCORE_WEIGHTS.wave
        elif worst.wave_height >= THRESHOLDS.high_wave_sig_m:
            score += 20
        elif worst.wave_height >= THRESHOLDS.moderate_wave_sig_m:
            score += 10

        # 安全係數風險（0–20 分）
        if not is_safe:
            score += SCORE_WEIGHTS.safety_factor if sf < 1.0 else 15

        # 時間窗口風險（0–10 分）
        if arr_window.has_risk:
            score += 5
        if dep_window.has_risk:
            score += 5

        # 夜間作業（0–5 分）
        if self._is_night(vessel.arrival_time) or self._is_night(vessel.departure_time):
            score += SCORE_WEIGHTS.night_ops

        # 港口風險加成（0–5 分）
        if self.port_risk_level >= 8:
            score += SCORE_WEIGHTS.port_level
        elif self.port_risk_level >= 6:
            score += 3

        return min(100.0, score)

    # ── 私有：建議文字生成 ────────────────────────────────────

    def _build_recommendations(
        self,
        vessel: VesselInfo,
        in_port: List[WeatherRecord],
        sf: float,
        is_safe: bool,
        req_kN: float,
        total_kN: float,
        arr_window: RiskWindowResult,
        dep_window: RiskWindowResult,
    ) -> List[str]:
        recs: List[str] = []

        # 安全係數
        if is_safe:
            recs.append(
                f"✅ 物理安全係數充足 (SF={sf:.2f} ≥ {self.required_safety_factor:.2f})"
            )
        else:
            deficit_kN = req_kN * self.required_safety_factor - total_kN
            recs.append(
                f"⚠️ 物理安全係數不足 (SF={sf:.2f} < {self.required_safety_factor:.2f})"
            )
            one_tug_kN = self._calc_tug_force(1, vessel.tug_hp)
            if one_tug_kN > 0:
                add_tugs = math.ceil(deficit_kN / one_tug_kN)
                recs.append(
                    f"🚤 建議增加 {add_tugs} 艘拖船以補足抗力 (缺口: {deficit_kN:.1f} kN)"
                )

        # 靠泊時窗
        if self._is_night(vessel.arrival_time):
            recs.append("🌙 注意：夜間靠泊作業，請加強照明與通訊")
        if arr_window.has_risk:
            recs.append(f"🚨 靠泊警示: {'; '.join(arr_window.risks)}")

        # 離泊時窗
        if self._is_night(vessel.departure_time):
            recs.append("🌙 注意：夜間離泊作業")
        if dep_window.has_risk:
            recs.append(f"🚨 離泊警示: {'; '.join(dep_window.risks)}")

        # 高風險統計
        high_wind_hours = sum(
            1 for r in in_port if r.wind_gust >= THRESHOLDS.high_gust_speed_kts
        )
        if high_wind_hours > 0:
            recs.append(
                f"⚠️ 在港期間有 {high_wind_hours} 小時陣風超過 "
                f"{THRESHOLDS.high_gust_speed_kts:.0f} kts，請加強巡艙"
            )

        return recs

    # ── 私有：組裝 AnalysisResult ─────────────────────────────

    def _build_result(
        self,
        vessel: VesselInfo,
        wind_res: WindForceResult,
        moor_res: MooringRestraintResult,
        tug_kN: float,
        sf: float,
        is_safe: bool,
        req_kN: float,
        total_kN: float,
        risk_score: float,
        risk_level: str,
        recommendations: List[str],
        worst: WeatherRecord,
    ) -> AnalysisResult:
        status_code = "OK" if is_safe else ("CRITICAL" if sf < 1.0 else "WARNING")
        utilization = (
            (req_kN / moor_res.transverse_restraint_kN * 100)
            if moor_res.transverse_restraint_kN > 0
            else 0.0
        )
        rec_text = "正常" if is_safe else "需增強"

        bow_status = MooringStatus(
            status=status_code,
            current_lines=vessel.bow_lines + vessel.bow_spring_lines,
            required_lines=vessel.bow_lines + vessel.bow_spring_lines,
            utilization=utilization,
            recommendation_text=rec_text,
        )
        stern_status = MooringStatus(
            status=status_code,
            current_lines=vessel.stern_lines + vessel.stern_spring_lines,
            required_lines=vessel.stern_lines + vessel.stern_spring_lines,
            utilization=utilization,
            recommendation_text=rec_text,
        )

        # 計算需補充的拖船數
        one_tug_kN = self._calc_tug_force(1, vessel.tug_hp)
        deficit_kN = max(0.0, req_kN * self.required_safety_factor - total_kN)
        add_tugs   = math.ceil(deficit_kN / one_tug_kN) if (not is_safe and one_tug_kN > 0) else 0
        base_tugs  = getattr(vessel, "tug_count", 2)

        return AnalysisResult(
            risk_score=risk_score,
            risk_level=risk_level,
            mitigated_risk_score=risk_score * (0.8 if is_safe else 0.9),
            mooring_split=MooringSplit(bow=bow_status, stern=stern_status),
            tug_recommendation={
                "base_tug_count":       base_tugs,
                "final_tug_count":      base_tugs + add_tugs,
                "adequacy":             is_safe,
                "enforcement_reasons":  [
                    r for r in recommendations if "警示" in r or "不足" in r
                ],
            },
            wind_force_summary={
                "max_gust_force_N":    wind_res.total_force_N,
                "max_trans_force_N":   wind_res.transverse_force_N,
                "max_long_force_N":    wind_res.longitudinal_force_N,
                "safety_factor":       sf,
                "wind_type":           wind_res.wind_type,
                "mooring_capacity_N":  moor_res.total_capacity_N,
                "total_restraint_kN":  total_kN,
                "required_force_kN":   req_kN,
                "max_gust_record": {
                    "time":        worst.time,
                    "wind_gust":   worst.wind_gust,
                    "wave_height": worst.wave_height,
                },
            },
            recommendations=recommendations,
        )


# ================= 便利函式 =================

def create_analyzer(
    port_name: str,
    data: List[WeatherRecord],
    port_risk_level: int = 5,
) -> WeatherAnalyzer:
    """建立 WeatherAnalyzer 實例的工廠函式"""
    return WeatherAnalyzer(port_name, data, port_risk_level)
