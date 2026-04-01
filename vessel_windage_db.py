# vessel_windage_db.py
"""
船型受風面積查找資料庫
資料來源：受風面積調查回船總表.xlsx（完整版）

船型分類：
  A Type 現代 (A_HYUNDAI) — WAN HAI A01 ~ A07（現代重工建造）
  A Type 三星 (A_SAMSUNG) — WAN HAI A08 ~ A20（三星重工建造）
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ===============================================================
# 資料結構
# ===============================================================

@dataclass
class WindageRecord:
    """單筆受風面積紀錄"""
    vessel_name:  str
    port:         str
    draft_fwd:    float   # 船艏吃水 (m)
    draft_aft:    float   # 船艉吃水 (m)
    windage_area: float   # 受風面積 (m²)
    dep_weight:   float   # 離港重量 (MT)
    dep_teus:     int     # 離港 TEU 數

    @property
    def avg_draft(self) -> float:
        return (self.draft_fwd + self.draft_aft) / 2.0


@dataclass
class WindageLookupResult:
    """查找結果容器"""
    windage_area:   float
    matched_record: WindageRecord
    distance:       float
    method:         str
    candidates:     List[WindageRecord] = field(default_factory=list)


# ===============================================================
# 子型分類函式
# ===============================================================

def _classify_a_subtype(vessel_name: str) -> str:
    """
    從船名萃取 A 型子型編號。
    A01–A07 → 'A_HYUNDAI'（現代）
    A08–A20 → 'A_SAMSUNG'（三星）
    """
    match = re.search(r"A(\d+)", vessel_name.upper())
    if match:
        num = int(match.group(1))
        if 1 <= num <= 7:
            return "A_HYUNDAI"
        if 8 <= num <= 20:
            return "A_SAMSUNG"
    return "UNKNOWN"


# ===============================================================
# 原始資料（來源：受風面積調查回船總表.xlsx 完整版）
# 欄位順序：vessel_name, port, draft_fwd, draft_aft,
#           windage_area, dep_weight, dep_teus
# ===============================================================

_RAW_DATA: List[Tuple] = [

    # ────────────────────────────────────────────────────────
    # WAN HAI A01
    # ────────────────────────────────────────────────────────
    ("Wan Hai A01", "CNXMN",  13.5, 14.6, 11036.7,  81514.0, 10626),
    ("Wan Hai A01", "TWTPE",  12.3, 13.3, 11059.6,  89194.0,  9528),
    ("Wan Hai A01", "VNTCT",  10.0, 12.5, 10992.5,  49963.0,  6740),
    ("Wan Hai A01", "CNCWN",   8.6,  7.7,  8558.4,  11790.0,   918),
    ("Wan Hai A01", "ECGYE",   9.8,  9.9, 10341.8,  43879.0,  8289),
    ("Wan Hai A01", "ECGYE",  10.6, 11.1, 11024.0,  54103.0,  8196),
    ("Wan Hai A01", "MXLZC",  11.8, 12.7, 10130.8,  72387.0,  8092),
    ("Wan Hai A01", "MXZLO",  10.8, 11.6, 11792.5,  61629.0, 10960),
    ("Wan Hai A01", "MXZLO",  10.3, 11.2, 11621.4,  53210.0, 10623),

    # ────────────────────────────────────────────────────────
    # WAN HAI A02
    # ────────────────────────────────────────────────────────
    ("Wan Hai A02", "CNSKU",  12.1, 13.0, 10435.6,  72860.1,  5831),
    ("Wan Hai A02", "CNSKU",   7.8, 10.2,  8604.8,  26655.5,  1381),
    ("Wan Hai A02", "GTPRQ",  11.2, 11.9, 11002.5,  59371.7,  5426),
    ("Wan Hai A02", "HKHKG",  10.9, 11.7, 10756.0,  51257.9,  4949),
    ("Wan Hai A02", "MXESE",  11.4, 11.6, 13239.8,  63939.0, 10788),
    ("Wan Hai A02", "MXLZC",  10.2, 12.1, 11061.1,  92019.4,  9080),
    ("Wan Hai A02", "MXZLO",  12.0, 13.1, 11061.5,  69524.0,  6743),

    # ────────────────────────────────────────────────────────
    # WAN HAI A03
    # ────────────────────────────────────────────────────────
    ("WAN HAI A03", "SGSIN",  15.3, 14.5, 10833.5, 100599.0, 10616),
    ("WAN HAI A03", "USCHS",  10.9, 11.7, 11791.3,  50749.1, 10622),
    ("WAN HAI A03", "USSAV",   9.2, 11.0, 12116.7,  37707.8, 10864),
    ("WAN HAI A03", "CNNGB",  12.7, 13.5, 11311.6,  73403.6,  9508),
    ("WAN HAI A03", "CNSHA",  14.1, 14.9, 11257.6,  90042.6, 11554),
    ("WAN HAI A03", "CNSKU",   8.5, 12.9, 11288.4,  43980.8,  5557),
    ("WAN HAI A03", "USLAX",  10.8, 11.0, 12228.4,  45433.5, 11250),

    # ────────────────────────────────────────────────────────
    # WAN HAI A05
    # ────────────────────────────────────────────────────────
    ("Wan Hai A05", "CNNGB",  12.3, 14.4, 11269.5,  77595.0,  9557),
    ("Wan Hai A05", "CNNGB",  10.6, 15.2, 11277.9,  75089.0,  9420),
    ("Wan Hai A05", "CNSHA",  13.2, 15.1, 11145.8,  94542.0, 11576),
    ("Wan Hai A05", "CNSHA",  13.7, 15.2, 11090.5,   9436.0, 11549),
    ("Wan Hai A05", "CNSKU",  12.9, 13.8, 11062.8,  74862.0,  9537),
    ("Wan Hai A05", "CNSKU",  10.6, 12.6, 10760.3,  60113.0,  7316),
    ("Wan Hai A05", "CNXMN",  14.0, 14.2, 11194.7,  84793.0, 10574),
    ("Wan Hai A05", "USLAX",  10.1, 10.5, 11937.2,  37949.0,  9783),
    ("Wan Hai A05", "USOAK",  10.6, 11.2, 11724.6,  45688.0, 10547),
    ("Wan Hai A05", "VNHPH",   7.4, 11.8, 10064.6,  32377.0,  5182),
    ("Wan Hai A05", "VNTCT",  11.8, 12.3, 10824.8,  55159.1,  6824),

    # ────────────────────────────────────────────────────────
    # WAN HAI A06
    # ────────────────────────────────────────────────────────
    ("WAN HAI A06", "TWTPE",  14.7, 14.9, 11273.8,  95924.0, 11564),
    ("WAN HAI A06", "CNNGB",  13.9, 13.6, 10161.3,  85908.0,  7825),
    ("WAN HAI A06", "CNNGB",  13.6, 13.8, 10183.2,  85909.0,  7825),
    ("WAN HAI A06", "CNNSS",  10.5, 10.8, 10767.3,  41026.0,  5321),
    ("WAN HAI A06", "ECGYE",  10.3, 12.0, 10691.9,  50682.0,  8074),
    ("WAN HAI A06", "GTPRQ",  11.3, 12.8, 11077.5,  69197.0,  8465),
    ("WAN HAI A06", "MXLZC",  12.3, 14.2, 11305.4,  78507.0,  9136),
    ("WAN HAI A06", "PECLL",  10.3, 12.3, 11103.7,  60675.0,  8885),
    ("WAN HAI A06", "TWKHH",  10.1, 10.7, 12540.8,  43557.0, 10094),

    # ────────────────────────────────────────────────────────
    # WAN HAI A08（三星）
    # ────────────────────────────────────────────────────────
    ("Wan Hai A08", "CNSHA",   9.3, 10.2, 10761.0,  45310.0,  8232),
    ("Wan Hai A08", "TWTPE",  10.0, 12.8, 10761.0,  59881.0,  8624),
    ("Wan Hai A08", "SGSIN",  13.8, 14.9,  9934.0, 100299.0, 10546),
    ("Wan Hai A08", "LKCMB",  13.0, 14.8, 10084.0,  96562.0, 10524),
    ("Wan Hai A08", "CNYTN",  10.7, 13.7, 10187.0,  70875.0,  9715),
    ("Wan Hai A08", "VNTCT",  13.4, 13.9, 10202.0,  90002.0,  9667),
    ("Wan Hai A08", "CNSHK",  11.7, 13.4, 10369.0,  69690.1,  8382),
    ("Wan Hai A08", "CNNGB",   9.9, 11.7, 10382.0,  50291.0,  7878),
    ("Wan Hai A08", "CNYTN",  11.2, 13.1, 10498.0,  63939.5,  9895),
    ("Wan Hai A08", "USNYC",  12.0, 12.2, 10604.0,  67656.7, 10252),
    ("Wan Hai A08", "TWTPE",  11.0, 11.1, 10635.0,  54407.7,  8686),
    ("Wan Hai A08", "CNYTN",   8.4, 12.5, 10667.4,  68142.5,  8715),
    ("Wan Hai A08", "CNNGB",   6.4, 10.8, 10683.3,  44668.5,  7159),
    ("Wan Hai A08", "UCORF",  11.4, 12.1, 10716.0,  62843.2, 10389),
    ("Wan Hai A08", "CNNGB",  10.1, 11.1, 10852.0,  48330.0,  8172),
    ("Wan Hai A08", "USSAV",  10.7, 10.8, 11051.0,  50831.2, 10636),
    ("Wan Hai A08", "USCHS",   9.3, 11.0, 11282.0,  42574.3, 10628),
    ("Wan Hai A08", "CNSHA",   8.9, 10.7, 11288.0,  42778.0,  9732),
    ("Wan Hai A08", "CNSHA",   4.7,  6.3,  8972.0,  17753.0,  1793),
    ("Wan Hai A08", "CNYTN",  10.4, 10.6,  9867.1,  67941.0,  7419),
    ("Wan Hai A08", "CNNGB",   7.2, 10.7,  9895.6,  49305.6,  6885),
    ("Wan Hai A08", "CNNGB",   6.4,  8.8, 10017.7,  32638.0,  3492),
    ("Wan Hai A08", "CNYTN",   7.2, 10.8, 10258.3,  49918.8,  5366),
    ("Wan Hai A08", "CNYTN",   8.4, 12.5, 10667.4,  68142.5,  8715),
    ("Wan Hai A08", "CNNGB",   6.4, 10.8, 10683.3,  44668.5,  7159),
    ("Wan Hai A08", "CNSHA",   5.6, 11.1, 12376.0,  42318.7,  7451),
    ("Wan Hai A08", "CNSHA",   6.4,  9.8, 12587.5,  39676.4,  8635),
    ("Wan Hai A08", "CNYTN",  10.1, 10.8, 11552.3,  67691.4,  8427),
    ("Wan Hai A08", "CNYTN",  10.2,  9.9, 11689.3,  62089.9,  7948),
    ("Wan Hai A08", "CNNGB",   7.2,  9.7, 11692.5,  43830.4,  7034),
    ("Wan Hai A08", "CNSHA",   6.5,  9.4, 11705.3,  38015.2,  6402),
    ("Wan Hai A08", "CNNGB",   8.3,  9.8, 11293.8,  51284.7,  7221),
    ("Wan Hai A08", "CNNGB",   7.4, 10.1, 11368.0,  46973.9,  6104),
    ("Wan Hai A08", "CNYTN",   8.8, 11.2, 11463.8,  61808.9,  7618),
    ("Wan Hai A08", "CNSHA",  14.1, 14.9, 11257.6,  90042.6, 11554),
    ("Wan Hai A08", "CNSKU",   8.5, 12.9, 11288.4,  43980.8,  5557),
    ("Wan Hai A08", "CNNGB",  12.7, 13.5, 11311.6,  73403.6,  9508),
    ("Wan Hai A08", "CNSHA",  15.2, 14.9,  9934.0, 100299.0, 10546),
    ("Wan Hai A08", "CNSHA",  13.0, 14.8, 10084.0,  96562.0, 10524),
    ("Wan Hai A08", "CNSHA",  11.4, 12.1, 10716.0,  62843.2, 10389),
    ("Wan Hai A08", "CNSHA",  10.0, 12.8, 10761.0,  59881.0,  8624),
    ("Wan Hai A08", "CNYTN",  11.2, 13.8, 11062.8,  74862.0,  9537),
    ("Wan Hai A08", "CNXMN",  14.0, 14.2, 11194.7,  84793.0, 10574),
    ("Wan Hai A08", "CNSKU",  12.9, 13.8, 11062.8,  74862.0,  9537),
    ("Wan Hai A08", "CNSKU",  10.6, 12.6, 10760.3,  60113.0,  7316),
    ("Wan Hai A08", "USNYC",  10.8, 11.6, 11792.5,  61629.0, 10960),
    ("Wan Hai A08", "USNYC",  10.3, 11.2, 11621.4,  53210.0, 10623),
    ("Wan Hai A08", "USLAX",  10.8, 11.0, 12228.4,  45433.5, 11250),
    ("Wan Hai A08", "USLAX",  10.1, 10.5, 11937.2,  37949.0,  9783),
    ("Wan Hai A08", "USOAK",  10.6, 11.2, 11724.6,  45688.0, 10547),
    ("Wan Hai A08", "USCHS",  10.9, 11.7, 11791.3,  50749.1, 10622),
    ("Wan Hai A08", "USSAV",   9.2, 11.0, 12116.7,  37707.8, 10864),
]


# ===============================================================
# 建立分類資料集
# ===============================================================

def _build_db() -> Dict[str, List[WindageRecord]]:
    db: Dict[str, List[WindageRecord]] = {
        "A_HYUNDAI": [],
        "A_SAMSUNG": [],
    }
    seen: set = set()  # 去除完全重複的紀錄

    for r in _RAW_DATA:
        subtype = _classify_a_subtype(r[0])
        if subtype not in db:
            continue

        key = (r[0], r[1], r[2], r[3])  # vessel + port + 兩吃水
        if key in seen:
            continue
        seen.add(key)

        db[subtype].append(
            WindageRecord(
                vessel_name  = r[0],
                port         = r[1],
                draft_fwd    = r[2],
                draft_aft    = r[3],
                windage_area = r[4],
                dep_weight   = r[5],
                dep_teus     = r[6],
            )
        )
    return db


VESSEL_TYPE_DB: Dict[str, List[WindageRecord]] = _build_db()


# ===============================================================
# 顯示名稱對照
# ===============================================================

VESSEL_TYPE_DISPLAY: Dict[str, str] = {
    "A_HYUNDAI": "A Type 現代 (A01–A07)",
    "A_SAMSUNG": "A Type 三星 (A08–A20)",
}

VESSEL_TYPE_KEY_MAP: Dict[str, str] = {
    v: k for k, v in VESSEL_TYPE_DISPLAY.items()
}


# ===============================================================
# 查找引擎
# ===============================================================

def lookup_windage_area(
    vessel_type: str,
    draft_fwd:   float,
    draft_aft:   float,
    top_n:       int = 3,
) -> Optional[WindageLookupResult]:
    """
    依船型與前後吃水查找最接近的受風面積。

    距離公式（加權歐氏距離）：
        d = sqrt( (Δfwd × 0.5)² + (Δaft × 0.5)² + (Δavg × 1.0)² )

    Args:
        vessel_type : DB key，"A_HYUNDAI" 或 "A_SAMSUNG"
        draft_fwd   : 船艏吃水 (m)
        draft_aft   : 船艉吃水 (m)
        top_n       : 回傳候選筆數（預設 3）

    Returns:
        WindageLookupResult；找不到船型時回傳 None
    """
    records = VESSEL_TYPE_DB.get(vessel_type)
    if not records:
        return None

    avg_input = (draft_fwd + draft_aft) / 2.0

    def _dist(rec: WindageRecord) -> float:
        d_fwd = (rec.draft_fwd - draft_fwd) * 0.5
        d_aft = (rec.draft_aft - draft_aft) * 0.5
        d_avg = (rec.avg_draft - avg_input) * 1.0
        return math.sqrt(d_fwd**2 + d_aft**2 + d_avg**2)

    scored = sorted(records, key=_dist)
    best   = scored[0]
    dist   = _dist(best)

    method = (
        f"精確匹配（吃水差 {dist:.2f} m）"
        if dist < 0.3
        else (
            f"最近鄰近似（吃水差 {dist:.2f} m，"
            f"參考 {best.vessel_name} @ {best.port} | "
            f"艏 {best.draft_fwd}m / 艉 {best.draft_aft}m）"
        )
    )

    return WindageLookupResult(
        windage_area   = best.windage_area,
        matched_record = best,
        distance       = dist,
        method         = method,
        candidates     = scored[:top_n],
    )


def get_windage_stats(vessel_type: str) -> Dict[str, float]:
    """回傳該船型受風面積的統計摘要（min / max / mean / count）"""
    records = VESSEL_TYPE_DB.get(vessel_type, [])
    if not records:
        return {}
    areas = [r.windage_area for r in records]
    return {
        "min":   round(min(areas),  1),
        "max":   round(max(areas),  1),
        "mean":  round(sum(areas) / len(areas), 1),
        "count": len(areas),
    }
