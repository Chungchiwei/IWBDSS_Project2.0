# font_loader.py
"""
中文字體載入模組

確保 Matplotlib 可以正確顯示中文字元。
根據作業系統自動選擇合適的字體，並提供除錯工具。

使用方式：
    from font_loader import ensure_chinese_font
    ensure_chinese_font()
"""
from __future__ import annotations

import logging
import platform
from typing import Dict, List, Optional

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ================= 字體候選清單 =================

# 各作業系統的字體優先順序（由高到低）
_FONT_CANDIDATES: Dict[str, List[str]] = {
    "Windows": [
        "Microsoft JhengHei",   # 微軟正黑體（繁體中文首選）
        "Microsoft YaHei",      # 微軟雅黑（簡體中文）
        "SimHei",               # 黑體
        "KaiTi",                # 標楷體
    ],
    "Darwin": [                 # macOS
        "PingFang TC",          # 蘋方（繁體）
        "Heiti TC",             # 黑體（繁體）
        "Arial Unicode MS",     # 萬國碼備援
    ],
    "Linux": [
        "Noto Sans CJK TC",     # Google Noto（繁體）
        "WenQuanYi Micro Hei",  # 文泉驛微米黑
        "AR PL UMing TW",       # 文鼎明體
    ],
}

# 所有平台的最終備援字體
_FALLBACK_FONT = "DejaVu Sans"

# 中文字體關鍵字（用於 list_available_fonts 過濾）
_CJK_KEYWORDS = frozenset(
    ["chinese", "cjk", "hei", "ming", "kai", "song", "noto", "pingfang", "jheng"]
)


# ================= 核心函式 =================

def ensure_chinese_font() -> str:
    """
    設定 Matplotlib 中文字體。
    根據作業系統自動選擇合適的字體，
    優先使用繁體中文字體，失敗時退回備援字體。
    Returns:
        實際套用的字體名稱。
    """
    # ✅ 正確清除字體快取的方式（相容 Matplotlib 3.7+）
    # 強制重建 FontManager，讓系統新安裝的字型（如 fonts-noto-cjk）被偵測到
    try:
        fm._fmcache = None
        fm.fontManager = fm.FontManager()
        logger.info("🔄 Matplotlib 字體快取已強制重建")
    except Exception as e:
        logger.warning("⚠️ 字體快取重建失敗（%s），繼續使用現有快取", e)

    system = platform.system()
    candidates = _FONT_CANDIDATES.get(system, [])

    # 重建後再取得字體清單，確保讀到最新狀態
    available = _get_available_font_names()

    logger.info("🖥️ 作業系統：%s，偵測到 %d 個字體", system, len(available))
    logger.info("🔍 候選字體清單：%s", candidates)

    for font_name in candidates:
        if font_name in available:
            _apply_font(font_name)
            logger.info("✅ 已載入中文字體：%s（%s）", font_name, system)
            return font_name

    # 所有候選字體均不可用，退回備援
    _apply_font(_FALLBACK_FONT)
    logger.warning(
        "⚠️ 系統 %s 上找不到任何中文字體候選，"
        "已退回使用 %s，中文字元可能無法正常顯示。",
        system, _FALLBACK_FONT,
    )
    return _FALLBACK_FONT





def list_available_fonts(cjk_only: bool = True) -> List[str]:
    """
    列出系統中可用的字體。

    Args:
        cjk_only: 若為 True，僅回傳疑似 CJK 字體（預設）；
                  若為 False，回傳全部字體。

    Returns:
        字體名稱列表（已排序）。
    """
    all_fonts = sorted({f.name for f in fm.fontManager.ttflist})

    if not cjk_only:
        return all_fonts

    filtered = [
        name for name in all_fonts
        if any(kw in name.lower() for kw in _CJK_KEYWORDS)
    ]

    if filtered:
        logger.info("找到 %d 個 CJK 相關字體：%s", len(filtered), filtered)
    else:
        logger.warning("未找到任何 CJK 相關字體，請確認系統字體安裝狀況。")

    return filtered


def get_font_info() -> Dict[str, object]:
    """
    回傳目前 Matplotlib 字體設定摘要（用於除錯）。

    Returns:
        包含 system、current_fonts、unicode_minus 的字典。
    """
    return {
        "system":         platform.system(),
        "current_fonts":  plt.rcParams.get("font.sans-serif", []),
        "unicode_minus":  plt.rcParams.get("axes.unicode_minus", False),
        "available_cjk":  list_available_fonts(cjk_only=True),
    }


# ================= 私有輔助函式 =================

def _apply_font(font_name: str) -> None:
    """套用字體至 Matplotlib 全域設定"""
    # ✅ 加入 "DejaVu Sans" 作為備援，避免字體完全失效
    plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False



def _get_available_font_names() -> frozenset:
    """
    取得系統中所有已安裝字體的名稱集合。

    使用 frozenset 加速 `in` 查詢（O(1) vs list 的 O(n)）。
    """
    return frozenset(f.name for f in fm.fontManager.ttflist)


# ================= 模組直接執行（除錯用）=================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    logger.info("=== 字體載入測試 ===")
    applied = ensure_chinese_font()
    logger.info("套用字體：%s", applied)

    logger.info("\n=== CJK 字體清單 ===")
    cjk_fonts = list_available_fonts(cjk_only=True)
    for f in cjk_fonts:
        logger.info("  - %s", f)

    logger.info("\n=== 目前設定摘要 ===")
    info = get_font_info()
    for key, val in info.items():
        logger.info("  %s: %s", key, val)
