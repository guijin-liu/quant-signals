"""
板块轮动特征
- 板块强度排名
- 个股所属板块的相对位置
- 板块资金流向
"""

import logging
import numpy as np
import pandas as pd
from config import STOCK_POOL, SECTORS

logger = logging.getLogger(__name__)


def extract_sector_features(
    sector_ranks: dict,
    stock_code: str,
) -> dict:
    """
    提取个股相关的板块特征
    sector_ranks: find_stock_sector_rank 的输出格式
    """
    stock_info = STOCK_POOL.get(stock_code, {})
    sectors = [stock_info.get("sector", ""), stock_info.get("sub_sector", "")]
    sectors = [s for s in sectors if s]

    feats = {
        "sector_rank": 999,
        "sector_pct": 0.0,
        "sector_rank_pct": 0.0,       # 排名百分比（越小越靠前）
        "sector_up_ratio": 0.0,        # 板块内上涨比例
        "sector_turnover": 0.0,
        "sector_is_leading": 0,        # 是否领涨板块
        "sector_strength_score": 0.0,  # 板块综合强度
    }

    sector_data = []
    for s in sectors:
        if s in sector_ranks:
            sector_data.append(sector_ranks[s])

    if not sector_data:
        return feats

    # 取最好的板块排名（个股可能属于多个板块）
    best = min(sector_data, key=lambda x: x.get("rank", 999))
    avg_pct = np.mean([d.get("pct_change", 0) for d in sector_data])

    feats["sector_rank"] = best.get("rank", 999)
    feats["sector_pct"] = round(avg_pct, 4)

    total_up = sum(d.get("up_count", 0) for d in sector_data)
    total_down = sum(d.get("down_count", 0) for d in sector_data)
    feats["sector_up_ratio"] = round(total_up / max(total_up + total_down, 1), 3)

    feats["sector_turnover"] = round(np.mean([d.get("turnover_rate", 0) for d in sector_data]), 4)
    feats["sector_is_leading"] = 1 if best.get("rank", 999) <= 5 else 0

    # 综合强度：排名分 + 涨幅分
    rank_score = max(0, 1 - feats["sector_rank"] / 100)
    pct_score = min(1, max(-1, avg_pct / 5))
    feats["sector_strength_score"] = round(0.6 * rank_score + 0.4 * pct_score, 4)

    return feats


def compute_sector_momentum(sector_rank_df: pd.DataFrame, lookback_days: int = 5) -> dict:
    """
    计算板块动量：连续N日排名变化
    需要历史板块排名数据（如果可用）
    """
    if sector_rank_df is None or sector_rank_df.empty:
        return {}

    result = {}
    if "sector_name" not in sector_rank_df.columns or "pct_change" not in sector_rank_df.columns:
        return result

    # 当日数据
    for _, row in sector_rank_df.head(30).iterrows():
        name = row.get("sector_name", "")
        if not name:
            continue
        pct = row.get("pct_change", 0)
        rank = row.get("rank", 999)
        result[name] = {
            "pct_change": pct,
            "rank": rank,
            "is_hot": rank <= 10,
        }

    return result
