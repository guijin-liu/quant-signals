"""
资金流向特征
- 主力资金净流入趋势
- 北向资金信号
- 板块资金轮动
"""

import logging
import numpy as np
import pandas as pd
from config import STOCK_CODES, STOCK_POOL

logger = logging.getLogger(__name__)


def extract_money_flow_features(
    stock_flow_df: pd.DataFrame,
    north_flow_df: pd.DataFrame,
    market_flow_df: pd.DataFrame,
    sector_flow_df: pd.DataFrame,
    stock_code: str,
) -> dict:
    """
    提取资金流向综合特征
    """
    feats = {
        # 个股资金流向
        "stock_main_net_3d": 0.0,
        "stock_main_net_5d": 0.0,
        "stock_main_pct_avg": 0.0,
        "stock_flow_trend": 0,          # 1=加速流入, 0=正常, -1=流出
        "stock_super_large_net_5d": 0.0,
        "stock_large_net_5d": 0.0,
        # 北向资金
        "north_flow_1d": 0.0,
        "north_flow_3d": 0.0,
        "north_flow_trend": 0,
        # 市场整体
        "market_main_net": 0.0,
        "market_main_pct": 0.0,
        # 板块资金
        "sector_main_net": 0.0,
        "sector_main_pct": 0.0,
        # 综合
        "capital_score": 0.0,
    }

    # === 个股资金流向 ===
    if stock_flow_df is not None and not stock_flow_df.empty:
        df = stock_flow_df.sort_values("datetime")
        net_col = "main_net" if "main_net" in df.columns else None

        if net_col:
            feats["stock_main_net_3d"] = round(df[net_col].tail(3).sum(), 2) if len(df) >= 3 else round(df[net_col].sum(), 2)
            feats["stock_main_net_5d"] = round(df[net_col].tail(5).sum(), 2) if len(df) >= 5 else round(df[net_col].sum(), 2)

        if "main_net_pct" in df.columns:
            feats["stock_main_pct_avg"] = round(df["main_net_pct"].tail(5).mean(), 2)

        # 趋势判断
        net_3d = feats["stock_main_net_3d"]
        net_5d = feats["stock_main_net_5d"]
        if net_5d > 0 and net_3d > abs(net_5d) * 0.3:
            feats["stock_flow_trend"] = 2  # 加速流入
        elif net_5d > 0 and net_3d > 0:
            feats["stock_flow_trend"] = 1  # 持续流入
        elif net_5d < 0 and net_3d < -abs(net_5d) * 0.3:
            feats["stock_flow_trend"] = -2  # 加速流出
        elif net_5d < 0:
            feats["stock_flow_trend"] = -1  # 流出

        for col, feat_name in [("super_large_net", "stock_super_large_net_5d"), ("large_net", "stock_large_net_5d")]:
            if col in df.columns:
                feats[feat_name] = round(df[col].tail(5).sum(), 2) if len(df) >= 5 else round(df[col].sum(), 2)

    # === 北向资金 ===
    if north_flow_df is not None and not north_flow_df.empty:
        flow_col = "net_inflow" if "net_inflow" in north_flow_df.columns else None
        if flow_col:
            df = north_flow_df.sort_values("datetime")
            feats["north_flow_1d"] = round(df[flow_col].iloc[-1], 2) if len(df) > 0 else 0
            feats["north_flow_3d"] = round(df[flow_col].tail(3).sum(), 2) if len(df) >= 3 else 0
            feats["north_flow_trend"] = 1 if feats["north_flow_3d"] > 0 else (-1 if feats["north_flow_3d"] < 0 else 0)

    # === 市场资金 ===
    if market_flow_df is not None and not market_flow_df.empty:
        if "main_net_inflow" in market_flow_df.columns:
            feats["market_main_net"] = round(market_flow_df["main_net_inflow"].iloc[-1], 2) if len(market_flow_df) > 0 else 0
        if "main_net_pct" in market_flow_df.columns:
            feats["market_main_pct"] = round(market_flow_df["main_net_pct"].iloc[-1], 2) if len(market_flow_df) > 0 else 0

    # === 板块资金 ===
    stock_info = STOCK_POOL.get(stock_code, {})
    sector_name = stock_info.get("sector", "")
    if sector_flow_df is not None and not sector_flow_df.empty and "sector_name" in sector_flow_df.columns:
        match = sector_flow_df[sector_flow_df["sector_name"].str.contains(sector_name, na=False)]
        if not match.empty:
            row = match.iloc[0]
            feats["sector_main_net"] = round(row.get("main_net", 0), 2)
            feats["sector_main_pct"] = round(row.get("main_net_pct", 0), 2)

    # === 综合评分 ===
    score = 0.0
    # 个股资金趋势 (+/- 0.3)
    if feats["stock_flow_trend"] >= 2:
        score += 0.3
    elif feats["stock_flow_trend"] >= 1:
        score += 0.15
    elif feats["stock_flow_trend"] <= -2:
        score -= 0.3
    elif feats["stock_flow_trend"] <= -1:
        score -= 0.15

    # 北向资金 (+/- 0.2)
    score += 0.2 if feats["north_flow_trend"] > 0 else (-0.2 if feats["north_flow_trend"] < 0 else 0)

    # 板块资金 (+/- 0.2)
    if feats["sector_main_net"] > 0:
        score += 0.2
    elif feats["sector_main_net"] < 0:
        score -= 0.2

    # 市场资金 (+/- 0.3)
    if feats["market_main_pct"] > 5:
        score += 0.3
    elif feats["market_main_pct"] > 0:
        score += 0.1
    elif feats["market_main_pct"] < -5:
        score -= 0.3
    elif feats["market_main_pct"] < 0:
        score -= 0.1

    feats["capital_score"] = round(max(-1.0, min(1.0, score)), 4)

    return feats
