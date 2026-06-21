"""
美股→A股映射特征
计算隔夜美股走势对A股开盘的影响
"""

import logging
import numpy as np
import pandas as pd
from config import US_MAPPING, STOCK_CODES, STOCK_POOL

logger = logging.getLogger(__name__)


def compute_us_overnight_features(us_map_data: dict) -> dict:
    """
    从美股数据计算隔夜影响特征
    us_map_data: {a_stock_code: {us_ticker: DataFrame}}
    返回: {a_stock_code: {features}}
    """
    features = {}

    for a_code, us_stocks in us_map_data.items():
        if a_code not in US_MAPPING:
            continue

        feats = {
            "us_weighted_return": 0.0,
            "us_positive_ratio": 0.0,
            "us_overnight_signal": "neutral",
            "us_trend_3d": 0.0,
            "us_trend_5d": 0.0,
            "us_volatility": 0.0,
        }

        returns = []
        volatilities = []
        weights_sum = 0.0

        for us_ticker, df in us_stocks.items():
            if df is None or df.empty or "close" not in df.columns:
                continue

            weight = US_MAPPING[a_code].get(us_ticker, {}).get("weight", 0)
            weights_sum += weight

            # 最近1日涨跌
            if len(df) >= 2:
                daily_ret = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1)
                returns.append((daily_ret, weight))

            # 近3日/5日趋势
            if len(df) >= 5:
                trend_3d = (df["close"].iloc[-1] / df["close"].iloc[-3] - 1) if len(df) >= 3 else 0
                trend_5d = (df["close"].iloc[-1] / df["close"].iloc[-5] - 1)
                feats["us_trend_3d"] += weight * trend_3d
                feats["us_trend_5d"] += weight * trend_5d

            # 波动率
            if len(df) >= 5:
                vol = df["close"].pct_change().tail(5).std()
                volatilities.append(vol * weight)

        # 加权收益率
        if weights_sum > 0 and returns:
            weighted_ret = sum(r * w for r, w in returns) / weights_sum
            feats["us_weighted_return"] = round(weighted_ret, 6)
            positive_count = sum(1 for r, _ in returns if r > 0)
            feats["us_positive_ratio"] = round(positive_count / len(returns), 3)

        # 综合波动率
        if volatilities:
            feats["us_volatility"] = round(sum(volatilities) / weights_sum if weights_sum > 0 else 0, 6)

        # 信号判断
        wr = feats["us_weighted_return"]
        if wr > 0.01:
            feats["us_overnight_signal"] = "bullish"
        elif wr < -0.01:
            feats["us_overnight_signal"] = "bearish"
        else:
            feats["us_overnight_signal"] = "neutral"

        features[a_code] = feats

    return features


def get_us_index_features(us_indices: dict) -> dict:
    """
    计算美股三大指数特征
    """
    feats = {}
    for code, df in us_indices.items():
        if df is None or df.empty or "close" not in df.columns:
            continue

        name = code.replace("^", "")
        feats[f"{name}_return_1d"] = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) if len(df) >= 2 else 0
        feats[f"{name}_return_5d"] = (df["close"].iloc[-1] / df["close"].iloc[-5] - 1) if len(df) >= 5 else 0
        feats[f"{name}_trend"] = 1 if feats.get(f"{name}_return_5d", 0) > 0 else -1

    # 综合信号
    trends = [v for k, v in feats.items() if k.endswith("_trend")]
    if trends:
        feats["us_overall"] = "bullish" if sum(trends) > 0 else "bearish"
    else:
        feats["us_overall"] = "unknown"

    return feats
