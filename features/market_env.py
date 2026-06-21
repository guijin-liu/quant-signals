"""
大盘环境特征
- A股大盘趋势
- 大盘与个股相关性
- 市场宽度
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_market_trend_features(index_df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    计算大盘指数的趋势特征
    """
    if index_df is None or index_df.empty:
        return pd.DataFrame()

    df = index_df.copy().sort_values("datetime")
    prefix = f"{label}_" if label else ""

    if "close" not in df.columns:
        return df

    # 均线
    df[f"{prefix}ma_20"] = df["close"].rolling(20).mean()
    df[f"{prefix}ma_60"] = df["close"].rolling(60).mean()

    # 趋势判断
    df[f"{prefix}above_ma20"] = (df["close"] > df[f"{prefix}ma_20"]).astype(int)
    df[f"{prefix}above_ma60"] = (df["close"] > df[f"{prefix}ma_60"]).astype(int)
    df[f"{prefix}ma_bullish"] = (df[f"{prefix}ma_20"] > df[f"{prefix}ma_60"]).astype(int)

    # 涨跌幅
    df[f"{prefix}pct_change"] = df["close"].pct_change() * 100
    df[f"{prefix}pct_change_5"] = (df["close"] / df["close"].shift(5) - 1) * 100
    df[f"{prefix}pct_change_20"] = (df["close"] / df["close"].shift(20) - 1) * 100

    # 波动率
    df[f"{prefix}volatility_5"] = df[f"{prefix}pct_change"].rolling(5).std()
    df[f"{prefix}volatility_20"] = df[f"{prefix}pct_change"].rolling(20).std()

    # 相对强弱（近期涨幅/波动率）
    vol = df[f"{prefix}volatility_20"].replace(0, np.nan)
    df[f"{prefix}strength"] = df[f"{prefix}pct_change_20"] / vol

    # 高点/低点
    df[f"{prefix}high_20"] = df["high"].rolling(20).max()
    df[f"{prefix}low_20"] = df["low"].rolling(20).min()
    df[f"{prefix}near_high"] = (df["close"] / df[f"{prefix}high_20"].replace(0, np.nan) > 0.98).astype(int)
    df[f"{prefix}near_low"] = (df["close"] / df[f"{prefix}low_20"].replace(0, np.nan) < 1.02).astype(int)

    return df


def compute_market_regime(index_dict: dict) -> dict:
    """
    判断当前市场状态（基于多个指数综合判断）
    返回: {regime: 'bull'/'bear'/'range', score: float}
    """
    scores = []
    for code, df in index_dict.items():
        if df is None or df.empty or "close" not in df.columns:
            continue

        # 取近期数据判断
        recent = df.tail(20)
        if len(recent) < 10:
            continue

        # 趋势得分
        start = recent["close"].iloc[0]
        end = recent["close"].iloc[-1]
        ret = (end / start - 1) * 100

        # 均线位置
        if "ma_20" in recent.columns:
            above_ma = (recent["close"].iloc[-1] > recent["ma_20"].iloc[-1])
        else:
            ma_20 = recent["close"].rolling(20).mean().iloc[-1] if len(recent) >= 20 else recent["close"].mean()
            above_ma = recent["close"].iloc[-1] > ma_20

        score = ret * (1.2 if above_ma else 0.8)  # 均线上加分
        scores.append(score)

    if not scores:
        return {"regime": "unknown", "score": 0.0}

    avg_score = np.mean(scores)

    if avg_score > 2:
        regime = "bull"
    elif avg_score < -2:
        regime = "bear"
    else:
        regime = "range"

    return {"regime": regime, "score": round(float(avg_score), 3)}


def compute_correlation_with_market(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    window: int = 20,
) -> float:
    """计算个股与大盘的相关性"""
    if stock_df is None or stock_df.empty or index_df is None or index_df.empty:
        return 0.0

    try:
        stock = stock_df.set_index("datetime")["close"].resample("15min").last().dropna()
        idx = index_df.set_index("datetime")["close"].resample("15min").last().dropna()

        common_idx = stock.index.intersection(idx.index)
        if len(common_idx) < window:
            return 0.0

        stock_aligned = stock.loc[common_idx].pct_change().dropna()
        idx_aligned = idx.loc[common_idx].pct_change().dropna()

        if len(stock_aligned) < window:
            return 0.0

        corr = stock_aligned.tail(window).corr(idx_aligned.tail(window))
        return round(float(corr), 4)
    except Exception:
        return 0.0


def add_market_features(
    stock_df: pd.DataFrame,
    market_indices: dict,
) -> pd.DataFrame:
    """
    为个股数据添加大盘环境特征
    """
    df = stock_df.copy()
    if df.empty:
        return df

    # 找上证指数作为主要参考
    sh_index = market_indices.get("sh000001")
    sz_index = market_indices.get("sz399001")

    # 对齐时间，合并大盘涨跌幅
    if sh_index is not None and not sh_index.empty:
        sh_idx = sh_index[["datetime", "close"]].copy()
        sh_idx.rename(columns={"close": "sh_close"}, inplace=True)
        sh_idx["sh_pct_change"] = sh_idx["sh_close"].pct_change() * 100
        df = pd.merge(df, sh_idx[["datetime", "sh_pct_change", "sh_close"]], on="datetime", how="left")

        if "sh_pct_change" in df.columns:
            df["sh_pct_change"] = df["sh_pct_change"].fillna(0)

    return df
