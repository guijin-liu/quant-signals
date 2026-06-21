"""
MA双周期共振特征模块 — 增强技术维度
不替代6维框架，而是让技术维度更强：
  15分钟均线 → 定方向（bull/bear/neutral + 评分）
  5分钟均线  → 找买卖点（金叉/死叉/排列 + 评分）
  双周期共振 → 两周期方向一致时加分

可以被 features/technical.py 和 models/signal.py 调用
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ======================== MA共振参数 ========================
MA_RESONANCE_CONFIG = {
    # 15min 方向均线
    "trend_ma_fast": 5,       # MA5
    "trend_ma_mid": 10,       # MA10
    "trend_ma_slow": 20,      # MA20
    "trend_ma_very_slow": 60, # MA60 (强支撑/压力)

    # 5min 入场均线
    "entry_ma_fast": 5,       # MA5
    "entry_ma_mid": 10,       # MA10
    "entry_ma_slow": 20,      # MA20

    # 金叉死叉质量评分
    "cross_quality": {
        "golden_cross": 0.35,    # 金叉加分
        "dead_cross": -0.35,     # 死叉减分
        "ma_alignment": 0.25,    # 均线多头排列加分
        "ma_bearish": -0.25,     # 均线空头排列减分
        "volume_confirm": 0.15,  # 放量确认加分
        "rsi_confirm": 0.10,     # RSI确认加分
        "price_vs_ma20": 0.15,   # 价格vs MA20
    },

    # 阈值
    "rsi_buy_low": 30,
    "rsi_buy_high": 65,
    "rsi_sell_high": 70,
    "volume_ratio_min": 1.2,     # 放量最低倍率
    "ma_convergence_pct": 0.005, # 均线粘合判断(0.5%)
    "ma_divergence_pct": 0.02,   # 均线发散判断(2%)

    # 共振
    "resonance_bonus": 0.10,     # 双周期方向一致加分
    "divergence_penalty": -0.10, # 双周期方向相反减分
}


def compute_ma_scores_15min(df_15min: pd.DataFrame) -> pd.DataFrame:
    """
    基于15分钟K线计算均线方向评分
    返回带 ma_* 列和 trend_direction + trend_score 的DataFrame
    """
    df = df_15min.copy()
    cfg = MA_RESONANCE_CONFIG
    periods = [cfg["trend_ma_fast"], cfg["trend_ma_mid"],
               cfg["trend_ma_slow"], cfg["trend_ma_very_slow"]]

    for p in periods:
        col = f"ma15_{p}"
        df[col] = df["close"].rolling(window=p).mean()

    # 均线排列评分
    scores = []
    directions = []

    for i in range(len(df)):
        row = df.iloc[i]
        ma5 = row.get(f"ma15_{cfg['trend_ma_fast']}", np.nan)
        ma10 = row.get(f"ma15_{cfg['trend_ma_mid']}", np.nan)
        ma20 = row.get(f"ma15_{cfg['trend_ma_slow']}", np.nan)
        ma60 = row.get(f"ma15_{cfg['trend_ma_very_slow']}", np.nan)

        if any(np.isnan(x) for x in [ma5, ma10, ma20]):
            scores.append(0.5)
            directions.append("neutral")
            continue

        score = 0.5

        # 完全多头: MA5 > MA10 > MA20
        if ma5 > ma10 > ma20:
            score += 0.30
            if not np.isnan(ma60) and row["close"] > ma60:
                score += 0.10
        elif ma5 > ma10:
            score += 0.15
        elif ma5 > ma20:
            score += 0.05

        # 完全空头: MA5 < MA10 < MA20
        if ma5 < ma10 < ma20:
            score -= 0.30
            if not np.isnan(ma60) and row["close"] < ma60:
                score -= 0.10
        elif ma5 < ma10:
            score -= 0.15

        # 均线粘合 → 向中性靠拢
        mas = [ma5, ma10, ma20]
        ma_std = np.std(mas) / np.mean(mas)
        if ma_std < cfg["ma_convergence_pct"]:
            score = 0.5 + (score - 0.5) * 0.5

        # 均线发散 → 顺势加分
        if ma_std > cfg["ma_divergence_pct"] and score > 0.6:
            score += 0.05

        score = max(0.0, min(1.0, score))
        scores.append(score)

        if score >= 0.65:
            directions.append("bull")
        elif score <= 0.35:
            directions.append("bear")
        else:
            directions.append("neutral")

    df["trend_direction"] = directions
    df["trend_score"] = scores

    return df


def compute_ma_scores_5min(df_5min: pd.DataFrame) -> pd.DataFrame:
    """
    基于5分钟K线计算入场均线特征
    返回带 entry_score, ma5_*, ma10_*, ma20_* 金叉/死叉标记的DataFrame
    """
    df = df_5min.copy()
    cfg = MA_RESONANCE_CONFIG
    fast, mid, slow = cfg["entry_ma_fast"], cfg["entry_ma_mid"], cfg["entry_ma_slow"]

    # 计算均线
    for p in [fast, mid, slow]:
        df[f"ma5_{p}"] = df["close"].rolling(window=p).mean()

    # 金叉死叉
    df["ma_golden_cross"] = (
        (df[f"ma5_{fast}"] > df[f"ma5_{mid}"]) &
        (df[f"ma5_{fast}"].shift(1) <= df[f"ma5_{mid}"].shift(1))
    ).astype(int)

    df["ma_dead_cross"] = (
        (df[f"ma5_{fast}"] < df[f"ma5_{mid}"]) &
        (df[f"ma5_{fast}"].shift(1) >= df[f"ma5_{mid}"].shift(1))
    ).astype(int)

    # 均线多头/空头排列
    df["ma_bullish_5min"] = (
        (df[f"ma5_{fast}"] > df[f"ma5_{mid}"]) &
        (df[f"ma5_{mid}"] > df[f"ma5_{slow}"])
    ).astype(int)

    df["ma_bearish_5min"] = (
        (df[f"ma5_{fast}"] < df[f"ma5_{mid}"]) &
        (df[f"ma5_{mid}"] < df[f"ma5_{slow}"])
    ).astype(int)

    # 均线部分多头（快线穿中线但未穿慢线）
    df["ma_partial_bull"] = (
        (df[f"ma5_{fast}"] > df[f"ma5_{mid}"]) &
        (df[f"ma5_{mid}"] < df[f"ma5_{slow}"])
    ).astype(int)

    df["ma_partial_bear"] = (
        (df[f"ma5_{fast}"] < df[f"ma5_{mid}"]) &
        (df[f"ma5_{mid}"] > df[f"ma5_{slow}"])
    ).astype(int)

    # 均线粘合度
    ma_cols = [f"ma5_{p}" for p in [fast, mid, slow]]
    df["ma_convergence_5min"] = df[ma_cols].std(axis=1) / df[ma_cols].mean(axis=1)

    # 价格相对MA20位置
    df["price_vs_ma20_5min"] = df["close"] / df[f"ma5_{slow}"] - 1

    return df


def score_entry_quality(row: pd.Series, trend_direction: str = "neutral") -> float:
    """
    单根5min K线的入场质量评分 (0~1)
    结合15min趋势方向做共振判断

    供 SignalGenerator.score_factors 调用
    """
    cfg = MA_RESONANCE_CONFIG
    w = cfg["cross_quality"]
    score = 0.50

    # === 1. 金叉/死叉 ===
    if row.get("ma_golden_cross", 0):
        score += w["golden_cross"]
    elif row.get("ma_dead_cross", 0):
        score += w["dead_cross"]  # 负值

    # === 2. 均线排列 ===
    if row.get("ma_bullish_5min", 0):
        score += w["ma_alignment"]
    elif row.get("ma_bearish_5min", 0):
        score += w["ma_bearish"]
    elif row.get("ma_partial_bull", 0):
        score += w["ma_alignment"] * 0.5
    elif row.get("ma_partial_bear", 0):
        score += w["ma_bearish"] * 0.5

    # === 3. 成交量 ===
    vol_ratio = row.get("volume_ratio", row.get("vol_price_up", 0))
    if not np.isnan(vol_ratio):
        if vol_ratio > cfg["volume_ratio_min"] and score > 0.55:
            score += w["volume_confirm"]
        elif vol_ratio < 0.5:
            score -= w["volume_confirm"] * 0.5

    # === 4. RSI ===
    rsi = row.get("rsi", 50)
    if not np.isnan(rsi):
        if cfg["rsi_buy_low"] < rsi < cfg["rsi_buy_high"] and score > 0.55:
            score += w["rsi_confirm"]
        elif rsi > cfg["rsi_sell_high"]:
            score -= w["rsi_confirm"]
        elif rsi < cfg["rsi_buy_low"]:
            score += w["rsi_confirm"] * 0.5  # 超卖反弹

    # === 5. 价格位置 ===
    dist = row.get("price_vs_ma20_5min", 0)
    if not np.isnan(dist):
        if -0.03 < dist < 0.03:
            if dist > 0:
                score += w["price_vs_ma20"] * 0.7
            elif dist < -0.01:
                score -= w["price_vs_ma20"] * 0.3

    # === 6. 双周期共振 ===
    if trend_direction == "bull":
        score += cfg["resonance_bonus"]
    elif trend_direction == "bear":
        score += cfg["divergence_penalty"]

    return max(0.0, min(1.0, score))


def map_trend_to_5min(df_5min: pd.DataFrame,
                       df_15min: pd.DataFrame) -> pd.DataFrame:
    """
    将15分钟趋势方向映射到每根5分钟K线
    每根15min K线覆盖3根5min K线（实际用时间对齐）

    Returns: df_5min with added 'trend_direction' and 'trend_score' columns
    """
    df5 = df_5min.copy()
    df15 = df_15min.copy()

    if "trend_direction" not in df15.columns:
        df15 = compute_ma_scores_15min(df15)

    df5["trend_direction"] = "neutral"
    df5["trend_score"] = 0.5

    for i in range(len(df5)):
        dt = df5.iloc[i]["datetime"]
        # 找时间上不晚于当前5min K线的最近15min K线
        mask = df15["datetime"] <= dt
        if mask.any():
            match = df15[mask].iloc[-1]
            df5.at[df5.index[i], "trend_direction"] = match["trend_direction"]
            df5.at[df5.index[i], "trend_score"] = match["trend_score"]

    return df5


def compute_ma_resonance_features(df_5min: pd.DataFrame,
                                   df_15min: pd.DataFrame = None) -> pd.DataFrame:
    """
    主入口：计算MA双周期共振的所有特征

    Args:
        df_5min: 5分钟K线 (或单周期数据)
        df_15min: 15分钟K线 (可选，有的话做双周期共振)

    Returns:
        带全MA共振特征的DataFrame
    """
    # 5min入场均线
    df = compute_ma_scores_5min(df_5min)

    # 如果有15min数据，做双周期共振
    if df_15min is not None and not df_15min.empty:
        df = map_trend_to_5min(df, df_15min)
    else:
        df["trend_direction"] = "neutral"
        df["trend_score"] = 0.5

    # 计算RSI
    if "rsi" not in df.columns:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

    # 计算量比
    if "volume_ratio" not in df.columns and "volume" in df.columns:
        vol_ma5 = df["volume"].rolling(5).mean()
        df["volume_ratio"] = df["volume"] / vol_ma5.replace(0, np.nan)

    # 逐行计算入场评分
    for i in range(len(df)):
        row = df.iloc[i]
        df.at[df.index[i], "ma_entry_score"] = score_entry_quality(
            row, row.get("trend_direction", "neutral")
        )

    # 共振维度数（MA共振可视为技术维度的子维度）
    df["ma_resonance_active"] = (
        (df["ma_entry_score"] > 0.65).astype(int)
    )

    logger.info(f"MA共振特征计算完成: {len(df)}行, "
                f"ma_entry_score>0.65: {(df['ma_entry_score']>0.65).sum()}行, "
                f"ma_entry_score<0.35: {(df['ma_entry_score']<0.35).sum()}行")

    return df
