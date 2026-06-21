"""
技术指标特征工程 - 基于5分钟/15分钟K线
均线系统、MACD、RSI、布林带、成交量、K线形态、分时特征
"""

import logging
import numpy as np
import pandas as pd
from config import TECH_PARAMS, TIMEFRAMES, PRIMARY_TF, ENTRY_TF

logger = logging.getLogger(__name__)


def add_ma_features(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
    """
    添加均线系统特征
    periods: MA周期列表，默认 [5, 10, 20, 60]
    """
    if periods is None:
        periods = TECH_PARAMS["ma_periods"]
    if "close" not in df.columns:
        return df

    df = df.copy()
    for p in periods:
        df[f"ma_{p}"] = df["close"].rolling(window=p).mean()

    # 均线位置关系
    if all(f"ma_{p}" in df.columns for p in [5, 10]):
        df["ma_5_10_cross"] = (df["ma_5"] > df["ma_10"]).astype(int)
        df["ma_5_10_dist"] = (df["ma_5"] / df["ma_10"] - 1) * 100  # 偏离百分比

    if all(f"ma_{p}" in df.columns for p in [5, 20]):
        df["ma_5_20_dist"] = (df["ma_5"] / df["ma_20"] - 1) * 100

    if all(f"ma_{p}" in df.columns for p in [10, 20]):
        df["ma_10_20_dist"] = (df["ma_10"] / df["ma_20"] - 1) * 100

    # 均线多空排列
    if all(f"ma_{p}" in df.columns for p in [5, 10, 20]):
        df["ma_bullish"] = ((df["ma_5"] > df["ma_10"]) & (df["ma_10"] > df["ma_20"])).astype(int)
        df["ma_bearish"] = ((df["ma_5"] < df["ma_10"]) & (df["ma_10"] < df["ma_20"])).astype(int)

    # 均线粘合度（标准差越小越粘合，可能突破）
    ma_cols = [f"ma_{p}" for p in periods[:3] if f"ma_{p}" in df.columns]
    if len(ma_cols) >= 3:
        df["ma_convergence"] = df[ma_cols].std(axis=1) / df[ma_cols].mean(axis=1)

    return df


def add_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加MACD及背离检测
    """
    if "close" not in df.columns:
        return df

    df = df.copy()
    fast = TECH_PARAMS["macd_fast"]
    slow = TECH_PARAMS["macd_slow"]
    signal = TECH_PARAMS["macd_signal"]

    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

    df["macd_dif"] = ema_fast - ema_slow
    df["macd_dea"] = df["macd_dif"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = (df["macd_dif"] - df["macd_dea"]) * 2  # 柱状图

    # MACD金叉死叉
    df["macd_golden_cross"] = (
        (df["macd_dif"] > df["macd_dea"]) &
        (df["macd_dif"].shift(1) <= df["macd_dea"].shift(1))
    ).astype(int)

    df["macd_dead_cross"] = (
        (df["macd_dif"] < df["macd_dea"]) &
        (df["macd_dif"].shift(1) >= df["macd_dea"].shift(1))
    ).astype(int)

    # MACD柱状图变化
    df["macd_hist_delta"] = df["macd_hist"] - df["macd_hist"].shift(1)
    df["macd_hist_sign_change"] = (
        (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)
    ).astype(int)  # 由负转正

    # MACD背离检测（简化版：价格新高而DIF不新高 = 顶背离）
    price_high_10 = df["high"].rolling(10).max()
    price_low_10 = df["low"].rolling(10).min()
    dif_high_10 = df["macd_dif"].rolling(10).max()
    dif_low_10 = df["macd_dif"].rolling(10).min()

    df["macd_bearish_divergence"] = (
        (df["high"] >= price_high_10) &
        (df["macd_dif"] < dif_high_10.shift(1))
    ).astype(int)

    df["macd_bullish_divergence"] = (
        (df["low"] <= price_low_10) &
        (df["macd_dif"] > dif_low_10.shift(1))
    ).astype(int)

    return df


def add_rsi_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加RSI特征"""
    if "close" not in df.columns:
        return df

    df = df.copy()
    period = TECH_PARAMS["rsi_period"]

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # RSI超买超卖
    df["rsi_overbought"] = (df["rsi"] > 70).astype(int)
    df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
    df["rsi_extreme_overbought"] = (df["rsi"] > 80).astype(int)
    df["rsi_extreme_oversold"] = (df["rsi"] < 20).astype(int)

    # RSI趋势
    df["rsi_rising"] = (df["rsi"] > df["rsi"].shift(3)).astype(int)
    df["rsi_falling"] = (df["rsi"] < df["rsi"].shift(3)).astype(int)

    return df


def add_bollinger_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加布林带特征"""
    if "close" not in df.columns:
        return df

    df = df.copy()
    period = TECH_PARAMS["boll_period"]
    std_mul = TECH_PARAMS["boll_std"]

    df["boll_mid"] = df["close"].rolling(window=period).mean()
    boll_std = df["close"].rolling(window=period).std()
    df["boll_upper"] = df["boll_mid"] + std_mul * boll_std
    df["boll_lower"] = df["boll_mid"] - std_mul * boll_std

    # 带宽（收窄预示变盘）
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / df["boll_mid"]
    df["boll_width_pct"] = df["boll_width"].rolling(20).apply(lambda x: (x.iloc[-1] / x.mean() - 1) if x.mean() > 0 else 0)

    # 价格在布林带中的位置（%B）
    df["boll_pct_b"] = (df["close"] - df["boll_lower"]) / (df["boll_upper"] - df["boll_lower"] + 1e-10)

    # 突破信号
    df["boll_upper_break"] = (df["high"] > df["boll_upper"]).astype(int)
    df["boll_lower_break"] = (df["low"] < df["boll_lower"]).astype(int)
    df["boll_squeeze"] = (df["boll_width_pct"] < -0.2).astype(int)  # 带宽收窄20%以上

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加成交量特征"""
    if "volume" not in df.columns:
        return df

    df = df.copy()

    # 成交量均线
    for p in TECH_PARAMS["volume_ma_periods"]:
        df[f"vol_ma_{p}"] = df["volume"].rolling(window=p).mean()

    # 量比 (当前量 / 5日均量)
    if "vol_ma_5" in df.columns:
        df["volume_ratio"] = df["volume"] / df["vol_ma_5"].replace(0, np.nan)
        df["volume_surge"] = (df["volume_ratio"] > 2.0).astype(int)
        df["volume_extreme"] = (df["volume_ratio"] > 3.0).astype(int)
        df["volume_shrink"] = (df["volume_ratio"] < 0.5).astype(int)

    # 量价关系
    price_up = (df["close"] > df["close"].shift(1)).astype(int)
    vol_up = (df["volume"] > df["volume"].shift(1)).astype(int)
    df["vol_price_up"] = (price_up & vol_up).astype(int)   # 量价齐升
    df["vol_price_div"] = (price_up & ~vol_up).astype(int)  # 价升量缩（背离）

    # 换手率（如果存在）
    if "turnover_rate" in df.columns:
        df["turnover_high"] = (df["turnover_rate"] > df["turnover_rate"].rolling(20).mean() * 1.5).astype(int)

    # 成交额变化
    if "amount" in df.columns:
        df["amount_ma_5"] = df["amount"].rolling(5).mean()
        df["amount_ratio"] = df["amount"] / df["amount_ma_5"].replace(0, np.nan)

    return df


def add_price_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加K线形态和价格特征"""
    if "close" not in df.columns or "open" not in df.columns:
        return df

    df = df.copy()

    # 单根K线特征
    df["body"] = df["close"] - df["open"]
    df["body_pct"] = df["body"] / df["open"].replace(0, np.nan) * 100
    df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["total_range"] = df["high"] - df["low"]

    # 实体占比
    df["body_ratio"] = abs(df["body"]) / df["total_range"].replace(0, 1e-10)

    # 锤子线/倒锤子 (下影线长，实体小)
    df["hammer"] = (
        (df["lower_shadow"] > abs(df["body"]) * 2) &
        (df["upper_shadow"] < abs(df["body"]) * 0.5) &
        (df["body_ratio"] < 0.3)
    ).astype(int)

    df["inverted_hammer"] = (
        (df["upper_shadow"] > abs(df["body"]) * 2) &
        (df["lower_shadow"] < abs(df["body"]) * 0.5) &
        (df["body_ratio"] < 0.3)
    ).astype(int)

    # 十字星
    df["doji"] = (abs(df["body"]) / df["total_range"].replace(0, 1e-10) < 0.1).astype(int)

    # 涨跌幅
    df["pct_change"] = df["close"].pct_change() * 100
    df["pct_change_3"] = (df["close"] / df["close"].shift(3) - 1) * 100
    df["pct_change_5"] = (df["close"] / df["close"].shift(5) - 1) * 100

    # 近期最高/最低
    for window in [10, 20, 60]:
        df[f"high_{window}"] = df["high"].rolling(window).max()
        df[f"low_{window}"] = df["low"].rolling(window).min()
        df[f"near_high_{window}"] = (df["close"] / df[f"high_{window}"].replace(0, np.nan) > 0.97).astype(int)
        df[f"near_low_{window}"] = (df["close"] / df[f"low_{window}"].replace(0, np.nan) < 1.03).astype(int)

    return df


def add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    分时特征（专用于5分钟/15分钟级别）
    """
    if "datetime" not in df.columns or "close" not in df.columns:
        return df

    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["time_minutes"] = df["hour"] * 60 + df["minute"]

    # 是否开盘/收盘时段
    df["is_open_period"] = ((df["hour"] == 9) & (df["minute"] >= 30)) | (df["hour"] == 10) | ((df["hour"] == 11) & (df["minute"] <= 30))
    df["is_close_period"] = (df["hour"] == 14) | ((df["hour"] == 15) & (df["minute"] == 0))
    df["is_morning"] = df["hour"] < 12
    df["is_afternoon"] = df["hour"] >= 13

    # 开盘强弱（以开盘后30分钟为窗口）
    df["date"] = df["datetime"].dt.date
    for date, group in df.groupby("date"):
        if len(group) < 10:
            continue
        idx = group.index
        open_price = group["open"].iloc[0]
        if open_price > 0:
            # 开盘30分钟后价格相对开盘价的变化
            after_30min = group[group["time_minutes"] >= group["time_minutes"].iloc[0] + 30]
            if not after_30min.empty:
                df.loc[after_30min.index, "open_strength"] = after_30min["close"].iloc[0] / open_price - 1

    if "open_strength" in df.columns:
        df["open_strength"] = df["open_strength"].fillna(0)
    else:
        df["open_strength"] = 0.0

    # 日内V型反转检测
    if "date" in df.columns:
        for date, group in df.groupby("date"):
            if len(group) < 20:
                continue
            idx = group.index
            close = group["close"].values
            # 简化的V转：先跌后涨
            min_idx = close.argmin()
            if 3 < min_idx < len(close) - 3:
                prior_drop = (close[0] / close[min_idx] - 1) > 0.01  # 跌了>1%
                later_rise = (close[-1] / close[min_idx] - 1) > 0.01  # 涨回>1%
                if prior_drop and later_rise:
                    df.loc[idx, "v_reversal"] = 1

    if "v_reversal" in df.columns:
        df["v_reversal"] = df["v_reversal"].fillna(0).astype(int)
    else:
        df["v_reversal"] = 0

    # ATR (平均真实波幅)
    if all(c in df.columns for c in ["high", "low", "close"]):
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1 / TECH_PARAMS["atr_period"], adjust=False).mean()
        df["atr_pct"] = df["atr"] / df["close"].replace(0, np.nan) * 100

    # 清理临时列
    df.drop(columns=["hour", "minute", "time_minutes", "is_morning", "is_afternoon"], inplace=True, errors="ignore")

    return df


def compute_all_technical_features(df: pd.DataFrame, tf: str = "15min") -> pd.DataFrame:
    """
    计算所有技术指标特征的主入口
    按顺序调用各项特征函数
    """
    logger.info(f"计算技术指标 ({tf})...")
    df = df.copy()

    df = add_ma_features(df)
    df = add_macd_features(df)
    df = add_rsi_features(df)
    df = add_bollinger_features(df)
    df = add_volume_features(df)
    df = add_price_pattern_features(df)
    df = add_intraday_features(df)

    # 移除初始NaN行（滚动计算导致的）
    df.dropna(subset=["close"], inplace=True)

    logger.info(f"技术指标计算完成, shape={df.shape}, features={len(df.columns)}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== 技术指标测试 ===")
    # 生成模拟5分钟数据
    dates = pd.date_range("2025-01-02 09:30", periods=200, freq="5min")
    np.random.seed(42)
    price = 20.0
    data = []
    for d in dates:
        change = np.random.normal(0, 0.2)
        price += change
        data.append({
            "datetime": d,
            "open": price - np.random.uniform(0, 0.1),
            "high": price + np.random.uniform(0, 0.2),
            "low": price - np.random.uniform(0, 0.2),
            "close": price,
            "volume": np.random.randint(10000, 100000),
            "amount": np.random.randint(1000000, 10000000),
        })
    test_df = pd.DataFrame(data)
    result = compute_all_technical_features(test_df)
    print(f"生成特征数: {len(result.columns)}")
    print(f"特征列表: {list(result.columns)}")
