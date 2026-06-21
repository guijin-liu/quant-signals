"""
特征工程流水线
协调所有特征模块，生成统一的训练/预测数据集
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from config import (
    STOCK_CODES, STOCK_POOL, TRAIN_PARAMS, PRIMARY_TF, ENTRY_TF,
    SIGNAL_PARAMS,
)

from features.technical import compute_all_technical_features
from features.market_env import add_market_features, compute_market_regime
from features.us_mapper import compute_us_overnight_features, get_us_index_features
from features.sector_features import extract_sector_features
from features.money_flow import extract_money_flow_features
from features.sentiment import extract_sentiment_features, compute_market_sentiment

logger = logging.getLogger(__name__)


def build_features_for_stock(
    stock_code: str,
    kline_df: pd.DataFrame,
    market_indices: dict,
    us_features: dict,
    sector_ranks: dict,
    capital_flow: dict,
    news_sentiment: dict,
    tf: str = "15min",
) -> pd.DataFrame:
    """
    为单只股票构建完整特征集
    """
    df = kline_df.copy()
    if df.empty:
        logger.warning(f"{stock_code} K线为空，跳过特征构建")
        return df

    logger.info(f"构建 {stock_code} ({STOCK_POOL.get(stock_code, {}).get('name', '')}) 特征集...")

    # 1. 技术指标
    df = compute_all_technical_features(df, tf)

    # 2. 大盘环境
    df = add_market_features(df, market_indices)

    # 3. 美股映射（同一只股票的所有K线用同样的隔夜特征）
    stock_us_feats = us_features.get(stock_code, {})
    for feat_name, feat_val in stock_us_feats.items():
        df[f"us_{feat_name}"] = feat_val

    # 4. 板块特征
    sector_feats = extract_sector_features(sector_ranks, stock_code)
    for feat_name, feat_val in sector_feats.items():
        df[f"{feat_name}"] = feat_val

    # 5. 资金流向特征
    stock_flow_df = capital_flow.get("stock_flows", {}).get(stock_code, pd.DataFrame())
    north_flow = capital_flow.get("north_flow", pd.DataFrame())
    market_flow = capital_flow.get("market_flow", pd.DataFrame())
    sector_flow = capital_flow.get("sector_flow", pd.DataFrame())
    money_feats = extract_money_flow_features(stock_flow_df, north_flow, market_flow, sector_flow, stock_code)
    for feat_name, feat_val in money_feats.items():
        df[f"flow_{feat_name}"] = feat_val

    # 6. 新闻情绪
    sent_feats = extract_sentiment_features(news_sentiment, stock_code)
    for feat_name, feat_val in sent_feats.items():
        df[f"news_{feat_name}"] = feat_val

    # 7. 股票代码标签
    df["symbol"] = stock_code
    df["stock_name"] = STOCK_POOL.get(stock_code, {}).get("name", "")

    return df


def build_training_target(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.015) -> pd.DataFrame:
    """
    构建训练目标 (Label)
    horizon: 预测未来N根K线后的涨跌
    threshold: 涨跌分类阈值

    标签:
      2 = 大涨 (>threshold)
      1 = 涨 (0 ~ threshold)
      0 = 平/跌 (-threshold ~ 0)
      (实际使用时合并为 2/1/0 三分类，或 1/0 二分类)
    """
    df = df.copy()
    df["future_close"] = df["close"].shift(-horizon)
    df["future_return"] = (df["future_close"] / df["close"] - 1)

    # 三分类
    df["target_3class"] = 0
    df.loc[df["future_return"] > threshold, "target_3class"] = 2   # 大涨 - 买入
    df.loc[(df["future_return"] > 0) & (df["future_return"] <= threshold), "target_3class"] = 1  # 小涨 - 持有
    df.loc[df["future_return"] <= -threshold, "target_3class"] = 0  # 大跌 - 卖出/做空

    # 二分类（简化版：涨/跌）
    df["target_binary"] = (df["future_return"] > 0).astype(int)

    # 回归目标
    df["target_regression"] = df["future_return"]

    df.drop(columns=["future_close"], inplace=True)
    return df


def build_full_dataset(
    stock_kline_data: dict,
    market_indices: dict,
    us_map_data: dict,
    us_indices: dict,
    sector_data: dict,
    capital_flow_data: dict,
    news_data: dict,
    tf: str = "15min",
) -> dict:
    """
    构建完整数据集，包含所有特征和目标变量
    返回: {
        'features': combined DataFrame,
        'feature_cols': list of feature column names,
        'stocks': {code: DataFrame}
    }
    """
    logger.info("=" * 60)
    logger.info("开始构建完整特征数据集...")
    logger.info("=" * 60)

    # 美股特征
    us_features = compute_us_overnight_features(us_map_data)
    us_index_feats = get_us_index_features(us_indices)
    # 将美股指数特征合并到每只股票
    for code in STOCK_CODES:
        if code not in us_features:
            us_features[code] = {}
        us_features[code].update(us_index_feats)

    # 板块排名
    sector_rank_df = sector_data.get("rank", pd.DataFrame())
    from features.sector_features import compute_sector_momentum
    sector_ranks_raw = compute_sector_momentum(sector_rank_df)

    # 新闻情绪 (get_all_news 返回的 sentiment 部分)
    news_sentiment = news_data.get("sentiment", {})

    # 逐股票构建
    all_stock_dfs = []
    for code in STOCK_CODES:
        kline = stock_kline_data.get(code)
        if kline is None or kline.empty:
            logger.warning(f"{code} 无K线数据，跳过")
            continue

        df = build_features_for_stock(
            stock_code=code,
            kline_df=kline,
            market_indices=market_indices,
            us_features=us_features,
            sector_ranks=sector_ranks_raw,
            capital_flow=capital_flow_data,
            news_sentiment=news_sentiment,
            tf=tf,
        )

        if not df.empty:
            # 构建目标变量
            df = build_training_target(
                df,
                horizon=TRAIN_PARAMS["target_horizon"],
                threshold=TRAIN_PARAMS["target_threshold"],
            )
            all_stock_dfs.append(df)

    if not all_stock_dfs:
        logger.error("没有成功构建任何特征数据")
        return {"features": pd.DataFrame(), "feature_cols": [], "stocks": {}}

    combined = pd.concat(all_stock_dfs, ignore_index=True)
    combined.sort_values(["datetime", "symbol"], inplace=True)

    # 识别特征列（排除非特征列）
    exclude_cols = {
        "datetime", "date", "symbol", "stock_name",
        "target_3class", "target_binary", "target_regression",
        "future_return", "index_code", "index_name",
    }
    feature_cols = [c for c in combined.columns if c not in exclude_cols]

    logger.info(f"特征数据集构建完成:")
    logger.info(f"  总行数: {len(combined)}")
    logger.info(f"  特征数: {len(feature_cols)}")
    logger.info(f"  股票数: {len(all_stock_dfs)}")

    return {
        "features": combined,
        "feature_cols": feature_cols,
        "stocks": {df["symbol"].iloc[0]: df for df in all_stock_dfs},
    }


def prepare_train_data(
    dataset: dict,
    test_size: float = 0.2,
) -> tuple:
    """
    准备训练/测试数据
    按时序切分（避免未来函数）
    """
    df = dataset["features"].copy()
    feature_cols = dataset["feature_cols"]

    # 移除NaN行
    df = df.dropna(subset=feature_cols + ["target_3class"])
    df = df[~df["target_3class"].isna()]

    if df.empty:
        logger.error("无有效训练数据")
        return None, None, None, None, None, None

    # 按时序切分
    split_idx = int(len(df) * TRAIN_PARAMS["train_test_split"])
    df = df.sort_values("datetime")

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[feature_cols].select_dtypes(include=[np.number])
    y_train = train["target_3class"]
    X_test = test[feature_cols].select_dtypes(include=[np.number])
    y_test = test["target_3class"]

    logger.info(f"训练数据: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"测试数据: X={X_test.shape}, y={y_test.shape}")
    logger.info(f"标签分布 - 训练: {y_train.value_counts().to_dict()}")
    logger.info(f"标签分布 - 测试: {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test, train, test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== 特征流水线测试 ===")
    print("需要实际数据源，此模块通过 main.py 调用")
