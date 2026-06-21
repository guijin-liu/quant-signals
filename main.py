#!/usr/bin/env python
"""
量化交易系统 - 主入口
多维度分析（美股映射 + 大盘 + 板块 + 新闻 + 资金流向）× 5/15分钟短线
目标胜率 > 88%

用法:
  python main.py fetch          # 获取所有数据
  python main.py train          # 训练模型
  python main.py backtest       # 运行回测（日线3-5年 + 短线信号）
  python main.py signal         # 生成最新交易信号
  python main.py all            # 完整流程
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    STOCK_CODES, STOCK_POOL, TIMEFRAMES, PRIMARY_TF, ENTRY_TF,
    BACKTEST_PARAMS, TRAIN_PARAMS,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "quant.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


# ======================== 数据获取 ========================

def fetch_all_data():
    """获取所有数据源"""
    logger.info("=" * 60)
    logger.info("开始获取所有数据...")
    logger.info("=" * 60)

    from data.a_share import get_all_stocks_kline, get_market_indices
    from data.us_market import get_us_mapping_data, get_us_indices
    from data.sector import get_relevant_sectors, find_stock_sector_rank
    from data.capital_flow import get_all_capital_flow
    from data.news import get_all_news
    from config import SECTORS

    data = {}

    # 1. A股个股K线 (5min + 15min)
    logger.info("\n[1/6] 获取A股个股分钟K线...")
    data["stock_5min"] = get_all_stocks_kline("5min")
    data["stock_15min"] = get_all_stocks_kline("15min")

    # 2. A股大盘指数
    logger.info("\n[2/6] 获取A股大盘指数...")
    data["market_15min"] = get_market_indices("15min")

    # 3. 美股数据
    logger.info("\n[3/6] 获取美股映射数据...")
    data["us_map"] = get_us_mapping_data()
    data["us_indices"] = get_us_indices("15min")

    # 4. 板块数据
    logger.info("\n[4/6] 获取板块数据...")
    sector_data = get_relevant_sectors()
    data["sector_rank"] = sector_data["rank"]
    data["sector_ranks"] = find_stock_sector_rank(sector_data["rank"], SECTORS)
    data["sector_data"] = sector_data

    # 5. 资金流向
    logger.info("\n[5/6] 获取资金流向数据...")
    data["capital_flow"] = get_all_capital_flow()

    # 6. 新闻情绪
    logger.info("\n[6/6] 获取新闻情绪数据...")
    data["news_data"] = get_all_news()

    logger.info("\n数据获取完成!")
    _print_data_summary(data)
    return data


def _print_data_summary(data: dict):
    """打印数据摘要"""
    print("\n" + "=" * 60)
    print("数据摘要")
    print("=" * 60)
    for tf, stock_data in [("5min", data.get("stock_5min", {})), ("15min", data.get("stock_15min", {}))]:
        for code, df in stock_data.items():
            if not df.empty:
                print(f"  {code} {STOCK_POOL[code]['name']} {tf}: {len(df)}条, "
                      f"{df['datetime'].min()} ~ {df['datetime'].max()}")

    market = data.get("market_15min", {})
    for code, df in market.items():
        if not df.empty:
            print(f"  大盘 {code}: {len(df)}条")

    us = data.get("us_map", {})
    us_count = sum(len(df) for stocks in us.values() for df in stocks.values())
    print(f"  美股映射: {us_count}条")

    cf = data.get("capital_flow", {})
    nf = cf.get("north_flow", None)
    if nf is not None and not nf.empty:
        print(f"  北向资金: {len(nf)}条")
    for code, df in cf.get("stock_flows", {}).items():
        if not df.empty:
            print(f"  资金流向 {code}: {len(df)}条")

    news = data.get("news_data", {})
    for code, sentiments in news.get("sentiment", {}).items():
        print(f"  新闻情绪 {code}: {len(sentiments)}条")

    print("=" * 60)


# ======================== 日线数据获取（用于3-5年回测） ========================

def fetch_daily_data(years: int = 5):
    """
    获取日线数据用于长期回测
    优先使用 baostock (公司网络屏蔽东方财富/akshare)
    """
    import pandas as pd
    from datetime import datetime
    from data.fetcher import cache
    from data.baostock_fetcher import fetch_daily_with_cache as bs_fetch_daily

    logger.info(f"获取近{years}年日线数据 (baostock)...")

    result = {}
    for code in STOCK_CODES:
        try:
            df = bs_fetch_daily(code, years)
            if df is not None and not df.empty:
                result[code] = df
                logger.info(f"  {code} {STOCK_POOL[code]['name']}: {len(df)}条, "
                           f"{df['datetime'].min().date()} ~ {df['datetime'].max().date()}")
            else:
                logger.warning(f"  {code}: 无数据")
        except Exception as e:
            logger.error(f"  获取 {code} 日线失败: {e}")

    return result


# ======================== 回测（日线级别，3-5年） ========================

def run_daily_backtest(data: dict = None, years: int = 5):
    """
    日线级别回测（3-5年历史数据）
    日线数据量大，用日线验证策略有效性
    """
    logger.info("=" * 60)
    logger.info(f"开始日线回测 (近{years}年)...")
    logger.info("=" * 60)

    # 获取日线数据
    daily_data = fetch_daily_data(years)
    if not daily_data:
        logger.error("无日线数据，无法回测")
        return

    # 获取其他数据源
    if data is None:
        logger.info("获取辅助数据...")
        data = {}
        try:
            from data.sector import get_relevant_sectors
            from data.capital_flow import get_all_capital_flow
            from data.news import get_all_news
            from data.us_market import get_us_mapping_data

            sector_result = get_relevant_sectors()
            data["sector_data"] = sector_result
            data["capital_flow"] = get_all_capital_flow()
            data["news_data"] = get_all_news()
            data["us_map"] = get_us_mapping_data()
            data["market_indices"] = {}
        except Exception as e:
            logger.warning(f"获取辅助数据失败: {e}，将使用简化回测")

    # 对日线数据计算技术指标
    from features.technical import compute_all_technical_features

    all_signals = []
    for code, df in daily_data.items():
        logger.info(f"处理 {code} {STOCK_POOL[code]['name']}...")
        df = compute_all_technical_features(df, "daily")
        if df.empty:
            continue

        # 简化信号生成（日线级别）
        df = _generate_daily_signals(df)
        all_signals.append(df)

    if not all_signals:
        logger.error("无有效信号数据")
        return

    # 合并，运行回测
    import pandas as pd
    signal_df = pd.concat(all_signals, ignore_index=True)
    signal_df.sort_values(["datetime", "symbol"], inplace=True)

    from backtest.engine import BacktestEngine
    from backtest.report import save_report

    engine = BacktestEngine()
    results = engine.run(signal_df)
    engine.print_summary()

    # 保存报告
    report_dir = save_report(results, str(Path(__file__).parent / "reports"))
    logger.info(f"报告已保存到: {report_dir}")

    return results


def _generate_daily_signals(df):
    """
    日线级别的简化信号生成（基于技术指标的多维共振）
    不依赖ML模型，仅用规则判断
    """
    import numpy as np

    df = df.copy()
    df["signal"] = "HOLD"
    df["signal_score"] = 0.0
    df["resonance_count"] = 0

    if df.empty or "close" not in df.columns:
        return df

    for i in range(max(60, len(df) // 10), len(df)):
        # === 技术维度 ===
        tech_bull = 0

        # 均线多头
        if all(f"ma_{p}" in df.columns for p in [5, 10, 20]):
            if df["ma_5"].iloc[i] > df["ma_10"].iloc[i] > df["ma_20"].iloc[i]:
                tech_bull += 1

        # MACD金叉
        if "macd_golden_cross" in df.columns and df["macd_golden_cross"].iloc[i]:
            tech_bull += 1

        # RSI健康
        if "rsi" in df.columns and 40 < df["rsi"].iloc[i] < 70:
            tech_bull += 1

        # 放量
        if "volume_surge" in df.columns and df["volume_surge"].iloc[i]:
            tech_bull += 1

        # 不破布林下轨
        if "boll_pct_b" in df.columns and df["boll_pct_b"].iloc[i] > 0.2:
            tech_bull += 1

        tech_score = tech_bull / 5.0

        # === 趋势维度 ===
        trend_bull = 0
        # 短期涨幅
        if "pct_change_5" in df.columns and df["pct_change_5"].iloc[i] > 0:
            trend_bull += 1
        # 价格在MA20上
        if "ma_20" in df.columns and df["close"].iloc[i] > df["ma_20"].iloc[i]:
            trend_bull += 1
        trend_score = trend_bull / 2.0

        # === 综合 ===
        composite = 0.5 * tech_score + 0.3 * trend_score + 0.2 * 0.5  # 其他维度默认中性
        resonance = (tech_bull >= 3) + (trend_bull >= 1)

        df.at[df.index[i], "signal_score"] = round(composite, 3)
        df.at[df.index[i], "resonance_count"] = resonance

        if composite > 0.65 and resonance >= 2:
            df.at[df.index[i], "signal"] = "BUY"
        elif composite < 0.3:
            df.at[df.index[i], "signal"] = "SELL"
        else:
            df.at[df.index[i], "signal"] = "HOLD"

    return df


# ======================== 短线信号（5/15分钟） ========================

def run_intraday_pipeline(data: dict = None):
    """
    分钟级别完整流程：训练 → 信号生成
    """
    logger.info("=" * 60)
    logger.info("开始分钟级别信号流水线...")
    logger.info("=" * 60)

    if data is None:
        data = fetch_all_data()

    # 特征工程
    from features.pipeline import build_full_dataset, prepare_train_data

    dataset = build_full_dataset(
        stock_kline_data=data.get("stock_15min", {}),
        market_indices=data.get("market_15min", {}),
        us_map_data=data.get("us_map", {}),
        us_indices=data.get("us_indices", {}),
        sector_data=data.get("sector_data", {}),
        capital_flow_data=data.get("capital_flow", {}),
        news_data=data.get("news_data", {}),
        tf="15min",
    )

    if dataset["features"].empty:
        logger.error("特征集为空，无法继续")
        return

    # 准备训练数据
    X_train, X_test, y_train, y_test, train_df, test_df = prepare_train_data(dataset)

    if X_train is None:
        logger.error("训练数据准备失败")
        return

    # 训练模型
    from models.train import train_model, save_model

    model, train_results = train_model(X_train, y_train, X_test, y_test)
    save_model(model, "xgboost_latest")

    # 信号生成
    from models.signal import SignalGenerator

    gen = SignalGenerator(model=model)
    test_signals = gen.generate_signals(test_df, dataset["feature_cols"])

    # 回测
    from backtest.engine import BacktestEngine

    engine = BacktestEngine()
    bt_results = engine.run(test_signals[test_signals["signal"] != "HOLD"])
    engine.print_summary()

    return {
        "train_results": train_results,
        "backtest_results": bt_results,
        "dataset": dataset,
        "signals": test_signals,
    }


# ======================== 最新信号 ========================

def generate_latest_signals(data: dict = None):
    """生成最新的交易信号（用于盘中决策）"""
    logger.info("生成最新交易信号...")

    if data is None:
        data = fetch_all_data()

    from features.pipeline import build_full_dataset
    from models.train import load_model
    from models.signal import create_signal_generator

    dataset = build_full_dataset(
        stock_kline_data=data.get("stock_15min", {}),
        market_indices=data.get("market_15min", {}),
        us_map_data=data.get("us_map", {}),
        us_indices=data.get("us_indices", {}),
        sector_data=data.get("sector_data", {}),
        capital_flow_data=data.get("capital_flow", {}),
        news_data=data.get("news_data", {}),
        tf="15min",
    )

    gen = create_signal_generator()

    print("\n" + "=" * 60)
    print("最新交易信号")
    print("=" * 60)

    for code, stock_df in dataset.get("stocks", {}).items():
        if stock_df.empty:
            continue
        signal = gen.get_latest_signal(stock_df, dataset["feature_cols"])
        print(f"\n{signal['symbol']} {signal['stock_name']}")
        print(f"  时间: {signal['datetime']}")
        print(f"  价格: {signal['close']}")
        print(f"  信号: {signal['signal']}")
        print(f"  评分: {signal['score']:.3f}")
        print(f"  置信度: {signal['confidence']:.3f}")
        print(f"  共振维度: {signal['resonance']}/6")

    print("\n" + "=" * 60)


# ======================== 命令行 ========================

def main():
    parser = argparse.ArgumentParser(description="量化交易系统")
    parser.add_argument("command", nargs="?", default="all",
                       choices=["fetch", "train", "backtest", "signal", "all"],
                       help="运行模式")
    parser.add_argument("--years", type=int, default=5,
                       help="回测年数 (默认5年)")
    parser.add_argument("--tf", type=str, default="15min",
                       choices=["5min", "15min", "daily"],
                       help="时间框架")
    parser.add_argument("--symbol", type=str, default="",
                       help="指定股票代码 (逗号分隔)")

    args = parser.parse_args()

    if args.symbol:
        import config as _cfg
        custom = [s.strip() for s in args.symbol.split(",")]
        _cfg.STOCK_CODES = custom

    print("=" * 60)
    print("  量化交易系统 v1.0")
    print("  标的:", ", ".join(f"{c}({STOCK_POOL.get(c, {}).get('name', '')})" for c in STOCK_CODES))
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if args.command == "fetch":
        fetch_all_data()

    elif args.command == "train":
        data = fetch_all_data()
        run_intraday_pipeline(data)

    elif args.command == "backtest":
        if args.tf == "daily":
            run_daily_backtest(years=args.years)
        else:
            data = fetch_all_data()
            run_intraday_pipeline(data)

    elif args.command == "signal":
        generate_latest_signals()

    elif args.command == "all":
        data = fetch_all_data()
        logger.info("\n" + "=" * 60)
        logger.info("第一步：日线回测 (3-5年)")
        logger.info("=" * 60)
        daily_results = run_daily_backtest(data, years=args.years)

        logger.info("\n" + "=" * 60)
        logger.info("第二步：分钟线信号流水线")
        logger.info("=" * 60)
        intraday_results = run_intraday_pipeline(data)

        logger.info("\n" + "=" * 60)
        logger.info("第三步：最新信号")
        logger.info("=" * 60)
        generate_latest_signals(data)

        logger.info("\n全部完成!")


if __name__ == "__main__":
    main()
