# 量化交易自动化流水线
# 用法: python auto_pipeline.py [daily|weekly|full]
"""
每日自动化流水线:
  python auto_pipeline.py daily    → 快速日线信号扫描
  python auto_pipeline.py weekly   → 完整回测+参数优化
  python auto_pipeline.py full     → 全流程(数据+回测+信号+报告)
"""
import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "auto_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("auto")

from config import STOCK_CODES, STOCK_POOL


def daily_quick_scan():
    """快速日线扫描 - 只获取最新数据+信号"""
    logger.info("=" * 50)
    logger.info(f"每日快速扫描 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 50)

    from data.baostock_fetcher import fetch_daily_kline
    from features.technical import compute_all_technical_features
    import pandas as pd
    import numpy as np

    signals = []
    for code in STOCK_CODES:
        name = STOCK_POOL[code]['name']
        df = fetch_daily_kline(code, years=1)
        if df.empty:
            continue

        df = compute_all_technical_features(df, "daily")
        if df.empty:
            continue

        latest = df.iloc[-1]
        signal_info = _analyze_latest(code, name, latest, df)
        signals.append(signal_info)

    # 汇总输出
    print("\n" + "=" * 60)
    print(f"  每日信号扫描 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    for s in signals:
        emoji = "[BUY]" if s['bias'] == 'BULL' else ("[SELL]" if s['bias'] == 'BEAR' else "[HOLD]")
        print(f"\n{emoji} {s['code']} {s['name']} - {s['bias']} ({s['score']:.0f}分)")
        print(f"   收盘价: {s['close']:.2f}  |  MA5: {s['ma5']:.2f}  |  MA20: {s['ma20']:.2f}")
        print(f"   RSI: {s['rsi']:.0f}  |  MACD: {s['macd_signal']}  |  量比: {s['vol_ratio']:.2f}")
        print(f"   共振维度: {s['resonance']}/2  |  建议: {s['suggestion']}")
    print("\n" + "=" * 60)

    return signals


def _analyze_latest(code, name, latest, df):
    """分析最新一根K线"""
    import numpy as np

    close = float(latest.get('close', 0))
    ma5 = float(latest.get('ma_5', close))
    ma20 = float(latest.get('ma_20', close))
    rsi = float(latest.get('rsi', 50))
    vol_ratio = float(latest.get('volume_ratio', 1.0))

    # 信号判断
    bull_signals = 0
    if close > ma5 > ma20:  # 均线多头
        bull_signals += 1
    if 40 < rsi < 70:  # RSI健康
        bull_signals += 0.5
    if vol_ratio > 1.2:  # 放量
        bull_signals += 0.5
    if 'macd_golden_cross' in latest and latest['macd_golden_cross']:
        bull_signals += 1

    score = bull_signals / 3.0 * 100
    if score > 65:
        bias = 'BULL'
        suggestion = '关注做多机会'
    elif score < 30:
        bias = 'BEAR'
        suggestion = '观望/减仓'
    else:
        bias = 'NEUTRAL'
        suggestion = '等待信号'

    return {
        'code': code,
        'name': name,
        'close': close,
        'ma5': ma5,
        'ma20': ma20,
        'rsi': rsi,
        'vol_ratio': vol_ratio,
        'macd_signal': '金叉' if ('macd_golden_cross' in latest and latest['macd_golden_cross']) else '死叉/观望',
        'resonance': int(bull_signals),
        'score': score,
        'bias': bias,
        'suggestion': suggestion,
    }


def weekly_full_report():
    """周度完整报告"""
    logger.info("=" * 50)
    logger.info(f"周度完整回测 - {datetime.now().strftime('%Y-%m-%d')}")
    logger.info("=" * 50)

    import subprocess
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "main.py"), "backtest", "--years", "3", "--tf", "daily"],
        capture_output=True, text=True, timeout=300
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-500:])

    # 生成参数优化建议
    logger.info("\n参数优化建议:")
    logger.info("  1. 如果胜率<50%：提高 buy_threshold 或 resonance_required")
    logger.info("  2. 如果回撤>10%：降低 position_pct 或提高 stop_loss")
    logger.info("  3. 如果交易太少：降低 buy_threshold 或 resonance_required")
    logger.info("  4. 如果盈亏比<1.5：收紧止损或放宽止盈")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="daily",
                       choices=["daily", "weekly", "full"])
    args = parser.parse_args()

    if args.mode == "daily":
        daily_quick_scan()
    elif args.mode == "weekly":
        weekly_full_report()
    elif args.mode == "full":
        daily_quick_scan()
        weekly_full_report()

    logger.info("流水线完成")
