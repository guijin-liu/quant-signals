"""
定时信号推送模块 — 每天自动扫描量化信号并推送到微信

运行方式:
  python signal_pusher.py           # 立即扫描一次并推送
  python signal_pusher.py --watch   # 持续监控,每15分钟扫描
  python signal_pusher.py --daily 09:00  # 每天9点推送信号
"""

import sys
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta
import json

sys.path.insert(0, str(Path(__file__).parent))

from push_notify import push_trade_signal, push_signal_summary, push_alert, push_msg
from config import STOCK_CODES, STOCK_POOL

logger = logging.getLogger(__name__)

# 交易时间
TRADING_START = "09:30"
TRADING_END = "15:00"


def scan_and_push():
    """扫描信号并推送"""
    from data.baostock_fetcher import fetch_minute_with_cache
    from features.technical import compute_all_technical_features

    signals = []
    now = datetime.now()
    time_str = now.strftime("%H:%M")

    logger.info(f"扫描信号... {now.strftime('%Y-%m-%d %H:%M:%S')}")

    for code in STOCK_CODES:
        try:
            # 获取最新15分钟数据
            df = fetch_minute_with_cache(code, "15", days=5, ttl_minutes=5)
            if df.empty or len(df) < 50:
                logger.warning(f"  {code}: 数据不足")
                signals.append({"code": code, "name": STOCK_POOL[code]["name"],
                              "signal": "HOLD", "close": 0, "score": 0, "resonance": 0})
                continue

            # 计算技术指标
            df = compute_all_technical_features(df, "15min")
            if df.empty:
                signals.append({"code": code, "name": STOCK_POOL[code]["name"],
                              "signal": "HOLD", "close": 0, "score": 0, "resonance": 0})
                continue

            latest = df.iloc[-1]
            close = latest.get("close", 0)
            if close <= 0 and len(df) > 1:
                close = df["close"].iloc[-2]

            # === 快速多维评分 ===
            score = 0.5
            reasons = []

            # 技术维度: MA多头 + MACD + RSI
            tech_score = 0.0
            if all(f"ma_{p}" in df.columns for p in [5, 10, 20]):
                if latest.get("close", 0) > latest.get("ma_5", 0) > latest.get("ma_10", 0):
                    tech_score += 0.3
                    reasons.append("MA多头")
            if "macd_golden_cross" in df.columns and latest.get("macd_golden_cross"):
                tech_score += 0.2
                reasons.append("MACD金叉")
            if "rsi" in df.columns and 40 < latest.get("rsi", 50) < 70:
                tech_score += 0.15
                reasons.append("RSI健康")
            if "boll_pct_b" in df.columns and 0.2 < latest.get("boll_pct_b", 0.5) < 0.9:
                tech_score += 0.15
            if "volume_surge" in df.columns and latest.get("volume_surge"):
                tech_score += 0.1
                reasons.append("放量")
            tech_score = min(tech_score, 1.0)

            # 趋势维度: 近5根K线方向
            trend_score = 0.0
            if "pct_change" in df.columns:
                recent = df["pct_change"].tail(5)
                if recent.mean() > 0.1:
                    trend_score += 0.3
                    reasons.append("短线向上")
            if "close" in df.columns and "ma_20" in df.columns:
                if latest.get("close", 0) > latest.get("ma_20", 0):
                    trend_score += 0.2

            # 综合
            signal_score = 0.5 * tech_score + 0.3 * trend_score + 0.2 * 0.5
            resonance = len(reasons)

            # 判定
            if signal_score > 0.65 and resonance >= 3:
                signal = "BUY"
            elif signal_score < 0.35:
                signal = "SELL"
            else:
                signal = "HOLD"

            name = STOCK_POOL[code]["name"]
            signals.append({
                "code": code, "name": name, "signal": signal,
                "close": close, "score": signal_score,
                "confidence": tech_score, "resonance": resonance,
            })

            reason_str = " + ".join(reasons) if reasons else ""
            logger.info(f"  {code} {name}: {signal} | price={close:.2f} score={signal_score:.3f} resonance={resonance}/6 {reason_str}")

        except Exception as e:
            logger.error(f"  {code} 扫描失败: {e}")
            signals.append({"code": code, "name": STOCK_POOL[code]["name"],
                          "signal": "HOLD", "close": 0, "score": 0, "resonance": 0})

    # 推送汇总
    buy_signals = [s for s in signals if s["signal"] == "BUY"]
    sell_signals = [s for s in signals if s["signal"] == "SELL"]

    push_signal_summary(signals)

    # 有买入信号时单独推送详情
    if buy_signals:
        for s in buy_signals:
            push_trade_signal(
                s["code"], s["name"], "BUY", s["close"],
                s["score"], s["confidence"], s["resonance"],
                f"时间: {time_str}"
            )
            time.sleep(0.3)  # 避免推送太快

    if sell_signals:
        for s in sell_signals:
            push_trade_signal(
                s["code"], s["name"], "SELL", s["close"],
                s["score"], s["confidence"], s["resonance"],
                f"时间: {time_str}"
            )
            time.sleep(0.3)

    logger.info(f"推送完成: {len(buy_signals)}买入 {len(sell_signals)}卖出 {len(signals)-len(buy_signals)-len(sell_signals)}持有")
    return signals


def watch_mode(interval_minutes: int = 15):
    """持续监控模式 - 每个交易日定时扫描"""
    import baostock as bs
    bs.login()

    logger.info(f"启动监控模式, 每{interval_minutes}分钟扫描一次")
    push_msg("🟢 量化监控已启动",
             f"<p>每<b>{interval_minutes}分钟</b>自动扫描信号</p>"
             f"<p>股票池: {', '.join(STOCK_POOL[c]['name'] for c in STOCK_CODES)}</p>")

    signals_pushed = set()

    try:
        while True:
            now = datetime.now()
            time_str = now.strftime("%H:%M")
            weekday = now.weekday()

            # 周末跳过
            if weekday >= 5:
                logger.info(f"周末跳过 ({time_str})")
                time.sleep(3600)
                continue

            # 盘前 9:00 推送一次
            if "09:00" <= time_str <= "09:05":
                key = now.strftime("%Y%m%d")
                if key not in signals_pushed:
                    signals_pushed.add(key)
                    push_msg("📅 今日开盘提醒",
                             f"<h3>量化系统已就绪</h3><p>股票池: 神火股份 雅化集团 锡业股份 亚钾国际</p>"
                             f"<p>时间框架: 15min日内短线</p><p>目标胜率: >88%</p>")

            # 交易时间扫描
            if TRADING_START <= time_str <= TRADING_END:
                scan_and_push()
            else:
                logger.info(f"非交易时间 ({time_str})")

            time.sleep(interval_minutes * 60)

    except KeyboardInterrupt:
        logger.info("监控已停止")
        push_alert("量化监控已停止", "warning")
    finally:
        try:
            bs.logout()
        except:
            pass


# ========== 命令行 ==========

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("push_notify").setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="量化信号推送")
    parser.add_argument("--watch", action="store_true", help="持续监控")
    parser.add_argument("--daily", type=str, default="", help="定时推送 (如: 09:00)")
    parser.add_argument("--interval", type=int, default=15, help="扫描间隔(分钟) 默认15")
    args = parser.parse_args()

    print("=" * 50)
    print("  量化信号 → PushPlus → 微信推送")
    print("=" * 50)

    if args.watch:
        watch_mode(args.interval)
    else:
        scan_and_push()
