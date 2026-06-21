"""
MA双周期共振策略 — 5分钟均线入场 + 15分钟均线定方向
纯均线系统，专注买卖点信号，不依赖外部数据源

核心理念:
  15分钟周期 → 定方向（牛市只做多，熊市不做/只做空）
  5分钟周期  → 找买卖点（MA金叉买、死叉卖、成交量确认）

用法:
  python strategy_ma.py backtest   # 回测
  python strategy_ma.py signal     # 最新信号
  python strategy_ma.py live       # 实时监控（单次）
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
import io

# 解决Windows GBK终端编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from config import STOCK_CODES, STOCK_POOL, TECH_PARAMS, RISK_PARAMS, BACKTEST_PARAMS
from data.baostock_fetcher import fetch_minute_with_cache, fetch_daily_with_cache

logger = logging.getLogger(__name__)

# ======================== 策略参数 ========================

MA_STRATEGY_CONFIG = {
    # 15min 方向判定
    "trend_tf": "15",           # 方向周期
    "trend_ma_fast": 5,         # 15min 快线
    "trend_ma_mid": 10,         # 15min 中线
    "trend_ma_slow": 20,        # 15min 慢线
    "trend_ma_very_slow": 60,   # 15min 长期均线（强支撑/压力）

    # 5min 入场判定
    "entry_tf": "5",            # 入场周期
    "entry_ma_fast": 5,         # 5min 快线
    "entry_ma_mid": 10,         # 5min 中线
    "entry_ma_slow": 20,        # 5min 慢线

    # 信号阈值
    "ma_alignment_score_min": 0.6,    # 15min均线多头排列最低分
    "volume_ratio_min": 1.2,          # 放量最低倍率
    "rsi_buy_zone": (30, 65),         # 买入RSI区间
    "rsi_sell_zone": (70, 100),       # 卖出RSI区间
    "ma_distance_max": 0.03,          # 价格偏离MA20最大距离（太远不追）
    "min_signal_interval": 3,         # 最小信号间隔（K线数）

    # 回测
    "backtest_days": 60,         # 分钟数据回看天数
    "hold_max_bars": 48,         # 最大持仓K线数（5min*48=4小时）

    # 买卖评分权重
    "score_weights": {
        "ma_crossover": 0.35,     # 金叉权重
        "ma_alignment": 0.25,     # 均线排列权重
        "volume": 0.20,           # 成交量确认权重
        "rsi": 0.10,              # RSI权重
        "price_position": 0.10,   # 价格位置权重
    },
}


class MAStrategy:
    """MA双周期共振策略"""

    def __init__(self, config: dict = None):
        self.cfg = MA_STRATEGY_CONFIG.copy()
        if config:
            self.cfg.update(config)

    # ==================== 均线计算 ====================

    def compute_mas(self, df: pd.DataFrame, periods: list) -> pd.DataFrame:
        """计算多条均线"""
        df = df.copy()
        for p in periods:
            col = f"ma_{p}"
            if col not in df.columns:
                df[col] = df["close"].rolling(window=p).mean()
        return df

    def compute_all_mas(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """根据周期计算对应的均线组"""
        if tf == "15":
            periods = [
                self.cfg["trend_ma_fast"],
                self.cfg["trend_ma_mid"],
                self.cfg["trend_ma_slow"],
                self.cfg["trend_ma_very_slow"],
            ]
        else:
            periods = [
                self.cfg["entry_ma_fast"],
                self.cfg["entry_ma_mid"],
                self.cfg["entry_ma_slow"],
            ]
        return self.compute_mas(df, periods)

    # ==================== 15分钟方向判定 ====================

    def score_trend_direction(self, row: pd.Series) -> dict:
        """
        15分钟均线方向评分
        返回: {direction: 'bull'|'bear'|'neutral', score: 0~1, details: str}
        """
        ma5 = row.get(f"ma_{self.cfg['trend_ma_fast']}", np.nan)
        ma10 = row.get(f"ma_{self.cfg['trend_ma_mid']}", np.nan)
        ma20 = row.get(f"ma_{self.cfg['trend_ma_slow']}", np.nan)
        ma60 = row.get(f"ma_{self.cfg['trend_ma_very_slow']}", np.nan)

        if any(np.isnan(x) for x in [ma5, ma10, ma20]):
            return {"direction": "neutral", "score": 0.5, "details": "均线数据不足"}

        score = 0.5
        details = []

        # 完全多头排列: MA5 > MA10 > MA20 → 最强
        if ma5 > ma10 > ma20:
            score += 0.30
            details.append("完全多头排列")

            # 价格在MA60之上添加强势
            if not np.isnan(ma60) and row["close"] > ma60:
                score += 0.10
                details.append("站上MA60")
        # 部分多头
        elif ma5 > ma10 and ma10 > ma20 * 0.99:
            score += 0.15
            details.append("短中期多头")
        elif ma5 > ma20:
            score += 0.05
            details.append("仅短期在长期之上")

        # 完全空头排列: MA5 < MA10 < MA20
        if ma5 < ma10 < ma20:
            score -= 0.30
            details.append("完全空头排列")
            if not np.isnan(ma60) and row["close"] < ma60:
                score -= 0.10
                details.append("跌破MA60")
        elif ma5 < ma10 and ma10 < ma20 * 1.01:
            score -= 0.15
            details.append("短中期空头")

        # 均线粘合度（标准差小 → 可能变盘）
        mas = [ma5, ma10, ma20]
        ma_std = np.std(mas) / np.mean(mas) if np.mean(mas) > 0 else 1
        if ma_std < 0.005:
            details.append("均线粘合(可能变盘)")
            score = 0.5 + (score - 0.5) * 0.5  # 粘合时向中性靠拢

        # 均线发散度加分
        if ma_std > 0.02 and score > 0.6:
            score += 0.05
            details.append("均线发散")

        score = max(0.0, min(1.0, score))

        if score >= 0.65:
            direction = "bull"
        elif score <= 0.35:
            direction = "bear"
        else:
            direction = "neutral"

        return {
            "direction": direction,
            "score": round(score, 3),
            "details": "; ".join(details) if details else "均线中性",
        }

    # ==================== 5分钟入场判定 ====================

    def score_entry_signal(self, row: pd.Series, prev_row: pd.Series,
                           trend_direction: str) -> dict:
        """
        5分钟入场评分
        结合15min方向，给出买入/卖出评分
        """
        fast = self.cfg["entry_ma_fast"]   # 5
        mid = self.cfg["entry_ma_mid"]     # 10
        slow = self.cfg["entry_ma_slow"]   # 20

        ma_fast = row.get(f"ma_{fast}", np.nan)
        ma_mid = row.get(f"ma_{mid}", np.nan)
        ma_slow = row.get(f"ma_{slow}", np.nan)
        prev_ma_fast = prev_row.get(f"ma_{fast}", np.nan)
        prev_ma_mid = prev_row.get(f"ma_{mid}", np.nan)

        if any(np.isnan(x) for x in [ma_fast, ma_mid, ma_slow,
                                       prev_ma_fast, prev_ma_mid]):
            return {"signal": "HOLD", "score": 0.5, "reasons": []}

        w = self.cfg["score_weights"]
        reasons = []
        total_score = 0.50

        # === 1. 金叉/死叉检测 ===
        golden_cross = (prev_ma_fast <= prev_ma_mid) and (ma_fast > ma_mid)
        dead_cross = (prev_ma_fast >= prev_ma_mid) and (ma_fast < ma_mid)

        if golden_cross:
            total_score += w["ma_crossover"]
            reasons.append(f"MA{fast}上穿MA{mid}金叉")
        elif dead_cross:
            total_score -= w["ma_crossover"]
            reasons.append(f"MA{fast}下穿MA{mid}死叉")

        # === 2. 均线排列 ===
        if ma_fast > ma_mid > ma_slow:
            total_score += w["ma_alignment"]
            reasons.append("5min均线多头排列")
        elif ma_fast < ma_mid < ma_slow:
            total_score -= w["ma_alignment"]
            reasons.append("5min均线空头排列")
        elif ma_fast > ma_mid and ma_mid < ma_slow:
            # 快线上穿中线，但还没超过慢线 → 弱多
            total_score += w["ma_alignment"] * 0.5
            reasons.append("5min均线部分转多")
        elif ma_fast < ma_mid and ma_mid > ma_slow:
            total_score -= w["ma_alignment"] * 0.5
            reasons.append("5min均线部分转空")

        # === 3. 成交量确认 ===
        vol_ratio = row.get("volume_ratio", 1.0)
        if not np.isnan(vol_ratio) and vol_ratio > self.cfg["volume_ratio_min"]:
            if total_score > 0.55:
                total_score += w["volume"]
                reasons.append(f"放量确认 (量比{vol_ratio:.1f})")
        if vol_ratio < 0.5:
            total_score -= w["volume"] * 0.5
            reasons.append("缩量(信号减弱)")

        # === 4. RSI过滤 ===
        rsi = row.get("rsi", 50)
        if not np.isnan(rsi):
            buy_low, buy_high = self.cfg["rsi_buy_zone"]
            sell_low, sell_high = self.cfg["rsi_sell_zone"]

            if buy_low < rsi < buy_high and total_score > 0.55:
                total_score += w["rsi"]
                reasons.append(f"RSI={rsi:.0f}在买入区")
            elif rsi > 70:
                total_score -= w["rsi"]
                reasons.append(f"RSI={rsi:.0f}超买")
            elif rsi < 30:
                total_score += w["rsi"] * 0.5
                reasons.append(f"RSI={rsi:.0f}超卖(潜在反弹)")
            elif rsi > rsi + 10:  # RSI上升中
                pass  # 微利好

        # === 5. 价格与MA20关系 ===
        dist_to_ma20 = (row["close"] / ma_slow - 1) if ma_slow > 0 else 0
        if abs(dist_to_ma20) < self.cfg["ma_distance_max"]:
            if dist_to_ma20 > 0:
                total_score += w["price_position"] * 0.7
                reasons.append(f"价格在MA{slow}上方(强势)")
            elif dist_to_ma20 < -0.01:
                total_score -= w["price_position"] * 0.3
                reasons.append(f"价格在MA{slow}下方(弱势)")
        elif dist_to_ma20 > self.cfg["ma_distance_max"]:
            reasons.append(f"偏离MA{slow}过远({dist_to_ma20*100:.1f}%)不宜追")

        # === 6. 15min方向叠加 ===
        if trend_direction == "bull":
            total_score += 0.05  # 顺大势加分
        elif trend_direction == "bear":
            total_score -= 0.05  # 逆大势减分

        total_score = max(0.0, min(1.0, total_score))

        # 信号判定
        if total_score >= 0.68:
            signal = "BUY"
        elif total_score <= 0.32:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "signal": signal,
            "score": round(total_score, 3),
            "reasons": reasons,
        }

    # ==================== 完整信号生成 ====================

    def generate_signals(self, df_5min: pd.DataFrame,
                         df_15min: pd.DataFrame) -> pd.DataFrame:
        """
        生成完整交易信号
        5分钟K线逐根判断，15分钟方向实时映射
        """
        if df_5min.empty:
            logger.warning("5分钟数据为空")
            return pd.DataFrame()

        # 计算均线
        df5 = self.compute_all_mas(df_5min.copy(), "5")
        df15 = self.compute_all_mas(df_15min.copy(), "15")

        # 计算辅助指标
        df5["rsi"] = self._calc_rsi(df5["close"])
        df5["volume_ma_5"] = df5["volume"].rolling(5).mean()
        df5["volume_ratio"] = df5["volume"] / df5["volume_ma_5"].replace(0, np.nan)
        df5["atr"] = self._calc_atr(df5)

        # 为每根5min K线找对应的15min方向
        # 15min数据时间对齐：每根15min K线覆盖3根5min K线
        trend_cache = {}

        def get_trend(dt_5min):
            # 向下取整到15分钟
            minute_block = (dt_5min.minute // 15) * 15
            trend_time = dt_5min.replace(minute=minute_block, second=0, microsecond=0)

            if trend_time in trend_cache:
                return trend_cache[trend_time]

            # 找最接近的15min K线
            row_15 = df15[df15["datetime"] <= dt_5min]
            if row_15.empty:
                return {"direction": "neutral", "score": 0.5, "details": ""}
            trend = self.score_trend_direction(row_15.iloc[-1])
            trend_cache[trend_time] = trend
            return trend

        # 逐根5min K线打分
        signals = []
        last_signal_idx = -999

        for i in range(1, len(df5)):
            row = df5.iloc[i]
            prev_row = df5.iloc[i - 1]

            trend = get_trend(row["datetime"])
            entry = self.score_entry_signal(row, prev_row, trend["direction"])

            # 信号间隔限制
            if entry["signal"] != "HOLD" and i - last_signal_idx < self.cfg["min_signal_interval"]:
                entry["signal"] = "HOLD"
                entry["score"] = 0.5
                entry["reasons"] = ["信号间隔不足"]

            if entry["signal"] != "HOLD":
                last_signal_idx = i

            signals.append({
                "datetime": row["datetime"],
                "symbol": row.get("symbol", ""),
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "ma5_5min": round(row[f"ma_{self.cfg['entry_ma_fast']}"], 3),
                "ma10_5min": round(row[f"ma_{self.cfg['entry_ma_mid']}"], 3),
                "ma20_5min": round(row[f"ma_{self.cfg['entry_ma_slow']}"], 3),
                "trend_direction": trend["direction"],
                "trend_score": trend["score"],
                "signal": entry["signal"],
                "signal_score": entry["score"],
                "rsi": round(row["rsi"], 1) if not np.isnan(row["rsi"]) else None,
                "volume_ratio": round(row["volume_ratio"], 1) if not np.isnan(row["volume_ratio"]) else None,
                "reasons": "; ".join(entry["reasons"]),
            })

        return pd.DataFrame(signals)

    # ==================== 回测 ====================

    def backtest(self, signals_df: pd.DataFrame,
                 capital: float = 50000) -> dict:
        """
        基于5min信号的回测
        模拟T+1约束（当天买次日才能卖）
        """
        if signals_df.empty:
            return {"total_return": 0, "win_rate": 0, "trades": []}

        df = signals_df.copy()
        df["date"] = df["datetime"].dt.date

        positions = {}  # {date: [{entry_price, shares}]}
        trades = []
        cash = capital
        equity = capital

        for i, row in df.iterrows():
            date = row["date"]
            signal = row["signal"]
            price = row["close"]

            # 检查是否有需要卖出的持仓（T+1已过）
            to_sell = []
            for entry_date, pos_list in list(positions.items()):
                if entry_date < date:  # 隔日了，可卖
                    to_sell.append((entry_date, pos_list))

            for entry_date, pos_list in to_sell:
                for pos in pos_list:
                    if row["trend_direction"] == "bear" or signal == "SELL":
                        pnl = (price - pos["entry_price"]) * pos["shares"]
                        pnl_pct = (price / pos["entry_price"] - 1) * 100

                        # 扣费
                        cost = price * pos["shares"] * (BACKTEST_PARAMS["commission_rate"] +
                                                        BACKTEST_PARAMS["stamp_tax_rate"] +
                                                        BACKTEST_PARAMS["slippage"])
                        pnl -= cost
                        cash += price * pos["shares"] - cost

                        trades.append({
                            "entry_date": str(entry_date),
                            "exit_date": str(date),
                            "entry_price": round(pos["entry_price"], 3),
                            "exit_price": round(price, 3),
                            "shares": pos["shares"],
                            "pnl": round(pnl, 2),
                            "pnl_pct": round(pnl_pct, 2),
                            "exit_reason": "卖出信号" if signal == "SELL" else "趋势转空",
                        })
                del positions[entry_date]

            # 止损检查
            for entry_date, pos_list in list(positions.items()):
                for pos in pos_list[:]:
                    loss_pct = (price / pos["entry_price"] - 1)
                    if loss_pct <= -RISK_PARAMS["stop_loss_pct"]:
                        pnl = loss_pct * pos["entry_price"] * pos["shares"]
                        cost = price * pos["shares"] * (BACKTEST_PARAMS["commission_rate"] +
                                                        BACKTEST_PARAMS["stamp_tax_rate"] +
                                                        BACKTEST_PARAMS["slippage"])
                        pnl -= cost
                        cash += price * pos["shares"] - cost

                        trades.append({
                            "entry_date": str(entry_date),
                            "exit_date": str(date),
                            "entry_price": round(pos["entry_price"], 3),
                            "exit_price": round(price, 3),
                            "shares": pos["shares"],
                            "pnl": round(pnl, 2),
                            "pnl_pct": round(loss_pct * 100, 2),
                            "exit_reason": f"止损({loss_pct*100:.1f}%)",
                        })
                        pos_list.remove(pos)

            # 买入信号
            if signal == "BUY" and row["trend_direction"] == "bull":
                # 检查今日是否已买入
                today_bought = sum(
                    sum(p["shares"] * p["entry_price"] for p in pl)
                    for d, pl in positions.items() if d == date
                )
                max_per_stock = capital * BACKTEST_PARAMS["position_pct"]

                if today_bought < max_per_stock:
                    shares = int(cash * BACKTEST_PARAMS["position_pct"] / price / 100) * 100
                    if shares >= 100 and cash >= price * shares * 1.01:  # 留1%缓冲
                        cost = price * shares * BACKTEST_PARAMS["commission_rate"]
                        cash -= (price * shares + cost)

                        if date not in positions:
                            positions[date] = []
                        positions[date].append({
                            "entry_price": price,
                            "shares": shares,
                        })

            # 更新权益
            position_value = sum(
                p["entry_price"] * p["shares"]
                for pl in positions.values() for p in pl
            )
            equity = cash + position_value

        # 收盘强平所有持仓
        for entry_date, pos_list in positions.items():
            for pos in pos_list:
                last_price = df["close"].iloc[-1]
                pnl = (last_price - pos["entry_price"]) * pos["shares"]
                cost = last_price * pos["shares"] * (BACKTEST_PARAMS["commission_rate"] +
                                                     BACKTEST_PARAMS["stamp_tax_rate"] +
                                                     BACKTEST_PARAMS["slippage"])
                pnl -= cost
                cash += last_price * pos["shares"] - cost
                trades.append({
                    "entry_date": str(entry_date),
                    "exit_date": str(df["date"].iloc[-1]),
                    "entry_price": round(pos["entry_price"], 3),
                    "exit_price": round(last_price, 3),
                    "shares": pos["shares"],
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((last_price / pos["entry_price"] - 1) * 100, 2),
                    "exit_reason": "收盘平仓",
                })

        # 统计
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in trades)
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t["pnl"]) for t in losses]) if losses else 0
        profit_factor = sum(t["pnl"] for t in wins) / max(abs(sum(t["pnl"] for t in losses)), 0.01)

        return {
            "initial_capital": capital,
            "final_equity": round(cash, 2),
            "total_return": round((cash / capital - 1) * 100, 2),
            "total_pnl": round(total_pnl, 2),
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "trades": trades,
        }

    # ==================== 辅助函数 ====================

    def _calc_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR"""
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    def print_signals(self, signals_df: pd.DataFrame, n: int = 20):
        """打印最近的信号"""
        if signals_df.empty:
            print("无信号数据")
            return

        buy_sells = signals_df[signals_df["signal"] != "HOLD"].tail(n)
        if buy_sells.empty:
            print("最近无买卖信号")
            # 打印最近几条数据
            print(signals_df.tail(5).to_string(index=False))
            return

        print(f"\n{'='*100}")
        print(f"最近 {len(buy_sells)} 条买卖信号")
        print(f"{'='*100}")
        print(f"{'时间':<20} {'信号':<6} {'评分':<6} {'价格':<8} {'MA5':<8} {'MA10':<8} {'MA20':<8} {'15min方向':<10} {'RSI':<6} {'量比':<6} {'原因'}")
        print(f"{'-'*100}")

        for _, row in buy_sells.iterrows():
            print(f"{str(row['datetime']):<20} "
                  f"{row['signal']:<6} "
                  f"{row['signal_score']:<6.3f} "
                  f"{row['close']:<8.2f} "
                  f"{row.get('ma5_5min', '-'):<8} "
                  f"{row.get('ma10_5min', '-'):<8} "
                  f"{row.get('ma20_5min', '-'):<8} "
                  f"{row['trend_direction']:<10} "
                  f"{row.get('rsi', '-'):<6} "
                  f"{row.get('volume_ratio', '-'):<6} "
                  f"{row['reasons']}")

        print(f"{'='*100}")


# ======================== 命令行入口 ========================

def run_backtest(codes: list = None, days: int = 60):
    """运行MA策略回测"""
    if codes is None:
        codes = STOCK_CODES

    strategy = MAStrategy()
    all_trades = []

    print("=" * 60)
    print("  MA双周期共振策略回测")
    print("  5min入场 + 15min方向")
    print("=" * 60)

    for code in codes:
        name = STOCK_POOL.get(code, {}).get("name", code)
        print(f"\n{'='*60}")
        print(f"  {code} {name}")
        print(f"{'='*60}")

        # 获取5min和15min数据
        print("获取5分钟数据...")
        df5 = fetch_minute_with_cache(code, "5", days=days)
        print("获取15分钟数据...")
        df15 = fetch_minute_with_cache(code, "15", days=days)

        if df5.empty or df15.empty:
            print(f"  {code}: 数据不足，跳过")
            continue

        print(f"  5min: {len(df5)}条  15min: {len(df15)}条")

        # 生成信号
        signals = strategy.generate_signals(df5, df15)
        if signals.empty:
            print(f"  {code}: 信号生成为空")
            continue

        # 回测
        results = strategy.backtest(signals)

        print(f"\n  --- 回测结果 ---")
        print(f"  初始资金: ¥{results['initial_capital']:,.0f}")
        print(f"  最终权益: ¥{results['final_equity']:,.0f}")
        print(f"  总收益率: {results['total_return']:.2f}%")
        print(f"  交易次数: {results['total_trades']}")
        print(f"  胜率: {results['win_rate']:.1f}%")
        print(f"  平均盈利: ¥{results['avg_win']:,.2f}")
        print(f"  平均亏损: ¥{results['avg_loss']:,.2f}")
        print(f"  盈亏比: {results['profit_factor']:.2f}")

        # 最近信号
        strategy.print_signals(signals, n=15)

        for t in results["trades"]:
            t["symbol"] = code
            t["name"] = name
        all_trades.extend(results["trades"])

    # 汇总
    if all_trades:
        wins = [t for t in all_trades if t["pnl"] > 0]
        total_pnl = sum(t["pnl"] for t in all_trades)
        print(f"\n{'='*60}")
        print(f"  汇总 (全部{len(codes)}只股票)")
        print(f"{'='*60}")
        print(f"  总交易: {len(all_trades)}")
        print(f"  总盈亏: ¥{total_pnl:,.2f}")
        print(f"  胜率: {len(wins)/len(all_trades)*100:.1f}%")
        print(f"  平均盈利: ¥{np.mean([t['pnl'] for t in wins]):,.2f}" if wins else "  无盈利交易")

        # 保存结果
        report_path = Path(__file__).parent / "reports" / f"ma_strategy_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # 简化trades以便JSON序列化
        serializable = {k: v for k, v in results.items() if k != "trades"}
        serializable["trades"] = all_trades
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n报告已保存: {report_path}")


def run_latest_signals(codes: list = None):
    """生成最新交易信号"""
    if codes is None:
        codes = STOCK_CODES

    strategy = MAStrategy()

    print("=" * 80)
    print("  MA双周期共振 — 最新交易信号")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    for code in codes:
        name = STOCK_POOL.get(code, {}).get("name", code)

        # 获取数据
        df5 = fetch_minute_with_cache(code, "5", days=10)
        df15 = fetch_minute_with_cache(code, "15", days=10)

        if df5.empty or df15.empty:
            print(f"\n{code} {name}: 数据不足")
            continue

        signals = strategy.generate_signals(df5, df15)
        if signals.empty:
            print(f"\n{code} {name}: 无信号")
            continue

        # 最新一条非HOLD信号
        recent = signals[signals["signal"] != "HOLD"].tail(3)
        latest = signals.iloc[-1]

        print(f"\n--- {code} {name} ---")
        print(f"  最新K线: {latest['datetime']}  价格: {latest['close']:.2f}")
        print(f"  15min方向: {latest['trend_direction']} (评分{latest['trend_score']:.3f})")
        print(f"  5min MA5={latest['ma5_5min']} MA10={latest['ma10_5min']} MA20={latest['ma20_5min']}")
        print(f"  RSI={latest['rsi']}  量比={latest['volume_ratio']}")

        if not recent.empty:
            print(f"\n  最近信号:")
            for _, s in recent.iterrows():
                icon = "[BUY]" if s["signal"] == "BUY" else "[SELL]" if s["signal"] == "SELL" else "[HOLD]"
                print(f"    {icon} {s['datetime']} {s['signal']} 评分{s['signal_score']:.3f} "
                      f"@{s['close']:.2f} → {s['reasons']}")
        else:
            print(f"  当前无买卖信号(HOLD)")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cmd = sys.argv[1] if len(sys.argv) > 1 else "signal"

    if cmd == "backtest":
        run_backtest(days=int(sys.argv[2]) if len(sys.argv) > 2 else 60)
    elif cmd == "signal":
        run_latest_signals()
    elif cmd == "live":
        run_latest_signals()
    else:
        print("用法: python strategy_ma.py [backtest|signal|live] [回看天数]")
