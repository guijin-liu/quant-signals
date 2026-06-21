"""
回测引擎 - 5分钟/15分钟短线级别
支持 T+1 约束、手续费、滑点模拟
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from config import BACKTEST_PARAMS, RISK_PARAMS, STOCK_POOL

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    日内短线回测引擎
    基于信号DataFrame逐K线模拟交易
    """

    def __init__(self, params: dict = None):
        p = BACKTEST_PARAMS.copy()
        if params:
            p.update(params)

        self.initial_capital = p["initial_capital"]
        self.commission_rate = p["commission_rate"]
        self.stamp_tax_rate = p["stamp_tax_rate"]
        self.transfer_fee_rate = p["transfer_fee_rate"]
        self.slippage = p["slippage"]
        self.min_hold_bars = p["min_hold_bars"]
        self.position_pct = p["position_pct"]
        self.max_positions = p["max_positions"]
        self.max_daily_trades = p["max_daily_trades"]

        # 风控参数
        self.stop_loss = RISK_PARAMS["stop_loss_pct"]
        self.take_profit = RISK_PARAMS["take_profit_pct"]

        # 状态
        self.capital = self.initial_capital
        self.positions = {}   # {symbol: {entry_price, shares, entry_bar, entry_date, ...}}
        self.trades = []      # 已完成交易记录
        self.equity_curve = []  # 资金曲线

    def reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

    def _calculate_cost(self, price: float, shares: int, is_sell: bool) -> float:
        """计算交易成本"""
        amount = price * shares
        commission = amount * self.commission_rate
        commission = max(commission, 5)  # 最低佣金5元
        stamp_tax = amount * self.stamp_tax_rate if is_sell else 0
        transfer_fee = amount * self.transfer_fee_rate
        slippage_cost = amount * self.slippage
        return commission + stamp_tax + transfer_fee + slippage_cost

    def _can_buy(self, price: float, current_date) -> tuple:
        """
        检查是否可以买入
        返回 (can_buy, max_shares)
        """
        # 仓位检查
        if len(self.positions) >= self.max_positions:
            return False, 0

        # 日交易次数限制
        today_trades = sum(1 for t in self.trades if t["entry_date"] == current_date)
        if today_trades >= self.max_daily_trades:
            return False, 0

        # 可用资金
        available = self.capital * self.position_pct
        if available < 10000:  # 至少1万
            return False, 0

        # 计算可买股数（100股整数倍）
        max_shares = int(available / price / 100) * 100
        return max_shares >= 100, max_shares

    def _open_position(self, symbol: str, price: float, shares: int, bar_idx: int, bar_date, bar_time) -> dict:
        """开仓"""
        cost = price * shares + self._calculate_cost(price, shares, is_sell=False)
        self.capital -= cost

        pos = {
            "symbol": symbol,
            "entry_price": price,
            "shares": shares,
            "entry_bar": bar_idx,
            "entry_date": bar_date,
            "entry_time": bar_time,
            "cost": cost,
            "high_since_entry": price,
            "low_since_entry": price,
            "atr": 0,
        }
        self.positions[symbol] = pos
        logger.debug(f"开仓 {symbol} @{price:.2f} x{shares}股, 成本={cost:.2f}")
        return pos

    def _close_position(self, symbol: str, price: float, bar_idx: int, bar_date, bar_time, reason: str) -> dict:
        """平仓"""
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None

        revenue = price * pos["shares"]
        cost = self._calculate_cost(price, pos["shares"], is_sell=True)
        self.capital += (revenue - cost)

        pnl = revenue - cost - pos["cost"]
        pnl_pct = pnl / pos["cost"] * 100

        trade = {
            "symbol": symbol,
            "name": STOCK_POOL.get(symbol, {}).get("name", ""),
            "entry_date": pos["entry_date"],
            "entry_time": pos["entry_time"],
            "entry_price": pos["entry_price"],
            "exit_date": bar_date,
            "exit_time": bar_time,
            "exit_price": price,
            "shares": pos["shares"],
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "bars_held": bar_idx - pos["entry_bar"],
            "reason": reason,
        }
        self.trades.append(trade)
        logger.debug(f"平仓 {symbol} @{price:.2f} PnL={pnl:.2f} ({pnl_pct:.2f}%) [{reason}]")
        return trade

    def run(self, signal_df: pd.DataFrame, price_col: str = "close") -> dict:
        """
        执行回测
        signal_df 必须包含列: datetime, symbol, signal, close, (可选) atr
        """
        self.reset()

        df = signal_df.sort_values(["datetime", "symbol"]).reset_index(drop=True)

        # 按时间分组
        df["date"] = df["datetime"].dt.date
        df["time_str"] = df["datetime"].dt.strftime("%H:%M")

        for i, row in df.iterrows():
            symbol = row["symbol"]
            signal = row["signal"]
            price = row.get(price_col, row.get("close", 0))
            bar_date = row["date"]
            bar_time = row["time_str"]

            if price <= 0:
                continue

            # 更新持仓浮动盈亏跟踪
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos["high_since_entry"] = max(pos["high_since_entry"], price)
                pos["low_since_entry"] = min(pos["low_since_entry"], price)
                pos["atr"] = row.get("atr", pos.get("atr", 0))

                # === 止损检查 ===
                loss_pct = (price - pos["entry_price"]) / pos["entry_price"]
                if loss_pct <= -self.stop_loss:
                    self._close_position(symbol, price, i, bar_date, bar_time, f"止损({loss_pct*100:.1f}%)")
                    self._record_equity(i, bar_date, bar_time)
                    continue

                # === 止盈检查 ===
                if loss_pct >= self.take_profit:
                    self._close_position(symbol, price, i, bar_date, bar_time, f"止盈({loss_pct*100:.1f}%)")
                    self._record_equity(i, bar_date, bar_time)
                    continue

                # === ATR追踪止损 ===
                if pos.get("atr", 0) > 0 and price > pos["entry_price"]:
                    trailing_stop = pos["high_since_entry"] - 1.5 * pos["atr"]
                    if price <= trailing_stop:
                        self._close_position(symbol, price, i, bar_date, bar_time, "ATR追踪止损")
                        self._record_equity(i, bar_date, bar_time)
                        continue

            # === 信号处理 ===
            if signal == "BUY" and symbol not in self.positions:
                can_buy, max_shares = self._can_buy(price, bar_date)
                if can_buy:
                    # 按仓位比例计算
                    shares = int(self.capital * self.position_pct / price / 100) * 100
                    shares = min(shares, max_shares)
                    if shares >= 100:
                        self._open_position(symbol, price, shares, i, bar_date, bar_time)

            elif signal == "SELL" and symbol in self.positions:
                self._close_position(symbol, price, i, bar_date, bar_time, "卖出信号")

            # 记录资金曲线
            self._record_equity(i, bar_date, bar_time)

        # === 收盘强制平仓 ===
        for symbol in list(self.positions.keys()):
            last_row = df[df["symbol"] == symbol].iloc[-1]
            price = last_row.get(price_col, last_row.get("close", 0))
            self._close_position(symbol, price, len(df) - 1, last_row["date"], last_row["time_str"], "收盘平仓")

        self._record_equity(len(df) - 1, df["date"].iloc[-1], df["time_str"].iloc[-1])

        logger.info(f"回测完成: {len(self.trades)} 笔交易")
        return self.get_results()

    def _record_equity(self, bar_idx, bar_date, bar_time):
        """记录资金曲线"""
        position_value = 0
        for pos in self.positions.values():
            position_value += pos["entry_price"] * pos["shares"]

        self.equity_curve.append({
            "bar": bar_idx,
            "date": bar_date,
            "time": bar_time,
            "cash": round(self.capital, 2),
            "position_value": round(position_value, 2),
            "total_equity": round(self.capital + position_value, 2),
        })

    def get_results(self) -> dict:
        """汇总回测结果"""
        trades = self.trades
        equity = pd.DataFrame(self.equity_curve)

        if not trades:
            return {
                "total_return": 0.0, "win_rate": 0, "total_trades": 0,
                "sharpe": 0, "max_drawdown": 0, "profit_factor": 0,
                "trades": [], "equity_curve": equity,
            }

        # 基础指标
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_rate = len(wins) / len(trades) * 100

        total_pnl = sum(t["pnl"] for t in trades)
        total_return = total_pnl / self.initial_capital * 100

        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t["pnl"]) for t in losses]) if losses else 0

        # 盈亏比
        profit_factor = sum(t["pnl"] for t in wins) / max(abs(sum(t["pnl"] for t in losses)), 0.01)

        # 最大回撤
        if not equity.empty and "total_equity" in equity.columns:
            equity["peak"] = equity["total_equity"].cummax()
            equity["drawdown"] = (equity["total_equity"] - equity["peak"]) / equity["peak"]
            max_dd = equity["drawdown"].min() * 100
        else:
            max_dd = 0

        # 夏普比率（简化版）
        if not equity.empty and len(equity) > 1:
            equity["daily_ret"] = equity["total_equity"].pct_change()
            std_val = equity["daily_ret"].std()
            if std_val and std_val > 0:
                sharpe = equity["daily_ret"].mean() / std_val * np.sqrt(252)
            else:
                sharpe = 0
        else:
            sharpe = 0

        # 卡玛比率
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0

        return {
            "total_return": round(total_return, 2),
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 2),
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd, 2),
            "calmar": round(calmar, 2),
            "trades": trades,
            "equity_curve": equity,
        }

    def print_summary(self):
        """打印回测摘要"""
        r = self.get_results()
        print("=" * 50)
        print("回测结果摘要")
        print("=" * 50)
        print(f"初始资金: RMB {self.initial_capital:,.0f}")
        print(f"最终权益: RMB {r.get('total_equity_at_end', self.capital):,.0f}")
        print(f"总收益率: {r['total_return']:.2f}%")
        print(f"总盈亏: RMB {r['total_pnl']:,.2f}")
        print(f"交易次数: {r['total_trades']}")
        print(f"胜率: {r['win_rate']:.1f}%")
        print(f"平均盈利: RMB {r['avg_win']:,.2f}")
        print(f"平均亏损: RMB {r['avg_loss']:,.2f}")
        print(f"盈亏比: {r['profit_factor']:.2f}")
        print(f"夏普比率: {r['sharpe']:.2f}")
        print(f"最大回撤: {r['max_drawdown']:.2f}%")
        print(f"卡玛比率: {r['calmar']:.2f}")
        print("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== 回测引擎测试 ===")
    # 模拟信号数据
    dates = pd.date_range("2025-01-02 09:30", periods=500, freq="15min")
    np.random.seed(42)
    price = 20.0
    data = []
    for d in dates:
        price += np.random.normal(0, 0.15)
        price = max(price, 15)
        signal = np.random.choice(["BUY", "SELL", "HOLD"], p=[0.1, 0.1, 0.8])
        data.append({
            "datetime": d,
            "symbol": "000933",
            "close": price,
            "atr": price * 0.01,
            "signal": signal,
        })
    test_df = pd.DataFrame(data)
    engine = BacktestEngine()
    results = engine.run(test_df)
    engine.print_summary()
