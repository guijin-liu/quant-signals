"""
风控管理模块
- 止损/止盈
- ATR动态止损
- 仓位管理
- 日内风控限制
"""

import logging
import numpy as np
from config import RISK_PARAMS, BACKTEST_PARAMS

logger = logging.getLogger(__name__)


class RiskManager:
    """交易风控管理器"""

    def __init__(self, params: dict = None):
        p = RISK_PARAMS.copy()
        if params:
            p.update(params)

        self.stop_loss_pct = p["stop_loss_pct"]
        self.take_profit_pct = p["take_profit_pct"]
        self.trailing_stop_atr = p["trailing_stop_atr"]
        self.max_daily_loss_pct = p["max_daily_loss_pct"]
        self.max_consecutive_losses = p["max_consecutive_losses"]
        self.kelly_fraction = p["kelly_fraction"]

        # 动态状态
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.current_date = None
        self.trading_allowed = True

    def reset_daily(self, date):
        """新交易日重置"""
        if self.current_date != date:
            self.daily_pnl = 0.0
            self.current_date = date
            self.trading_allowed = True
            logger.debug(f"新交易日: {date}")

    def check_stop_loss(self, entry_price: float, current_price: float, atr: float = 0) -> tuple:
        """
        检查止损/止盈
        返回: (should_exit: bool, reason: str)
        """
        if entry_price <= 0:
            return False, ""

        pnl_pct = (current_price - entry_price) / entry_price

        # 固定止损
        if pnl_pct <= -self.stop_loss_pct:
            return True, f"止损触发 ({pnl_pct*100:.2f}%)"

        # 固定止盈
        if pnl_pct >= self.take_profit_pct:
            return True, f"止盈触发 ({pnl_pct*100:.2f}%)"

        return False, ""

    def check_trailing_stop(
        self, entry_price: float, current_price: float,
        high_since_entry: float, atr: float,
    ) -> tuple:
        """
        ATR追踪止损检查
        只在盈利状态下启用追踪
        """
        if entry_price <= 0 or atr <= 0:
            return False, ""

        pnl_pct = (current_price - entry_price) / entry_price

        # 只在盈利>1%后启用追踪
        if pnl_pct < 0.01:
            return False, ""

        trailing_stop = high_since_entry - self.trailing_stop_atr * atr
        if current_price <= trailing_stop:
            return True, f"ATR追踪止损 ({pnl_pct*100:.2f}%)"

        return False, ""

    def compute_position_size(
        self, capital: float, price: float,
        win_rate: float = 0.5, avg_win: float = 0.02, avg_loss: float = 0.015,
    ) -> int:
        """
        基于凯利公式的仓位计算
        返回建议股数（100股整数倍）
        """
        if price <= 0 or capital <= 0:
            return 0

        # 凯利比例
        b = avg_win / avg_loss if avg_loss > 0 else 1  # 盈亏比
        p = win_rate
        q = 1 - p

        kelly = (p * b - q) / b if b > 0 else 0
        kelly = max(0, min(kelly, 0.25))  # 凯利上限25%

        # 使用凯利分数（保守）
        bet_pct = kelly * self.kelly_fraction
        bet_pct = max(0.01, min(bet_pct, 0.25))  # 1%~25%

        # 风控调整
        if self.consecutive_losses >= self.max_consecutive_losses:
            bet_pct *= 0.5  # 连续亏损减半
            logger.warning(f"连续亏损{self.consecutive_losses}次，仓位减半")

        if not self.trading_allowed:
            return 0

        max_amount = capital * bet_pct
        shares = int(max_amount / price / 100) * 100
        min_shares = 100

        return max(shares, min_shares) if max_amount >= price * min_shares else 0

    def record_trade(self, pnl: float, date):
        """记录交易结果，更新风控状态"""
        self.daily_pnl += pnl

        if pnl <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # 检查日亏损限制
        if abs(self.daily_pnl) >= BACKTEST_PARAMS["initial_capital"] * self.max_daily_loss_pct:
            self.trading_allowed = False
            logger.warning(f"日亏损达到{self.max_daily_loss_pct*100}%上限，暂停交易")

        # 检查连续亏损
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.trading_allowed = False
            logger.warning(f"连续亏损{self.consecutive_losses}次，暂停交易")

    def daily_summary(self) -> dict:
        """每日风控摘要"""
        return {
            "daily_pnl": round(self.daily_pnl, 2),
            "consecutive_losses": self.consecutive_losses,
            "trading_allowed": self.trading_allowed,
            "daily_loss_limit": f"{self.max_daily_loss_pct*100}%",
        }


class PositionSizer:
    """仓位计算器（简化版）"""

    @staticmethod
    def equal_weight(capital: float, price: float, max_positions: int = 4) -> int:
        """等权仓位"""
        per_position = capital / max_positions
        shares = int(per_position / price / 100) * 100
        return max(shares, 0)

    @staticmethod
    def risk_based(capital: float, price: float, stop_loss_pct: float, risk_per_trade: float = 0.01) -> int:
        """
        基于风险的仓位
        每笔交易风险 = 总资金 * risk_per_trade
        止损幅度 = stop_loss_pct
        """
        risk_amount = capital * risk_per_trade
        per_share_risk = price * stop_loss_pct
        if per_share_risk <= 0:
            return 0
        shares = int(risk_amount / per_share_risk / 100) * 100
        return max(shares, 0)

    @staticmethod
    def volatility_based(capital: float, price: float, atr: float, atr_multiple: float = 2.0) -> int:
        """基于波动率的仓位（ATR）"""
        risk_amount = capital * 0.01
        per_share_risk = atr * atr_multiple
        if per_share_risk <= 0:
            return 0
        shares = int(risk_amount / per_share_risk / 100) * 100
        return max(shares, 0)
