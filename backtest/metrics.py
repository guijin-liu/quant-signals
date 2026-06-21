"""
绩效指标计算
- 胜率、盈亏比
- 夏普比率
- 最大回撤
- 卡玛比率
- 风险调整收益
"""

import numpy as np
import pandas as pd
from typing import List, Dict


def compute_win_rate(trades: List[Dict]) -> float:
    """胜率"""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return round(wins / len(trades) * 100, 2)


def compute_profit_factor(trades: List[Dict]) -> float:
    """盈亏比 (总盈利/总亏损)"""
    total_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    total_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    return round(total_profit / total_loss, 2) if total_loss > 0 else float("inf")


def compute_avg_trade(trades: List[Dict]) -> dict:
    """平均交易统计"""
    if not trades:
        return {"avg_pnl": 0, "avg_pnl_pct": 0, "avg_win": 0, "avg_loss": 0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    return {
        "avg_pnl": round(np.mean([t["pnl"] for t in trades]), 2),
        "avg_pnl_pct": round(np.mean([t["pnl_pct"] for t in trades]), 2),
        "avg_win": round(np.mean([t["pnl"] for t in wins]), 2) if wins else 0,
        "avg_loss": round(np.mean([abs(t["pnl"]) for t in losses]), 2) if losses else 0,
        "max_win": round(max(t["pnl"] for t in trades), 2),
        "max_loss": round(min(t["pnl"] for t in trades), 2),
    }


def compute_sharpe_ratio(equity_df: pd.DataFrame, risk_free: float = 0.03) -> float:
    """
    夏普比率
    假设：年化252个交易日，日无风险利率=3%
    """
    if equity_df.empty or "total_equity" not in equity_df.columns:
        return 0.0

    df = equity_df.copy()
    df["return"] = df["total_equity"].pct_change()
    returns = df["return"].dropna()
    if len(returns) < 2:
        return 0.0

    excess = returns - risk_free / 252
    return round(excess.mean() / excess.std().replace(0, 1e-10) * np.sqrt(252), 2)


def compute_max_drawdown(equity_df: pd.DataFrame) -> dict:
    """
    最大回撤分析
    """
    if equity_df.empty or "total_equity" not in equity_df.columns:
        return {"max_drawdown_pct": 0.0, "max_drawdown_amount": 0.0, "max_dd_duration": 0}

    df = equity_df.copy()
    df["peak"] = df["total_equity"].cummax()
    df["drawdown"] = (df["total_equity"] - df["peak"]) / df["peak"]
    df["is_drawdown"] = df["drawdown"] < 0

    max_dd = df["drawdown"].min()
    max_dd_idx = df["drawdown"].idxmin()

    # 回撤恢复时间
    peak_val = df["peak"].iloc[0]
    peak_bar = 0
    max_duration = 0
    current_duration = 0

    for i, (_, row) in enumerate(df.iterrows()):
        if row["total_equity"] >= peak_val:
            peak_val = row["total_equity"]
            peak_bar = i
            current_duration = 0
        elif row["drawdown"] < 0:
            current_duration = i - peak_bar
            max_duration = max(max_duration, current_duration)

    return {
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_drawdown_amount": round(max_dd * df.loc[max_dd_idx, "peak"] if max_dd_idx is not None else 0, 2),
        "max_dd_duration": max_duration,
    }


def compute_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """卡玛比率 = 年化收益率 / 最大回撤"""
    if max_drawdown == 0:
        return 0.0
    return round(abs(total_return / max_drawdown), 2)


def compute_sortino_ratio(equity_df: pd.DataFrame, risk_free: float = 0.03) -> float:
    """
    Sortino比率（只考虑下行风险）
    """
    if equity_df.empty or "total_equity" not in equity_df.columns:
        return 0.0

    df = equity_df.copy()
    df["return"] = df["total_equity"].pct_change()
    returns = df["return"].dropna()
    if len(returns) < 2:
        return 0.0

    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0

    excess = returns.mean() - risk_free / 252
    downside_std = downside.std()
    return round(excess / downside_std.replace(0, 1e-10) * np.sqrt(252), 2)


def compute_consecutive_stats(trades: List[Dict]) -> dict:
    """连续盈/亏统计"""
    if not trades:
        return {"max_consecutive_wins": 0, "max_consecutive_losses": 0}

    max_wins = 0
    max_losses = 0
    cur_wins = 0
    cur_losses = 0

    for t in trades:
        if t["pnl"] > 0:
            cur_wins += 1
            cur_losses = 0
            max_wins = max(max_wins, cur_wins)
        else:
            cur_losses += 1
            cur_wins = 0
            max_losses = max(max_losses, cur_losses)

    return {
        "max_consecutive_wins": max_wins,
        "max_consecutive_losses": max_losses,
    }


def compute_monthly_returns(equity_df: pd.DataFrame) -> pd.DataFrame:
    """月度收益率统计"""
    if equity_df.empty or "date" not in equity_df.columns:
        return pd.DataFrame()

    df = equity_df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly = df.groupby("month").agg({
        "total_equity": ["first", "last"],
    }).reset_index()
    monthly.columns = ["month", "start", "end"]
    monthly["return"] = (monthly["end"] / monthly["start"] - 1) * 100
    return monthly


def full_performance_report(trades: List[Dict], equity_df: pd.DataFrame, initial_capital: float) -> dict:
    """完整的绩效报告"""
    if not trades:
        return {"message": "无交易记录"}

    final_equity = equity_df["total_equity"].iloc[-1] if not equity_df.empty else initial_capital
    total_return = (final_equity / initial_capital - 1) * 100
    max_dd_info = compute_max_drawdown(equity_df)

    report = {
        # 收益指标
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return, 2),
        "total_pnl": round(final_equity - initial_capital, 2),
        # 交易统计
        "total_trades": len(trades),
        "win_rate": compute_win_rate(trades),
        "profit_factor": compute_profit_factor(trades),
        **compute_avg_trade(trades),
        **compute_consecutive_stats(trades),
        # 风险指标
        "sharpe": compute_sharpe_ratio(equity_df),
        "sortino": compute_sortino_ratio(equity_df),
        "max_drawdown": max_dd_info["max_drawdown_pct"],
        "max_dd_duration": max_dd_info["max_dd_duration"],
        "calmar": compute_calmar_ratio(total_return, max_dd_info["max_drawdown_pct"]),
    }

    return report
