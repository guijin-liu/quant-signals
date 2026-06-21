"""
回测报告生成与可视化
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无GUI后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_equity_curve(equity_df: pd.DataFrame, title: str = "资金曲线", save_path: str = None):
    """绘制资金曲线 + 回撤"""
    if equity_df.empty:
        logger.warning("无数据，跳过绘图")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})

    df = equity_df.copy()
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"])

    # 图1: 资金曲线
    ax = axes[0]
    ax.plot(df.index, df["total_equity"], color="#1f77b4", linewidth=0.8, label="总权益")
    ax.axhline(y=df["total_equity"].iloc[0], color="gray", linestyle="--", alpha=0.5, label="初始资金")
    ax.set_ylabel("权益 (元)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # 图2: 回撤
    ax = axes[1]
    df["peak"] = df["total_equity"].cummax()
    df["drawdown"] = (df["total_equity"] - df["peak"]) / df["peak"] * 100
    ax.fill_between(df.index, 0, df["drawdown"], color="red", alpha=0.3)
    ax.set_ylabel("回撤 (%)")
    ax.grid(True, alpha=0.3)

    # 图3: 每日收益分布
    ax = axes[2]
    df["daily_ret"] = df["total_equity"].pct_change() * 100
    colors = ["green" if r > 0 else "red" for r in df["daily_ret"].fillna(0)]
    ax.bar(df.index, df["daily_ret"].fillna(0), color=colors, alpha=0.6, width=1)
    ax.set_ylabel("收益 (%)")
    ax.set_xlabel("K线序号")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"图表已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_trade_analysis(trades: List[Dict], save_path: str = None):
    """交易分析图表"""
    if not trades:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pnls = [t["pnl"] for t in trades]
    pnl_pcts = [t["pnl_pct"] for t in trades]

    # 1. PnL分布
    ax = axes[0, 0]
    colors = ["green" if p > 0 else "red" for p in pnls]
    ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("每笔交易盈亏")
    ax.set_xlabel("交易序号")
    ax.set_ylabel("盈亏 (元)")
    ax.grid(True, alpha=0.3)

    # 2. 累计盈亏
    ax = axes[0, 1]
    cumulative = np.cumsum(pnls)
    ax.plot(cumulative, color="#1f77b4", linewidth=1.5)
    ax.fill_between(range(len(cumulative)), 0, cumulative, alpha=0.2)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("累计盈亏")
    ax.set_xlabel("交易序号")
    ax.set_ylabel("累计盈亏 (元)")
    ax.grid(True, alpha=0.3)

    # 3. 收益率分布直方图
    ax = axes[1, 0]
    ax.hist(pnl_pcts, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(x=0, color="red", linestyle="--")
    ax.set_title("收益率分布")
    ax.set_xlabel("收益率 (%)")
    ax.set_ylabel("频次")
    ax.grid(True, alpha=0.3)

    # 4. 按股票统计
    ax = axes[1, 1]
    stock_pnl = {}
    for t in trades:
        name = t.get("name", t["symbol"])
        stock_pnl[name] = stock_pnl.get(name, 0) + t["pnl"]
    names = list(stock_pnl.keys())
    values = list(stock_pnl.values())
    colors_bar = ["green" if v > 0 else "red" for v in values]
    ax.barh(names, values, color=colors_bar, alpha=0.7)
    ax.set_title("各股票累计盈亏")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"交易分析图已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def generate_html_report(results: dict, output_path: str = None) -> str:
    """生成HTML格式的回测报告"""
    if not results or not results.get("trades"):
        return "<html><body><h2>无交易记录</h2></body></html>"

    r = results
    trades = r["trades"]
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
    <meta charset="UTF-8">
    <title>量化交易回测报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
        .metric {{ text-align: center; padding: 15px; background: #f8f9fa; border-radius: 6px; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #1f77b4; }}
        .metric-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .positive {{ color: green !important; }}
        .negative {{ color: red !important; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; font-size: 13px; }}
        th {{ background: #f8f9fa; font-weight: bold; }}
        .win {{ color: green; }}
        .loss {{ color: red; }}
        h2 {{ color: #333; border-left: 4px solid #1f77b4; padding-left: 10px; }}
    </style>
    </head>
    <body>
    <div class="container">
        <h1 style="text-align:center;">量化交易回测报告</h1>
        <p style="text-align:center;color:#999;">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="card">
            <h2>核心指标</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value">{r['total_return']}%</div>
                    <div class="metric-label">总收益率</div>
                </div>
                <div class="metric">
                    <div class="metric-value {'positive' if r['win_rate'] >= 50 else 'negative'}">{r['win_rate']}%</div>
                    <div class="metric-label">胜率</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{r['profit_factor']}</div>
                    <div class="metric-label">盈亏比</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{r['sharpe']}</div>
                    <div class="metric-label">夏普比率</div>
                </div>
                <div class="metric">
                    <div class="metric-value negative">{r['max_drawdown']}%</div>
                    <div class="metric-label">最大回撤</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{r['total_trades']}</div>
                    <div class="metric-label">交易次数</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{r['calmar']}</div>
                    <div class="metric-label">卡玛比率</div>
                </div>
                <div class="metric">
                    <div class="metric-value">¥{r['total_pnl']:,.0f}</div>
                    <div class="metric-label">总盈亏</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>交易统计</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-value">{len(wins)}</div>
                    <div class="metric-label">盈利次数</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(losses)}</div>
                    <div class="metric-label">亏损次数</div>
                </div>
                <div class="metric">
                    <div class="metric-value positive">¥{r.get('avg_win', 0):,.0f}</div>
                    <div class="metric-label">平均盈利</div>
                </div>
                <div class="metric">
                    <div class="metric-value negative">¥{r.get('avg_loss', 0):,.0f}</div>
                    <div class="metric-label">平均亏损</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>最近交易记录</h2>
            <table>
                <tr>
                    <th>时间</th><th>股票</th><th>方向</th><th>入场价</th><th>出场价</th>
                    <th>盈亏</th><th>收益率</th><th>持仓</th><th>原因</th>
                </tr>
    """

    for t in trades[-30:]:
        pnl_class = "win" if t["pnl"] > 0 else "loss"
        html += f"""
                <tr>
                    <td>{t.get('entry_date', '')} {t.get('entry_time', '')}</td>
                    <td>{t.get('name', t['symbol'])}</td>
                    <td>买入</td>
                    <td>{t['entry_price']:.2f}</td>
                    <td>{t['exit_price']:.2f}</td>
                    <td class="{pnl_class}">¥{t['pnl']:,.0f}</td>
                    <td class="{pnl_class}">{t['pnl_pct']:.2f}%</td>
                    <td>{t['bars_held']}K</td>
                    <td>{t['reason']}</td>
                </tr>"""

    html += """
            </table>
        </div>
    </div>
    </body>
    </html>
    """

    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"HTML报告已保存: {output_path}")

    return html


def save_report(results: dict, output_dir: str = "./reports"):
    """保存完整回测报告（HTML + 图表）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # HTML报告
    html_path = output_dir / f"backtest_report_{timestamp}.html"
    generate_html_report(results, str(html_path))

    # 图表
    equity = results.get("equity_curve", pd.DataFrame())
    if not equity.empty:
        chart_path = output_dir / f"equity_curve_{timestamp}.png"
        plot_equity_curve(equity, save_path=str(chart_path))

        trade_chart_path = output_dir / f"trade_analysis_{timestamp}.png"
        plot_trade_analysis(results.get("trades", []), save_path=str(trade_chart_path))

    logger.info(f"报告已保存到: {output_dir}")
    return str(output_dir)
