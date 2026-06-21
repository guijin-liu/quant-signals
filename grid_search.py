"""
参数网格搜索 — 5年日线回测 × MA共振 × 多维参数组合
目标: 找出胜率最高的参数组合
"""
import sys, io, json, logging
from datetime import datetime
from pathlib import Path
import pandas as pd, numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
logging.basicConfig(level=logging.WARNING)

from data.baostock_fetcher import fetch_all_daily_klines
from features.technical import compute_all_technical_features
from backtest.engine import BacktestEngine
from config import BACKTEST_PARAMS, RISK_PARAMS, STOCK_CODES

# ============================================================
# 1. 准备数据(只需一次)
# ============================================================
print("="*70)
print("  参数网格搜索 — 目标胜率>88%")
print("="*70)

print("\n[1/2] 加载5年日线...")
daily_data = fetch_all_daily_klines(5)

print("\n[2/2] 计算技术指标...")
all_stock_dfs = {}
for code in STOCK_CODES:
    df = daily_data.get(code)
    if df is not None and not df.empty:
        all_stock_dfs[code] = compute_all_technical_features(df, 'daily')
        print(f"  {code}: {len(all_stock_dfs[code])}条, {all_stock_dfs[code].columns.tolist()[-3:]}")

# ============================================================
# 2. 参数网格定义
# ============================================================
param_grid = {
    "buy_threshold":      [0.60, 0.65, 0.70, 0.75],
    "resonance_required": [1, 2, 3],
    "stop_loss_pct":      [0.015, 0.02, 0.025],
    "take_profit_pct":    [0.04, 0.05, 0.06],
    "tech_weight":        [0.30, 0.35, 0.40],
    "trend_weight":       [0.15, 0.20, 0.25],
}

# ============================================================
# 3. 信号生成函数(参数化)
# ============================================================
def generate_signals(code, df, params):
    """根据参数生成BUY/SELL/HOLD信号"""
    df = df.copy()
    df['signal'] = 'HOLD'
    df['signal_score'] = 0.0
    df['resonance_count'] = 0.0  # float

    tw = params['tech_weight']
    trw = params['trend_weight']
    other_w = (1.0 - tw - trw) / 4  # 4个维度平分剩余权重
    bt = params['buy_threshold']
    rr = params['resonance_required']

    for i in range(max(60, len(df)//10), len(df)):
        # 技术维度
        tech_bull = 0
        if all(f'ma_{p}' in df.columns for p in [5,10,20]):
            if df['ma_5'].iloc[i] > df['ma_10'].iloc[i] > df['ma_20'].iloc[i]:
                tech_bull += 2
            elif df['ma_5'].iloc[i] > df['ma_10'].iloc[i]:
                tech_bull += 1
        if 'ma_60' in df.columns and df['close'].iloc[i] > df['ma_60'].iloc[i]:
            tech_bull += 1
        if 'macd_golden_cross' in df.columns and df['macd_golden_cross'].iloc[i]: tech_bull += 1
        if 'macd_hist_sign_change' in df.columns and df['macd_hist_sign_change'].iloc[i]: tech_bull += 1
        if 'rsi' in df.columns and 40 < df['rsi'].iloc[i] < 70: tech_bull += 1
        if 'volume_surge' in df.columns and df['volume_surge'].iloc[i]: tech_bull += 1
        if 'boll_pct_b' in df.columns and df['boll_pct_b'].iloc[i] > 0.4: tech_bull += 1
        tech_score = min(tech_bull/7.0, 1.0)

        # 趋势维度
        trend_bull = 0
        if 'pct_change_5' in df.columns and df['pct_change_5'].iloc[i] > 0: trend_bull += 1
        if 'ma_20' in df.columns and df['close'].iloc[i] > df['ma_20'].iloc[i]: trend_bull += 1
        if 'pct_change_3' in df.columns and df['pct_change_3'].iloc[i] > 0: trend_bull += 1
        trend_score = min(trend_bull/3.0, 1.0)

        # 综合评分
        composite = tw*tech_score + trw*trend_score + other_w*0.5*4
        resonance = float((tech_score>=0.65) + (trend_score>=0.6))

        df.at[df.index[i], 'signal_score'] = round(composite, 3)
        df.at[df.index[i], 'resonance_count'] = float(resonance)

        if composite > bt and resonance >= rr:
            df.at[df.index[i], 'signal'] = 'BUY'
        elif composite < 0.30:
            df.at[df.index[i], 'signal'] = 'SELL'

    return df

# ============================================================
# 4. 网格搜索
# ============================================================
from itertools import product

param_names = list(param_grid.keys())
param_values = list(param_grid.values())
total_combos = 1
for v in param_values:
    total_combos *= len(v)
print(f"\n网格搜索: {total_combos}个参数组合...")

best = None
best_win_rate = 0
all_results = []

combo_idx = 0
for combo in product(*param_values):
    params = dict(zip(param_names, combo))
    combo_idx += 1

    # 更新风控参数
    RISK_PARAMS['stop_loss_pct'] = params['stop_loss_pct']
    RISK_PARAMS['take_profit_pct'] = params['take_profit_pct']

    # 按参数生成信号
    all_signals = []
    for code, df in all_stock_dfs.items():
        signals = generate_signals(code, df, params)
        all_signals.append(signals)

    signal_df = pd.concat(all_signals, ignore_index=True)
    signal_df.sort_values(['datetime', 'symbol'], inplace=True)

    # 回测
    engine = BacktestEngine()
    results = engine.run(signal_df)
    wr = results['win_rate']
    ret = results['total_return']
    tr = results['total_trades']

    entry = {**params, 'win_rate': wr, 'total_return': ret, 'trades': tr}
    all_results.append(entry)

    if wr > best_win_rate:
        best_win_rate = wr
        best = entry
        # 不设上限，只找胜率最高

    if combo_idx % 50 == 0 or combo_idx == total_combos:
        print(f"  [{combo_idx}/{total_combos}] best_wr={best_win_rate:.1f}% "
              f"current_wr={wr:.1f}% params={params['buy_threshold']}/{params['resonance_required']}/"
              f"{params['tech_weight']}/{params['stop_loss_pct']}")

# ============================================================
# 5. 输出结果
# ============================================================
print(f"\n{'='*70}")
print(f"  最优参数 (胜率最高)")
print(f"{'='*70}")
print(f"  胜率: {best['win_rate']:.1f}%")
print(f"  收益率: {best['total_return']:.2f}%")
print(f"  交易次数: {best['trades']}")
print(f"")
print(f"  买入阈值:     {best['buy_threshold']}")
print(f"  共振维度:     {best['resonance_required']}")
print(f"  技术权重:     {best['tech_weight']}")
print(f"  趋势权重:     {best['trend_weight']}")
print(f"  止损:         {best['stop_loss_pct']*100:.1f}%")
print(f"  止盈:         {best['take_profit_pct']*100:.1f}%")

# Top 10
print(f"\n{'='*70}")
print(f"  Top 10 参数组合(按胜率)")
print(f"{'='*70}")
top10 = sorted(all_results, key=lambda x: x['win_rate'], reverse=True)[:10]
for i, r in enumerate(top10):
    print(f"  {i+1}. WR={r['win_rate']:.1f}% Ret={r['total_return']:.2f}% "
          f"Trades={r['trades']} BuyTh={r['buy_threshold']} Res={r['resonance_required']} "
          f"Tw={r['tech_weight']} Twd={r['trend_weight']} SL={r['stop_loss_pct']}")

# 保存结果
report_file = Path(__file__).parent / "reports" / f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
report_file.parent.mkdir(parents=True, exist_ok=True)
with open(report_file, 'w', encoding='utf-8') as f:
    json.dump({'best': best, 'top10': top10, 'all': all_results[:200]}, f, ensure_ascii=False, indent=2)
print(f"\n结果保存: {report_file}")
