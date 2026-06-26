#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
逐票深入：逆向找规律 — 从高胜率买点反推指标特征
每只股票独立分析，找到各自的最优买卖点参数区间
"""
import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json, os, sys

# Fix console encoding
sys.stdout.reconfigure(encoding='utf-8')

bs.login()
CACHE_DIR = "C:/Users/Administrator/quant_trading/data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

STOCKS = {
    "sz.000933": "神火股份",
    "sz.002497": "雅化集团",
    "sz.000960": "锡业股份",
    "sz.000893": "亚钾国际",
}

def fetch_15min(code):
    cache_file = f"{CACHE_DIR}/{code.split('.')[1]}_15min.csv"
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, dtype={'time': str})
        df['time'] = df['time'].astype(str).str.zfill(17)  # pad to 17 chars
        df['datetime'] = pd.to_datetime(df['date'] + ' ' +
            df['time'].str[8:10] + ':' + df['time'].str[10:12] + ':' + df['time'].str[12:14])
        for c in ['open','high','low','close','volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    rs = bs.query_history_k_data_plus(code,
        'date,time,open,high,low,close,volume',
        start_date=start, end_date=end, frequency='15', adjustflag='2')
    rows = []
    while (rs.error_code == '0') & rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['date','time','open','high','low','close','volume'])
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['datetime'] = pd.to_datetime(df['date'] + ' ' +
        df['time'].str[:2] + ':' + df['time'].str[2:4] + ':' + df['time'].str[4:6])
    df.to_csv(cache_file, index=False)
    return df

def compute_features(df):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    n = len(close)

    df['ma5'] = np.convolve(close, np.ones(5)/5, mode='same')
    df['ma10'] = np.convolve(close, np.ones(10)/10, mode='same')
    df['ma20'] = np.convolve(close, np.ones(20)/20, mode='same')

    prev_ma5 = np.roll(df['ma5'].values, 1); prev_ma5[0] = prev_ma5[1]
    prev_ma10 = np.roll(df['ma10'].values, 1); prev_ma10[0] = prev_ma10[1]
    df['golden'] = ((df['ma5'] > df['ma10']) & (prev_ma5 <= prev_ma10)).astype(int)
    df['dead'] = ((df['ma5'] < df['ma10']) & (prev_ma5 >= prev_ma10)).astype(int)
    df['ma_bull'] = ((df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20'])).astype(int)
    df['ma_dist'] = (df['ma5'] - df['ma20']) / (df['close'] + 0.0001)  # MA离散度

    # RSI
    deltas = np.diff(close)
    rsi14 = np.full(n, 50.0)
    for i in range(14, n):
        g = np.sum(np.maximum(deltas[i-14:i], 0))
        l = np.sum(np.abs(np.minimum(deltas[i-14:i], 0)))
        rsi14[i] = 100 - 100/(1 + g/l) if l > 0 else 100
    df['rsi'] = rsi14

    # RSI momentum (direction change)
    df['rsi_delta'] = df['rsi'].diff(3)
    df['rsi_rising'] = (df['rsi_delta'] > 0).astype(int)

    # Bollinger
    df['bb_mid'] = df['ma20']
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_pct'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.0001)).clip(0, 1)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['close'] + 0.0001)

    # Volume
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['vol_ma5'] / (df['vol_ma20'] + 1)
    df['vol_surge'] = (df['vol_ratio'] > 1.3).astype(int)

    # Price position
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['pos'] = ((df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 0.0001)).clip(0, 1)

    # Short-term momentum
    df['roc_4'] = df['close'].pct_change(4) * 100
    df['roc_8'] = df['close'].pct_change(8) * 100
    df['roc_16'] = df['close'].pct_change(16) * 100

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_turning'] = ((df['macd_hist'] > 0) & (df['macd_hist'].shift(1) <= 0)).astype(int)
    df['macd_bull'] = (df['macd'] > df['macd_signal']).astype(int)

    # ATR-based volatility
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr_pct'] = (tr.rolling(14).mean() / (df['close'] + 0.0001)) * 100

    return df

def forward_returns(df):
    """1-day (32 bars), 2-day (64 bars), 3-day (96 bars) forward return"""
    close = df['close'].values
    for p, name in [(16, 'f8h'), (32, 'f1d'), (64, 'f2d'), (96, 'f3d')]:
        df[name] = np.nan
        for i in range(len(df) - p):
            df.loc[df.index[i], name] = (close[i+p] - close[i]) / close[i] * 100
    return df

# ====== GRID SEARCH per stock ======
def grid_search_buy(df, code, name):
    """Grid search for optimal buy parameter ranges per stock"""
    df = df.dropna(subset=['rsi', 'bb_pct', 'pos', 'ma_dist', 'vol_ratio', 'roc_4', 'roc_8',
                           'macd_hist', 'atr_pct', 'f1d', 'f2d', 'f3d']).copy()
    if len(df) < 500:
        return []

    n = len(df)
    results = []

    # Parameter grids for buy signals
    param_grids = {
        'rsi':       [(20,35), (30,45), (35,50), (40,55), (45,60)],
        'pos':       [(0.0,0.15), (0.1,0.3), (0.15,0.35), (0.2,0.4), (0.25,0.5)],
        'bb_pct':    [(0.0,0.2), (0.1,0.3), (0.15,0.4), (0.2,0.5)],
        'roc_8':     [(-3,-1), (-2,0), (-1,1), (0,2)],
        'vol_ratio': [(0.8,1.2), (1.0,1.5), (1.2,2.0)],
        'macd_hist': [(-0.05,0.0), (-0.02,0.02), (0.0,0.05)],
        'ma_dist':   [(-0.05,0.0), (-0.02,0.01), (0.0,0.03)],
        'atr_pct':   [(0.5,1.5), (1.0,2.5), (2.0,4.0)],
    }

    # Grid search with golden cross as base
    base_mask = df['golden'] == 1
    if base_mask.sum() < 20:
        base_mask = pd.Series(True, index=df.index)

    best_buys = []

    # Single-parameter optimization
    for feature, ranges in param_grids.items():
        for (lo, hi) in ranges:
            mask = base_mask & (df[feature] >= lo) & (df[feature] <= hi)
            sig_count = mask.sum()
            if sig_count < 15:
                continue

            for h in ['f1d', 'f2d', 'f3d']:
                returns = df.loc[mask, h].dropna()
                if len(returns) < 12:
                    continue
                wr = (returns > 0).mean() * 100
                avg = returns.mean()
                if wr >= 70:
                    best_buys.append({
                        'cond': f"金叉+{feature}{lo}-{hi}",
                        'sig': sig_count, 'h': h, 'n': len(returns),
                        'wr': round(wr,1), 'avg': round(avg,2),
                    })

    # Two-parameter optimization
    two_param_sets = [
        ('rsi', [(30,50), (35,50), (40,55)]),
        ('pos', [(0.0,0.3), (0.1,0.35), (0.15,0.4)]),
        ('bb_pct', [(0.0,0.3), (0.1,0.4)]),
        ('vol_ratio', [(0.8,1.3), (1.0,1.5)]),
        ('roc_8', [(-3,0), (-2,1)]),
        ('macd_hist', [(-0.03,0.01), (0.0,0.05)]),
    ]

    for i in range(len(two_param_sets)):
        for j in range(i+1, len(two_param_sets)):
            f1, r1_list = two_param_sets[i]
            f2, r2_list = two_param_sets[j]
            for r1 in r1_list:
                for r2 in r2_list:
                    mask = base_mask & \
                           (df[f1] >= r1[0]) & (df[f1] <= r1[1]) & \
                           (df[f2] >= r2[0]) & (df[f2] <= r2[1])
                    sig_count = mask.sum()
                    if sig_count < 12:
                        continue

                    for h in ['f1d', 'f2d', 'f3d']:
                        returns = df.loc[mask, h].dropna()
                        if len(returns) < 10:
                            continue
                        wr = (returns > 0).mean() * 100
                        avg = returns.mean()
                        if wr >= 80:
                            best_buys.append({
                                'cond': f"金叉+{f1}{r1[0]}-{r1[1]}+{f2}{r2[0]}-{r2[1]}",
                                'sig': sig_count, 'h': h, 'n': len(returns),
                                'wr': round(wr,1), 'avg': round(avg,2),
                            })

    # Three-parameter: golden+RSI+pos+bb_pct (most powerful combo)
    rsi_ranges = [(30,50), (35,50), (40,55)]
    pos_ranges = [(0.0,0.3), (0.1,0.35), (0.15,0.4)]
    bb_ranges  = [(0.0,0.3), (0.1,0.4)]

    for r in rsi_ranges:
        for p in pos_ranges:
            for b in bb_ranges:
                mask = base_mask & \
                       (df['rsi'] >= r[0]) & (df['rsi'] <= r[1]) & \
                       (df['pos'] >= p[0]) & (df['pos'] <= p[1]) & \
                       (df['bb_pct'] >= b[0]) & (df['bb_pct'] <= b[1])
                sig_count = mask.sum()
                if sig_count < 10:
                    continue

                for h in ['f1d', 'f2d', 'f3d']:
                    returns = df.loc[mask, h].dropna()
                    if len(returns) < 8:
                        continue
                    wr = (returns > 0).mean() * 100
                    avg = returns.mean()
                    key = f"金叉+RSI{r[0]}-{r[1]}+pos{p[0]}-{p[1]}+BB{b[0]}-{b[1]}"
                    best_buys.append({
                        'cond': key, 'sig': sig_count, 'h': h, 'n': len(returns),
                        'wr': round(wr,1), 'avg': round(avg,2),
                    })

    # Sort by win rate descending, then by sample count
    best_buys.sort(key=lambda x: (x['wr'], x['n']), reverse=True)
    return best_buys[:20]

def grid_search_sell(df, code, name):
    """Grid search for optimal sell parameter ranges"""
    df = df.dropna(subset=['rsi', 'bb_pct', 'pos', 'ma_dist', 'vol_ratio',
                           'macd_hist', 'atr_pct', 'f1d', 'f2d', 'f3d']).copy()
    if len(df) < 500:
        return []

    n = len(df)

    # Sell base: high RSI + high position
    base_mask = (df['rsi'] > 60) & (df['pos'] > 0.5)

    best_sells = []

    rsi_ranges = [(65,85), (70,85), (75,90)]
    pos_ranges = [(0.6,1.0), (0.7,1.0), (0.8,1.0)]
    bb_ranges  = [(0.7,1.0), (0.8,1.0), (0.85,1.0)]

    for r in rsi_ranges:
        for p in pos_ranges:
            for b in bb_ranges:
                mask = (df['rsi'] >= r[0]) & (df['rsi'] <= r[1]) & \
                       (df['pos'] >= p[0]) & (df['pos'] <= p[1]) & \
                       (df['bb_pct'] >= b[0]) & (df['bb_pct'] <= b[1])
                sig_count = mask.sum()
                if sig_count < 10:
                    continue

                for h in ['f1d', 'f2d', 'f3d']:
                    returns = df.loc[mask, h].dropna()
                    if len(returns) < 8:
                        continue
                    fall = (returns < 0).mean() * 100
                    avg = returns.mean()
                    if fall >= 35 and avg < 0:
                        best_sells.append({
                            'cond': f"RSI{r[0]}-{r[1]}+pos{p[0]}-{p[1]}+BB{b[0]}-{b[1]}",
                            'sig': sig_count, 'h': h, 'n': len(returns),
                            'fall': round(fall,1), 'avg': round(avg,2),
                        })

    # Also: dead cross + high RSI combos
    rsi_ranges2 = [(60,80), (65,80), (70,85)]
    pos_ranges2 = [(0.5,1.0), (0.6,1.0)]

    for r in rsi_ranges2:
        for p in pos_ranges2:
            mask = (df['dead'] == 1) & \
                   (df['rsi'] >= r[0]) & (df['rsi'] <= r[1]) & \
                   (df['pos'] >= p[0]) & (df['pos'] <= p[1])
            sig_count = mask.sum()
            if sig_count < 10:
                continue

            for h in ['f1d', 'f2d', 'f3d']:
                returns = df.loc[mask, h].dropna()
                if len(returns) < 8:
                    continue
                fall = (returns < 0).mean() * 100
                avg = returns.mean()
                key = f"死叉+RSI{r[0]}-{r[1]}+pos{p[0]}-{p[1]}"
                best_sells.append({
                    'cond': key, 'sig': sig_count, 'h': h, 'n': len(returns),
                    'fall': round(fall,1), 'avg': round(avg,2),
                })

    best_sells.sort(key=lambda x: (x['fall'], -x['avg']), reverse=True)
    return best_sells[:15]

# ====== MAIN ======
all_results = {}

for code, name in STOCKS.items():
    print(f"\n{'='*70}")
    print(f"  {name} ({code}) — 逐票网格搜索")
    print(f"{'='*70}")

    df = fetch_15min(code)
    if df.empty or len(df) < 500:
        print(f"  SKIP: insufficient data")
        continue

    df = compute_features(df)
    df = forward_returns(df)
    print(f"  Data: {len(df)} bars")

    # --- Buy search ---
    print(f"\n  [买入] 最优参数网格搜索...")
    buys = grid_search_buy(df, code, name)
    print(f"  找到 {len(buys)} 个候选 (WR>=70%)")
    for b in buys[:10]:
        bar_icon = '#' * min(int(b['wr']/2), 45)
        print(f"    WR={b['wr']:.1f}% {bar_icon}")
        print(f"    {b['cond']} | {b['h']} n={b['n']} | avg={b['avg']:+.2f}%")

    # --- Sell search ---
    print(f"\n  [卖出] 最优参数网格搜索...")
    sells = grid_search_sell(df, code, name)
    print(f"  找到 {len(sells)} 个候选")
    for s in sells[:10]:
        bar_icon = '-' * min(int(s['fall']/2), 30)
        print(f"    FALL={s['fall']:.1f}% {bar_icon}")
        print(f"    {s['cond']} | {s['h']} n={s['n']} | avg={s['avg']:+.2f}%")

    all_results[code] = {
        'name': name,
        'top_buys': buys[:10],
        'top_sells': sells[:10],
    }

# ====== Save ======
with open(f"{CACHE_DIR}/per_stock_grid_search.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

print(f"\n{'='*70}")
print(f"  Results saved to {CACHE_DIR}/per_stock_grid_search.json")
print(f"{'='*70}")

# Print actionable summary
print(f"\n")
print(f"{'='*70}")
print(f"  实战买卖点规则总结")
print(f"{'='*70}")

for code, res in all_results.items():
    name = res['name']
    print(f"\n--- {name} ({code.split('.')[1]}) ---")

    buys = res['top_buys']
    sells = res['top_sells']

    # Find the best buy with WR>=88% and decent samples
    buy_88 = [b for b in buys if b['wr'] >= 88 and b['n'] >= 10]
    buy_85 = [b for b in buys if b['wr'] >= 85 and b['n'] >= 12]
    buy_80 = [b for b in buys if b['wr'] >= 80 and b['n'] >= 15]

    if buy_88:
        b = buy_88[0]
        print(f"  BUY(>=88%): WR={b['wr']:.1f}% | {b['cond']} | {b['h']} | n={b['n']}")
    elif buy_85:
        b = buy_85[0]
        print(f"  BUY(>=85%): WR={b['wr']:.1f}% | {b['cond']} | {b['h']} | n={b['n']}")
    elif buy_80:
        b = buy_80[0]
        print(f"  BUY(>=80%): WR={b['wr']:.1f}% | {b['cond']} | {b['h']} | n={b['n']}")

    # Best sell
    sell_good = [s for s in sells if s['fall'] >= 40 and s['n'] >= 10]
    if sell_good:
        s = sell_good[0]
        print(f"  SELL: FALL={s['fall']:.1f}% | {s['cond']} | {s['h']} | n={s['n']}")
    elif sells:
        s = sells[0]
        print(f"  SELL(best): FALL={s['fall']:.1f}% | {s['cond']} | {s['h']} | n={s['n']}")

bs.logout()
