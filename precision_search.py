#!/usr/bin/env python
"""
逐票精确买卖点搜索 — 买点要求在N日最低价附近(误差<X%), 卖点要求在N日最高价附近
每只票独立搜索最优参数组合
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json, os, sys

sys.stdout.reconfigure(encoding='utf-8')

CACHE = "C:/Users/Administrator/quant_trading/data/cache"

STOCKS = {"000933":"神火","002497":"雅化","000960":"锡业","000893":"亚钾"}

def calc_features(close, high, low, volume):
    """向量化特征计算"""
    n = len(close)
    ma5  = np.convolve(close, np.ones(5)/5, mode='same')
    ma10 = np.convolve(close, np.ones(10)/10, mode='same')
    ma20 = np.convolve(close, np.ones(20)/20, mode='same')
    p_ma5 = np.roll(ma5,1); p_ma5[0]=p_ma5[1]
    p_ma10 = np.roll(ma10,1); p_ma10[0]=p_ma10[1]
    golden = (ma5 > ma10) & (p_ma5 <= p_ma10)
    dead = (ma5 < ma10) & (p_ma5 >= p_ma10)

    # RSI14
    d = np.diff(close); rsi = np.full(n, 50.)
    for i in range(14, n):
        g = np.sum(np.maximum(d[i-14:i],0)); l = np.sum(np.abs(np.minimum(d[i-14:i],0)))
        rsi[i] = 100-100/(1+g/l) if l>0 else 100

    # BB%
    bb_std = np.array([np.std(close[max(0,i-19):i+1]) for i in range(n)])
    bb_pct = np.clip((close - ma20 - 2*bb_std) / (4*bb_std + 0.0001) + 0.5, 0, 1)

    # Position in 20-bar range
    h20 = pd.Series(high).rolling(20).max().values
    l20 = pd.Series(low).rolling(20).min().values
    pos = np.clip((close - l20) / (h20 - l20 + 0.0001), 0, 1)

    # Volume ratio
    v5  = pd.Series(volume).rolling(5).mean().values
    v20 = pd.Series(volume).rolling(20).mean().values
    vol_ratio = v5 / (v20 + 1)

    return golden, dead, rsi, bb_pct, pos, vol_ratio


def find_buy_quality(df, code, name):
    """
    搜索最优买点:
    要求: 信号出现后, 1-3日内价格在附近最低价的X%以内
    输出: 精确度(距最低点%) + 胜率 + 平均获利
    """
    close = df['close'].values; high = df['high'].values
    low = df['low'].values; volume = df['volume'].values; n = len(close)
    golden, dead, rsi, bb_pct, pos, vol_ratio = calc_features(close, high, low, volume)
    print(f"   {name}: {n} bars")

    # 追索每种条件组合的买卖点质量
    results = []

    # === 搜索买点: 金叉 + 各种限制条件 ===
    buy_conditions = [
        ("金叉", golden),
        ("金叉+BB<0.3", golden & (bb_pct < 0.3)),
        ("金叉+pos<0.3", golden & (pos < 0.3)),
        ("金叉+BB<0.3+pos<0.3", golden & (bb_pct < 0.3) & (pos < 0.3)),
        ("金叉+BB<0.3+pos<0.35", golden & (bb_pct < 0.3) & (pos < 0.35)),
        ("金叉+BB<0.25+pos<0.25", golden & (bb_pct < 0.25) & (pos < 0.25)),
        ("金叉+RSI30-50+BB<0.3", golden & (rsi>=30) & (rsi<=50) & (bb_pct<0.3)),
        ("金叉+RSI30-50+pos<0.3", golden & (rsi>=30) & (rsi<=50) & (pos<0.3)),
        ("金叉+RSI35-55+BB<0.3+pos<0.35", golden & (rsi>=35) & (rsi<=55) & (bb_pct<0.3) & (pos<0.35)),
        ("金叉+RSI40-55+BB<0.3+pos<0.4", golden & (rsi>=40) & (rsi<=55) & (bb_pct<0.3) & (pos<0.4)),
        ("金叉+RSI<50+BB<0.3+pos<0.3", golden & (rsi<50) & (bb_pct<0.3) & (pos<0.3)),
        ("金叉+vol>1.2+BB<0.3", golden & (vol_ratio>1.2) & (bb_pct<0.3)),
        ("金叉+vol>1.3+pos<0.3+BB<0.3", golden & (vol_ratio>1.3) & (pos<0.3) & (bb_pct<0.3)),
    ]

    for pat_name, mask in buy_conditions:
        idxs = np.where(mask & (np.arange(n) < n-64))[0]  # 至少要有64根bar(~2天)的未来数据
        if len(idxs) < 8:
            continue

        # 对每个买点, 计算:
        # 1) 未来1日(16bar), 2日(32bar), 3日(48bar)的涨跌
        # 2) 买点价格与期间最低价的关系
        scores = []
        for h_bars, h_name in [(16,"1日"),(32,"2日"),(48,"3日")]:
            valid_idxs = idxs[idxs < n - h_bars]
            if len(valid_idxs) < 8: continue

            fwd_ret = np.array([(close[i+h_bars]-close[i])/close[i]*100 for i in valid_idxs])
            wr = (fwd_ret > 0).mean() * 100

            # 计算买入点与N日后最低价的距离
            near_lows = []
            for i in valid_idxs:
                future_lows = low[i:i+h_bars+1]
                min_price = np.min(future_lows)
                buy_price = close[i]
                pct_above_low = (buy_price - min_price) / min_price * 100
                near_lows.append(pct_above_low)

            avg_near_low = np.mean(near_lows)  # 越小越精确!
            p50_near_low = np.percentile(near_lows, 50)
            perfect_pct = (np.array(near_lows) <= 2.0).mean() * 100  # 距最低点<2%的比例

            scores.append({
                'name': pat_name, 'h': h_name, 'n': len(valid_idxs),
                'wr': round(wr,1), 'avg_ret': round(fwd_ret.mean(),2),
                'near_low_pct': round(avg_near_low, 2),  # 平均距最低点%
                'within_2pct': round(perfect_pct,1),
                'median_ret': round(np.median(fwd_ret),2),
            })

        # 取最佳时间窗口
        if scores:
            best = max(scores, key=lambda x: (x['wr']>=80, x['near_low_pct']<3, x['wr']))
            if best['wr'] >= 75:
                results.append(best)

    results.sort(key=lambda x: (x['wr'], -x['near_low_pct']), reverse=True)
    return results


def find_sell_quality(df, code, name):
    """搜索最优卖点: 信号后价格在未来N日最高价附近"""
    close = df['close'].values; high = df['high'].values
    low = df['low'].values; volume = df['volume'].values; n = len(close)
    golden, dead, rsi, bb_pct, pos, vol_ratio = calc_features(close, high, low, volume)

    results = []

    sell_conditions = [
        ("RSI>70+pos>0.7", (rsi>70) & (pos>0.7)),
        ("RSI>70+BB>0.8+pos>0.7", (rsi>70) & (bb_pct>0.8) & (pos>0.7)),
        ("RSI>75+BB>0.8+pos>0.7", (rsi>75) & (bb_pct>0.8) & (pos>0.7)),
        ("RSI>70+BB>0.85+pos>0.75", (rsi>70) & (bb_pct>0.85) & (pos>0.75)),
        ("RSI>75+BB>0.8+pos>0.8", (rsi>75) & (bb_pct>0.8) & (pos>0.8)),
        ("RSI>70+BB>0.85+pos>0.8", (rsi>70) & (bb_pct>0.85) & (pos>0.8)),
        ("RSI>65+BB>0.85+pos>0.8", (rsi>65) & (bb_pct>0.85) & (pos>0.8)),
        ("RSI>70+BB>0.9+pos>0.8", (rsi>70) & (bb_pct>0.9) & (pos>0.8)),
        ("死叉+RSI>65", dead & (rsi>65)),
        ("死叉+RSI>65+pos>0.7", dead & (rsi>65) & (pos>0.7)),
    ]

    for pat_name, mask in sell_conditions:
        idxs = np.where(mask & (np.arange(n) < n-32))[0]
        if len(idxs) < 8: continue

        scores = []
        for h_bars, h_name in [(16,"1日"),(32,"2日")]:
            valid_idxs = idxs[idxs < n - h_bars]
            if len(valid_idxs) < 8: continue

            fwd_ret = np.array([(close[i+h_bars]-close[i])/close[i]*100 for i in valid_idxs])
            fall_rate = (fwd_ret < 0).mean() * 100

            # 卖出点与期间最高价的距离
            near_highs = []
            for i in valid_idxs:
                future_highs = high[i:i+h_bars+1]
                max_price = np.max(future_highs)
                sell_price = close[i]
                pct_below_high = (max_price - sell_price) / max_price * 100
                near_highs.append(pct_below_high)

            avg_near_high = np.mean(near_highs)
            perfect_pct = (np.array(near_highs) <= 2.0).mean() * 100

            scores.append({
                'name': pat_name, 'h': h_name, 'n': len(valid_idxs),
                'fall': round(fall_rate,1),
                'avg_ret': round(fwd_ret.mean(),2),
                'near_high_pct': round(avg_near_high, 2),
                'within_2pct': round(perfect_pct,1),
            })

        if scores:
            best = max(scores, key=lambda x: (x['fall']>=30, x['near_high_pct']<3, x['fall']))
            if best['fall'] >= 30:
                results.append(best)

    results.sort(key=lambda x: (x['fall'], -x['near_high_pct']), reverse=True)
    return results


# ====== MAIN ======
all_findings = {}

for code, name in STOCKS.items():
    print(f"\n{'='*70}")
    print(f"  {name} ({code})")
    print(f"{'='*70}")

    df = pd.read_csv(f"{CACHE}/{code}_15min.csv", dtype={'time': str})
    df['time'] = df['time'].astype(str).str.zfill(17)
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    buys = find_buy_quality(df, code, name)
    sells = find_sell_quality(df, code, name)

    all_findings[code] = {'name': name, 'buys': buys, 'sells': sells}

    print(f"\n  [买入] 最精确的买点 (距最低点近 + 高胜率):")
    for b in buys[:5]:
        bar = '#'*min(int(b['wr']/2), 40)
        star = '★' if b['near_low_pct'] < 2 else ('●' if b['near_low_pct'] < 3 else '')
        print(f"    {star} {b['name']:45s} | {b['h']} WR={b['wr']:.1f}% {bar}")
        print(f"      距最低点平均{b['near_low_pct']:.1f}% | 精确(<2%):{b['within_2pct']:.0f}% | avg={b['avg_ret']:+.2f}% | n={b['n']}")

    print(f"\n  [卖出] 最精确的卖点 (距最高点近 + 高下跌):")
    for s in sells[:5]:
        bar = '-'*min(int(s['fall']/2), 30)
        star = '★' if s['near_high_pct'] < 2 else ('●' if s['near_high_pct'] < 3 else '')
        print(f"    {star} {s['name']:45s} | {s['h']} FALL={s['fall']:.1f}% {bar}")
        print(f"      距最高点平均{s['near_high_pct']:.1f}% | 精确(<2%):{s['within_2pct']:.0f}% | ret={s['avg_ret']:+.2f}% | n={s['n']}")


# ====== Generate optimized rules ======
print(f"\n\n{'='*70}")
print(f"  Per-stock precision buy/sell rules")
print(f"{'='*70}")

rules = {}
for code, f in all_findings.items():
    name = f['name']; buys = f['buys']; sells = f['sells']

    # Best buy: highest WR with near_low_pct < 3%
    strict_buys = [b for b in buys if b['near_low_pct'] < 3 and b['wr'] >= 80]
    best_buy = strict_buys[0] if strict_buys else (buys[0] if buys else None)

    # Best sell: highest FALL with near_high_pct < 3%
    strict_sells = [s for s in sells if s['near_high_pct'] < 3 and s['fall'] >= 35]
    best_sell = strict_sells[0] if strict_sells else (sells[0] if sells else None)

    rules[code] = {'name': name, 'buy': best_buy, 'sell': best_sell}

    b = best_buy; s = best_sell
    if b:
        print(f"\n{name}: 买入 → {b['name']} | {b['h']}WR={b['wr']:.1f}% 距底{b['near_low_pct']:.1f}% avg+{b['avg_ret']:.2f}%")
    if s:
        print(f"       卖出 → {s['name']} | {s['h']}FALL={s['fall']:.1f}% 距顶{s['near_high_pct']:.1f}% avg{s['avg_ret']:+.2f}%")

with open(f"{CACHE}/precision_rules.json", "w") as f:
    json.dump(all_findings, f, ensure_ascii=False, indent=2, default=str)

print(f"\nSaved: {CACHE}/precision_rules.json")
