"""回溯挖掘引擎 v2.0 — 腾讯日线 + 沿用v16评分框架
逐票回溯 → 筛选≥77%胜率的评分阈值 → 计算7折卖点
"""
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import json, os, time
from stock_pool import STOCK_POOL

UA = "Mozilla/5.0"
POOL = STOCK_POOL  # v16.1: 直接使用股票池

def _prefix(code):
    return "sh" if code.startswith(("6","9")) else "sz"

# ═══════════════════════════════════════
# 腾讯日线
# ═══════════════════════════════════════

def fetch_daily_tencent(code):
    """腾讯日线，最多~640条(2.5年)"""
    pfx = _prefix(code)
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{pfx}{code},day,,,2000,qfq"}
    try:
        r = requests.get(url, params=params,
                         headers={"User-Agent": UA, "Referer": "https://gu.qq.com/"}, timeout=15)
        d = r.json()
        rows = d.get("data", {}).get(f"{pfx}{code}", {}).get("qfqday", []) or \
               d.get("data", {}).get(f"{pfx}{code}", {}).get("day", [])
        if not rows: return pd.DataFrame()
        data = []
        for row in rows:
            if len(row) < 6: continue
            data.append({"date": str(row[0]), "open": float(row[1]), "close": float(row[2]),
                         "high": float(row[3]), "low": float(row[4]), "volume": float(row[5])})
        df = pd.DataFrame(data)
        for c in ["open","close","high","low","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna()
    except Exception as e:
        print(f"  腾讯API异常: {e}")
        return pd.DataFrame()

# ═══════════════════════════════════════
# 特征计算（与cloud_function.py完全一致）
# ═══════════════════════════════════════

def compute_features(close, high, low, vol):
    n = len(close)
    if n < 60: return None
    f = {}
    f['close'] = close[-1]
    f['ma5']  = np.mean(close[-5:])
    f['ma10'] = np.mean(close[-10:])
    f['ma20'] = np.mean(close[-20:])
    f['ma60'] = np.mean(close[-min(60,n):])
    p_ma5 = np.mean(close[-6:-1]); p_ma10 = np.mean(close[-11:-1])
    f['golden'] = (f['ma5'] > f['ma10']) and (p_ma5 <= p_ma10)
    f['dead']   = (f['ma5'] < f['ma10']) and (p_ma5 >= p_ma10)
    deltas = np.diff(close[-15:])
    g = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    l = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 1e-9
    f['rsi'] = 100 - 100/(1+g/l) if l > 0 else 50
    bb_m = np.mean(close[-20:]); bb_s = np.std(close[-20:])
    bb_u = bb_m + 2*bb_s; bb_l = bb_m - 2*bb_s
    f['bb_pct'] = max(0.0, min(1.0, (close[-1] - bb_l) / (bb_u - bb_l + 0.0001)))
    f['bb_width'] = (bb_u - bb_l) / bb_m
    h20 = np.max(high[-20:]); l20 = np.min(low[-20:])
    f['pos'] = max(0.0, min(1.0, (close[-1] - l20) / (h20 - l20 + 0.0001)))
    f['vol_ratio'] = np.mean(vol[-5:]) / (np.mean(vol[-20:]) + 1)
    f['vol_trend'] = np.mean(vol[-10:]) / (np.mean(vol[-30:]) + 1) if n >= 30 else 1.0
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    dif = ema12 - ema26; dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    f['macd'] = 2*(dif[-1] - dea[-1])
    f['macd_up'] = dif[-1] > dea[-1]
    h9 = np.max(high[-9:]); l9 = np.min(low[-9:])
    rsv = (close[-1] - l9) / (h9 - l9 + 0.0001) * 100
    f['k'] = 2/3*50 + 1/3*rsv
    f['d'] = 2/3*50 + 1/3*f['k']; f['j'] = 3*f['k'] - 2*f['d']
    trs = []
    for i in range(-14, 0):
        h, lv, pc = high[i], low[i], close[i-1]
        trs.append(max(h-lv, abs(h-pc), abs(lv-pc)))
    f['atr'] = np.mean(trs)
    obv_changes = [vol[i] if close[i] > close[i-1] else (-vol[i] if close[i] < close[i-1] else 0) for i in range(-10, 0)]
    f['obv_up'] = sum(obv_changes) > 0
    f['above_ma20'] = close[-1] > f['ma20']
    f['above_ma60'] = close[-1] > f['ma60']
    return f

# ═══════════════════════════════════════
# v16评分（与cloud_function.py 100%一致）
# ═══════════════════════════════════════

def score_v16(f):
    """返回 (total_score, reason_list)"""
    score = 0; reasons = []
    if f['golden']:
        score += 2; reasons.append("金叉")
    else:
        return score, reasons  # 无金叉 = 0分（保持v16策略核心）
    if f['bb_pct'] <= 0.25:
        score += 2; reasons.append("BB低位")
    elif f['bb_pct'] <= 0.35:
        score += 1; reasons.append("BB中低位")
    elif f['bb_pct'] > 0.7:
        return 0, []  # BB高位不买
    if 40 <= f['rsi'] <= 60:
        score += 1; reasons.append("RSI健康")
    if f['j'] < 30:
        score += 1; reasons.append("J值超卖")
    elif f['j'] > 80:
        return 0, []  # J值超买不买
    if f['vol_ratio'] > 1.0:
        score += 1; reasons.append("放量")
    if f['obv_up']:
        score += 1; reasons.append("OBV向上")
    if f['above_ma60']:
        score += 1; reasons.append("多头趋势")
    if f['close'] > f['ma20'] * 1.05:
        return 0, []  # 追高不买
    return score, reasons

# ═══════════════════════════════════════
# 单票回溯
# ═══════════════════════════════════════

def backtest_stock(code, name):
    df = fetch_daily_tencent(code)
    if df.empty or len(df) < 120:
        return None, f"数据不足({len(df)}条)"

    close = df['close'].values; high = df['high'].values
    low = df['low'].values; vol = df['volume'].values

    # 按评分分组收集信号
    score_signals = defaultdict(list)

    for i in range(60, len(df) - 20):
        f = compute_features(close[i-60:i+1], high[i-60:i+1], low[i-60:i+1], vol[i-60:i+1])
        if not f: continue
        s, reasons = score_v16(f)
        if s < 3: continue  # v16原版阈值
        entry = close[i]
        forward_high = high[i+1:i+21]
        max_gain = (np.max(forward_high) - entry) / entry * 100
        score_signals[s].append({
            'date': str(df['date'].iloc[i])[:10],
            'reasons': '+'.join(reasons),
            'entry': round(entry, 2),
            'max_gain': round(max_gain, 2),
            'win_1pct': max_gain >= 1.0,
            'win_2pct': max_gain >= 2.0,
            'win_3pct': max_gain >= 3.0,
            'win_0pct': max_gain > 0,
        })

    # 按评分统计
    results = []
    for score, trades in sorted(score_signals.items()):
        n = len(trades)
        if n < 5: continue
        wr_1pct = sum(1 for t in trades if t['win_1pct']) / n
        wr_2pct = sum(1 for t in trades if t['win_2pct']) / n
        wr_0pct = sum(1 for t in trades if t['win_0pct']) / n
        gain_list = [t['max_gain'] for t in trades]
        p50 = np.percentile(gain_list, 50)
        p60 = np.percentile(gain_list, 60)
        p70 = np.percentile(gain_list, 70)
        p80 = np.percentile(gain_list, 80)
        p90 = np.percentile(gain_list, 90)

        # 胜者分布（≥1%）
        win_gains = [t['max_gain'] for t in trades if t['win_1pct']]
        win_p70 = np.percentile(win_gains, 70) if win_gains else 0
        win_p80 = np.percentile(win_gains, 80) if win_gains else 0

        results.append({
            'score': score,
            'n': n,
            'wr_1pct': round(wr_1pct*100, 1),
            'wr_2pct': round(wr_2pct*100, 1),
            'wr_0pct': round(wr_0pct*100, 1),
            'p50': round(p50, 1), 'p70': round(p70, 1),
            'p80': round(p80, 1), 'p90': round(p90, 1),
            'sell_70': round(p70 * 0.7, 1),
            'sell_80': round(p80 * 0.7, 1),
            'avg_gain': round(np.mean(gain_list), 1),
            'sample_dates': f"{trades[0]['date']}~{trades[-1]['date']}",
            'top_reason': max(set(t['reasons'] for t in trades), key=lambda r: sum(1 for t in trades if t['reasons']==r)),
        })

    return results, None

# ═══════════════════════════════════════
# 主流程
# ═══════════════════════════════════════

def main():
    print(f"回溯挖掘引擎 v2.0 — 腾讯日线 | {len(POOL)}只票")
    print("策略: v16评分框架, 金叉必须, ≥3分触发")
    print(f"开始: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 85)

    all_configs = {}
    skipped = []

    for idx, (code, name) in enumerate(POOL.items()):
        print(f"[{idx+1:02d}/{len(POOL)}] {code} {name} ", end="", flush=True)
        try:
            results, err = backtest_stock(code, name)
            if err:
                print(f"  ⚠ {err}")
                skipped.append((code, name, err))
                continue

            # 找≥77%胜率的最高评分
            high_wr = [r for r in results if r['wr_1pct'] >= 77.0]
            best = high_wr[0] if high_wr else (results[0] if results else None)

            if best:
                wr = best['wr_1pct']
                tag = "★★★" if wr >= 77 else ("★★" if wr >= 70 else "★")
                print(f"{tag} 评分{best['score']} n={best['n']} 胜率(≥1%)={wr}% 卖点={best['sell_70']}%")
                if best['wr_2pct'] >= 60:
                    print(f"       ≥2%胜率={best['wr_2pct']}% P70={best['p70']}% P80={best['p80']}% | 信号特征: {best['top_reason']}")
            else:
                print(f"  无≥3分信号")

            all_configs[code] = {
                'name': name,
                'results': results,
                'best': best
            }
            sys.stdout.flush()

        except Exception as e:
            print(f"  ✗ {e}")
            skipped.append((code, name, str(e)))

        time.sleep(0.15)  # 腾讯API友好间隔

    # ═══ 汇总 ═══
    print("\n" + "=" * 85)
    print(f"回溯完成 {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 85)

    # 按胜率排序
    ranked = []
    for code, cfg in all_configs.items():
        if cfg['best']:
            ranked.append((code, cfg['name'], cfg['best']))
    ranked.sort(key=lambda x: x[2]['wr_1pct'], reverse=True)

    print(f"\n{'代码':<8} {'名称':<10} {'评分':>4} {'样本':>5} {'胜率≥1%':>8} {'胜率≥2%':>8} {'P70%':>6} {'卖点%':>6} 特征")
    print("-" * 95)
    for code, name, b in ranked:
        wr1 = b['wr_1pct']; wr2 = b['wr_2pct']
        star = "🔴" if wr1 >= 77 else ("🟡" if wr1 >= 70 else "⚪")
        print(f"{star} {code:<6} {name:<10} {b['score']:>4} {b['n']:>5} {wr1:>7.1f}% {wr2:>7.1f}% {b['p70']:>5.1f}% {b['sell_70']:>5.1f}% {b['top_reason'][:30]}")

    # 统计
    n_77 = sum(1 for _,_,b in ranked if b['wr_1pct'] >= 77)
    n_70 = sum(1 for _,_,b in ranked if 70 <= b['wr_1pct'] < 77)
    n_low = sum(1 for _,_,b in ranked if b['wr_1pct'] < 70)
    print(f"\n≥77%: {n_77}只 | 70-77%: {n_70}只 | <70%: {n_low}只")
    print(f"跳过: {len(skipped)}只")

    if skipped:
        for c, n, e in skipped:
            print(f"  {c} {n}: {e}")

    # ═══ 保存策略配置 ═══
    output = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'strategy': 'v16 — 7指标共振, ≥3分买入, 金叉必须',
        'data_source': '腾讯日线 (web.ifzq.gtimg.cn)',
        'total_stocks': len(POOL),
        'win_threshold': '≥1%涨幅为胜',
        'stocks_77pct': n_77,
        'stocks_70pct': n_70,
        'stocks_below_70': n_low,
        'configs': {}
    }

    for code, cfg in all_configs.items():
        if not cfg['best']: continue
        b = cfg['best']
        output['configs'][code] = {
            'name': cfg['name'],
            'score_threshold': b['score'],
            'win_rate_1pct': b['wr_1pct'],
            'win_rate_2pct': b['wr_2pct'],
            'p70_gain': b['p70'],
            'p80_gain': b['p80'],
            'sell_pct': b['sell_70'],  # P70 × 0.7
            'sell_pct_aggressive': b['sell_80'],  # P80 × 0.7
            'n_samples': b['n'],
            'avg_gain': b['avg_gain'],
            'signal_pattern': b['top_reason'],
        }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strategy_config.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n配置已保存: {out_path}")

    return all_configs

if __name__ == "__main__":
    import sys
    main()
