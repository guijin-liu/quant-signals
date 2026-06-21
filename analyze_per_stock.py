"""
逐票历史规律挖掘 — 用数据说话
每只票独立分析:
  1. 什么指标组合在买入后5天走势最好
  2. 哪些指标值的历史区间胜率>80%
  3. 每只票独立的参数
"""
import sys, io, json, logging, warnings
from datetime import datetime
from pathlib import Path
import pandas as pd, numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

from data.baostock_fetcher import fetch_all_daily_klines
from features.technical import compute_all_technical_features
from config import STOCK_CODES, STOCK_POOL

print('='*70)
print('  逐票历史规律挖掘 — 寻找>80%胜率的信号组合')
print('='*70)

daily_data = fetch_all_daily_klines(5)

# 信号定义: 买入后N天涨了就算赢
HOLDING_DAYS = 5  # 持5个交易日

all_findings = {}

for code in STOCK_CODES:
    name = STOCK_POOL[code]['name']
    df = daily_data.get(code)
    if df is None or df.empty:
        continue

    df = compute_all_technical_features(df, 'daily')

    # 未来N天收益
    df['future_return'] = df['close'].shift(-HOLDING_DAYS) / df['close'] - 1
    df['is_win'] = (df['future_return'] > 0).astype(int)
    df['return_pct'] = df['future_return'] * 100

    print(f'\n{"="*70}')
    print(f'  {code} {name} — 5年数据, {len(df)}条日线')
    print(f'{"="*70}')

    # === 1. 基础统计 ===
    overall_win = df['is_win'].mean() * 100
    avg_return = df['return_pct'].mean()
    volatility = df['close'].pct_change().std() * 100
    print(f'  基础: 自然胜率={overall_win:.1f}% 日均收益={avg_return:+.3f}% 波动率={volatility:.2f}%')

    # 趋势: 牛市vs熊市
    df['ma20'] = df['close'].rolling(20).mean()
    df['trend'] = 'sideways'
    df.loc[df['close'] > df['ma20'] * 1.05, 'trend'] = 'bull'
    df.loc[df['close'] < df['ma20'] * 0.95, 'trend'] = 'bear'

    for trend in ['bull', 'sideways', 'bear']:
        subset = df[df['trend'] == trend]
        wr = subset['is_win'].mean() * 100 if len(subset) > 20 else 0
        print(f'    {trend:>8}: {len(subset)}条, 胜率={wr:.1f}%')

    # === 2. 寻找高胜率信号组合 ===
    print(f'\n  >>> 高胜率信号挖掘 (目标>80%) <<<')
    findings = []

    # 2a. MA金叉质量
    if 'ma_golden_cross' in df.columns:
        golden = df[df['ma_golden_cross'] == 1]
        if len(golden) > 10:
            wr = golden['is_win'].mean() * 100
            avg_r = golden['return_pct'].mean()
            print(f'  MA金叉: {len(golden)}次, 胜率={wr:.1f}%, 均收益={avg_r:+.2f}%')

            # 金叉在不同趋势下
            for trend in ['bull', 'sideways', 'bear']:
                g_in = golden[golden['trend'] == trend]
                if len(g_in) > 5:
                    wr_t = g_in['is_win'].mean() * 100
                    findings.append({
                        'signal': f'金叉+{trend}',
                        'count': len(g_in),
                        'win_rate': wr_t,
                        'avg_return': g_in['return_pct'].mean(),
                    })
                    print(f'    金叉+{trend}: {len(g_in)}次, 胜率={wr_t:.1f}%')

    # 2b. MA多头排列
    if 'ma_bullish' in df.columns:
        bull_ma = df[df['ma_bullish'] == 1]
        if len(bull_ma) > 10:
            wr = bull_ma['is_win'].mean() * 100
            print(f'  MA多头排列: {len(bull_ma)}条, 胜率={wr:.1f}%')

            for trend in ['bull', 'sideways', 'bear']:
                b_in = bull_ma[bull_ma['trend'] == trend]
                if len(b_in) > 5:
                    wr_t = b_in['is_win'].mean() * 100
                    findings.append({
                        'signal': f'MA多头+{trend}',
                        'count': len(b_in),
                        'win_rate': wr_t,
                        'avg_return': b_in['return_pct'].mean(),
                    })

    # 2c. RSI区间扫描
    if 'rsi' in df.columns:
        for lo, hi in [(20,30),(30,40),(40,50),(50,60),(60,70),(70,80)]:
            rsi_zone = df[(df['rsi']>=lo)&(df['rsi']<hi)]
            if len(rsi_zone) > 20:
                wr = rsi_zone['is_win'].mean() * 100
                if wr > 55 or wr < 45:  # 只记录有意外的
                    findings.append({
                        'signal': f'RSI_{lo}-{hi}',
                        'count': len(rsi_zone),
                        'win_rate': wr,
                        'avg_return': rsi_zone['return_pct'].mean(),
                    })

    # 2d. 布林带位置
    if 'boll_pct_b' in df.columns:
        for lo, hi, label in [(0,0.2,'下轨'),(0.2,0.4,'中下'),(0.4,0.6,'中轨'),
                              (0.6,0.8,'中上'),(0.8,1.0,'上轨')]:
            bb = df[(df['boll_pct_b']>=lo)&(df['boll_pct_b']<hi)]
            if len(bb) > 20:
                wr = bb['is_win'].mean() * 100
                findings.append({
                    'signal': f'BB_{label}',
                    'count': len(bb),
                    'win_rate': wr,
                    'avg_return': bb['return_pct'].mean(),
                })

    # 2e. 成交量
    if 'volume_surge' in df.columns:
        vs = df[df['volume_surge']==1]
        if len(vs) > 10:
            wr = vs['is_win'].mean() * 100
            findings.append({
                'signal': '放量(>2倍)',
                'count': len(vs),
                'win_rate': wr,
                'avg_return': vs['return_pct'].mean(),
            })

    # 2f. 复合信号: MA多头 + RSI健康 + 放量
    if all(c in df.columns for c in ['ma_bullish','rsi']):
        compound = df[(df['ma_bullish']==1)&(df['rsi'].between(30,65))]
        for trend in ['bull','sideways','bear']:
            c_in = compound[compound['trend']==trend]
            if len(c_in) > 5:
                wr_c = c_in['is_win'].mean() * 100
                findings.append({
                    'signal': f'MA多头+RSI30-65+{trend}',
                    'count': len(c_in),
                    'win_rate': wr_c,
                    'avg_return': c_in['return_pct'].mean(),
                })

        # 再加成交量
        if 'volume_surge' in df.columns:
            comp2 = df[(df['ma_bullish']==1)&(df['rsi'].between(30,65))&(df['volume_surge']==1)]
            for trend in ['bull','sideways','bear']:
                c2_in = comp2[comp2['trend']==trend]
                if len(c2_in) > 3:
                    wr_c2 = c2_in['is_win'].mean() * 100
                    findings.append({
                        'signal': f'MA多头+RSI30-65+放量+{trend}',
                        'count': len(c2_in),
                        'win_rate': wr_c2,
                        'avg_return': c2_in['return_pct'].mean(),
                    })

    # 2g. MACD+MA双重
    if all(c in df.columns for c in ['macd_golden_cross','ma_bullish']):
        for trend in ['bull','sideways','bear']:
            dbl = df[(df['macd_golden_cross']==1)&(df['ma_bullish']==1)&(df['trend']==trend)]
            if len(dbl) > 3:
                wr_d = dbl['is_win'].mean() * 100
                findings.append({
                    'signal': f'MACD金叉+MA多头+{trend}',
                    'count': len(dbl),
                    'win_rate': wr_d,
                    'avg_return': dbl['return_pct'].mean(),
                })

    # 排序
    findings.sort(key=lambda x: x['win_rate'], reverse=True)

    # 输出>75%胜率的信号
    high_wr = [f for f in findings if f['win_rate'] >= 75 and f['count'] >= 5]
    print(f'\n  >>> 胜率>=75%且至少5次出现的信号 ({len(high_wr)}个) <<<')

    if high_wr:
        for f in high_wr[:10]:
            sig = f['signal']
            cnt = f['count']
            wr = f['win_rate']
            ar = f['avg_return']
            print(f'    [{sig}] {cnt}次 胜率={wr:.1f}% 均收益={ar:+.2f}%')
    else:
        print(f'    (无符合条件的信号)')
        print(f'    最接近的Top 10:')
        for f in findings[:10]:
            sig = f['signal']
            cnt = f['count']
            wr = f['win_rate']
            ar = f['avg_return']
            print(f'    [{sig}] {cnt}次 胜率={wr:.1f}% 均收益={ar:+.2f}%')

    all_findings[code] = {
        'name': name,
        'overall_win': overall_win,
        'volatility': volatility,
        'top_signals': findings[:20],
    }

# === 汇总 ===
print(f'\n\n{"="*70}')
print(f'  4只票对比总结')
print(f'{"="*70}')
for code, f in all_findings.items():
    top1 = f['top_signals'][0] if f['top_signals'] else None
    top_wr = top1['win_rate'] if top1 else 0
    name2 = f['name']
    ow = f['overall_win']
    vol = f['volatility']
    print(f'  {code} {name2}: 自然胜率={ow:.1f}% 波动={vol:.2f}% 最佳信号胜率={top_wr:.1f}%')

# 保存
out = Path(__file__).parent / 'reports' / f'signal_patterns_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w', encoding='utf-8') as f:
    json.dump(all_findings, f, ensure_ascii=False, indent=2, default=str)
print(f'\n详细报告: {out}')
