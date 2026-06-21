"""
历史走势盈利概率点位分析
逐票分析:
  1. 把5年走势分段，找每个"买入-持有N天"的收益分布
  2. 在不同技术条件下(金叉/死叉/超买/超卖/放量)统计胜率
  3. 最优持仓天数搜索
  4. 盈亏概率热力图
"""
import sys,io,json,logging,warnings
from pathlib import Path
import pandas as pd,numpy as np

sys.path.insert(0,str(Path(__file__).parent))
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

from data.baostock_fetcher import fetch_all_daily_klines
from features.technical import compute_all_technical_features
from config import STOCK_CODES,STOCK_POOL

print('='*70)
print('  历史走势盈利概率点位分析')
print('='*70)

daily_data=fetch_all_daily_klines(5)

# 持N天收益分布
HOLD_DAYS=[1,3,5,10,20,40]
INIT_CAP=100000
BET_PCT=0.20

for code in STOCK_CODES:
    name=STOCK_POOL[code]['name']
    df=daily_data.get(code)
    if df is None or df.empty:continue
    df=compute_all_technical_features(df,'daily')

    print(f'\n')
    print(f'  {code} {name}')
    print(f'')

    # === 1. 基础概率 ===
    print(f'\n  >>> 1. 任意点买入N天后收益率分布 <<<')
    print(f'  {\"持有\":<8} {\"均收益\":<10} {\"中位收益\":<10} {\"胜率\":<8} {\"P10\":<10} {\"P90\":<10}')
    print(f'  {\"-\"*54}')

    for hd in HOLD_DAYS:
        df['fwd_ret']=df['close'].shift(-hd)/df['close']-1
        valid=df['fwd_ret'].dropna()
        rets=valid*100
        wr=(rets>0).mean()*100
        p10=np.percentile(rets,10)
        p90=np.percentile(rets,90)
        print(f'  {f\"{hd}天\":<8} {rets.mean():+10.2f}% {np.median(rets):+10.2f}% {wr:<8.1f}% {p10:+10.2f}% {p90:+10.2f}%')

    # === 2. 条件胜率 ===
    print(f'\n  >>> 2. 不同技术条件下的胜率(持5天) <<<')
    df['fwd5']=df['close'].shift(-5)/df['close']-1
    df['win5']=(df['fwd5']>0).astype(int)

    conditions={
        '任意买入': df.index.tolist(),
        'MA多头': df[df.get('ma_bullish',pd.Series(0))==1].index.tolist(),
        'MA空头': df[df.get('ma_bearish',pd.Series(0))==1].index.tolist(),
        'MACD金叉': df[df.get('macd_golden_cross',pd.Series(0))==1].index.tolist(),
        'MACD死叉': df[df.get('macd_dead_cross',pd.Series(0))==1].index.tolist(),
        'RSI超卖(<30)': df[(df.get('rsi',pd.Series(50))<30)].index.tolist(),
        'RSI超买(>70)': df[(df.get('rsi',pd.Series(50))>70)].index.tolist(),
        '放量(>2倍)': df[df.get('volume_surge',pd.Series(0))==1].index.tolist(),
        '布林下轨(<0.2)': df[(df.get('boll_pct_b',pd.Series(0.5))<0.2)].index.tolist(),
        '布林上轨(>0.8)': df[(df.get('boll_pct_b',pd.Series(0.5))>0.8)].index.tolist(),
        'MA多头+MACD金叉': df[(df.get('ma_bullish',pd.Series(0))==1)&(df.get('macd_golden_cross',pd.Series(0))==1)].index.tolist(),
        'MA多头+RSI适中(30-65)': df[(df.get('ma_bullish',pd.Series(0))==1)&(df.get('rsi',pd.Series(50)).between(30,65))].index.tolist(),
        'MA多头+MACD金叉+放量': df[(df.get('ma_bullish',pd.Series(0))==1)&(df.get('macd_golden_cross',pd.Series(0))==1)&(df.get('volume_surge',pd.Series(0))==1)].index.tolist(),
    }

    print(f'  {\"条件\":<22} {\"出现次数\":<10} {\"胜率\":<8} {\"均收益\":<10} {\"盈均值\":<10} {\"亏均值\":<10}')
    print(f'  {\"-\"*68}')

    best_cond=None;best_wr=0;best_cnt=0
    for cond_name,idx_list in conditions.items():
        total=len(idx_list)
        if total<3:continue
        subset=df.loc[idx_list].dropna(subset=['fwd5'])
        if len(subset)<3:continue
        wr=subset['win5'].mean()*100
        avg_ret=subset['fwd5'].mean()*100
        wins=subset[subset['fwd5']>0]['fwd5'].mean()*100 if (subset['fwd5']>0).any() else 0
        losses=subset[subset['fwd5']<=0]['fwd5'].mean()*100 if (subset['fwd5']<=0).any() else 0

        if wr>best_wr and total>=5:
            best_wr=wr;best_cond=cond_name;best_cnt=total

        print(f'  {cond_name:<22} {total:<10} {wr:<8.1f}% {avg_ret:+10.2f}% {wins:+10.2f}% {losses:+10.2f}%')

    print(f'\n  >>> 最优条件: \"{best_cond}\" → 出现{best_cnt}次, 胜率{best_wr:.1f}%')

    # === 3. 持股天数vs胜率曲线 ===
    print(f'\n  >>> 3. 最优条件下的持股天数vs胜率 <<<')

    idx_list=conditions[best_cond]
    subset=df.loc[idx_list].copy()

    print(f'  {\"持有\":<8} {\"胜率\":<8} {\"均收益\":<10} {\"最大收益\":<10} {\"最大亏损\":<10}')
    print(f'  {\"-\"*44}')

    for hd in [1,2,3,5,7,10,15,20,30,40,60]:
        col=f'fwd{hd}'
        subset[col]=subset['close'].shift(-hd)/subset['close']-1
        valid=subset[col].dropna()
        if len(valid)<3:continue
        wr=(valid>0).mean()*100
        avg=valid.mean()*100
        best2=valid.max()*100
        worst=valid.min()*100
        print(f'  {f\"{hd}天\":<8} {wr:<8.1f}% {avg:+10.2f}% {best2:+10.2f}% {worst:+10.2f}%')

    # === 4. 全量概率热力(简化) ===
    print(f'\n  >>> 4. 买入评分×持股天数 → 胜率矩阵 <<<')

    bins=[0,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.8,1.0]
    labels=['0-.3','.3-.4','.4-.5','.5-.55','.55-.6','.6-.65','.65-.7','.7-.8','.8-1']

    # 计算综合评分
    df['comp']=0.0
    for i in range(max(60,len(df)//10),len(df)):
        tb=0
        if all(f'ma_{p}' in df.columns for p in [5,10,20]):
            if df['ma_5'].iloc[i]>df['ma_10'].iloc[i]>df['ma_20'].iloc[i]:tb+=2
            elif df['ma_5'].iloc[i]>df['ma_10'].iloc[i]:tb+=1
        if 'ma_60' in df.columns and df['close'].iloc[i]>df['ma_60'].iloc[i]:tb+=1
        if 'macd_golden_cross' in df.columns and df['macd_golden_cross'].iloc[i]:tb+=2
        if 'rsi' in df.columns and 30<df['rsi'].iloc[i]<75:tb+=1
        if 'boll_pct_b' in df.columns and 0.2<df['boll_pct_b'].iloc[i]<0.9:tb+=1
        ts=min(tb/7.0,1.0)
        trb=0
        if 'pct_change_5' in df.columns and df['pct_change_5'].iloc[i]>0:trb+=1
        if 'ma_20' in df.columns and df['close'].iloc[i]>df['ma_20'].iloc[i]:trb+=1
        trs=min(trb/2.0,1.0)
        df.at[df.index[i],'comp']=0.5*ts+0.3*trs+0.2*0.5

    for hd in [1,5,10,20]:
        cname=f'fwd{hd}'
        df[cname]=df['close'].shift(-hd)/df['close']-1

    print(f'  {\"评分区间\":<8}',end='')
    for hd in [1,5,10,20]:print(f'  {\"T+\"+str(hd)+\"天\":>10}',end='')
    print()

    valid=df.dropna(subset=['fwd1','fwd5','fwd10','fwd20'])
    for j in range(len(labels)):
        mask=(valid['comp']>=bins[j])&(valid['comp']<bins[j+1])
        zone=valid[mask]
        if len(zone)<10:
            print(f'  {labels[j]:<8} {\"-\":>10} {\"-\":>10} {\"-\":>10} {\"-\":>10}')
            continue
        print(f'  {labels[j]:<8}',end='')
        for hd in [1,5,10,20]:
            cname=f'fwd{hd}'
            wr=(zone[cname]>0).mean()*100
            avg=zone[cname].mean()*100
            print(f'  {f\"{wr:.0f}%({avg:+.1f}%)\":>10}',end='')
        print()
