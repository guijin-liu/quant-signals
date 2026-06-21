"""
精简版历史走势概率分析 — 每票自己跑, 只算到T+20
"""
import sys,io,logging,warnings
import pandas as pd,numpy as np

sys.path.insert(0,'.')
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

from data.baostock_fetcher import fetch_all_daily_klines
from features.technical import compute_all_technical_features
from config import STOCK_CODES,STOCK_POOL

print('='*70)
print('  历史走势盈利概率点位分析')
print('='*70)

daily=fetch_all_daily_klines(5)

for code in STOCK_CODES:
    name=STOCK_POOL[code]['name']
    df=daily.get(code)
    if df is None or df.empty:continue
    df=compute_all_technical_features(df,'daily')
    sep='='*60
    print('\n'+sep)
    print('  {} — {} (5年 {}条)'.format(code,name,len(df)))
    print(sep)

    # T+N 全量
    print('\n  T+N 任意点的胜率与收益分布:')
    print('  {:>10} | {:>8} | {:>10} | {:>10} | {:>10}'.format('持有','胜率','均收益','P25','P75'))
    print('  '+'-'*55)

    for hd in [1,3,5,10,20,40]:
        col='hd{}'.format(hd)
        df[col]=df['close'].shift(-hd)/df['close']-1
        valid=df[col].dropna()
        rets=valid*100
        wr=(rets>0).mean()*100
        avg=rets.mean()
        p25=np.percentile(rets,25)
        p75=np.percentile(rets,75)
        print('  {:>10} | {:>7.1f}% | {:>+9.2f}% | {:>+9.2f}% | {:>+9.2f}%'.format(
            'T+{}'.format(hd),wr,avg,p25,p75))

    # 条件胜率 (持5天)
    df['fwd5']=df['close'].shift(-5)/df['close']-1
    df['win5']=(df['fwd5']>0).astype(int)

    conds=[]
    if 'ma_bullish' in df.columns:conds.append(('MA多头',df[df['ma_bullish']==1].index))
    if 'ma_bearish' in df.columns:conds.append(('MA空头',df[df['ma_bearish']==1].index))
    if 'macd_golden_cross' in df.columns:conds.append(('MACD金叉',df[df['macd_golden_cross']==1].index))
    if 'macd_dead_cross' in df.columns:conds.append(('MACD死叉',df[df['macd_dead_cross']==1].index))
    if 'rsi' in df.columns:
        conds.append(('RSI<30',df[df['rsi']<30].index))
        conds.append(('RSI>70',df[df['rsi']>70].index))
    if 'volume_surge' in df.columns:conds.append(('放量2x',df[df['volume_surge']==1].index))
    if 'boll_pct_b' in df.columns:
        conds.append(('BB下轨<.2',df[df['boll_pct_b']<0.2].index))
        conds.append(('BB上轨>.8',df[df['boll_pct_b']>0.8].index))
    if 'ma_bullish' in df.columns and 'macd_golden_cross' in df.columns:
        conds.append(('MA多+金叉',df[(df['ma_bullish']==1)&(df['macd_golden_cross']==1)].index))
    if 'ma_bullish' in df.columns and 'rsi' in df.columns:
        conds.append(('MA多+RSI30-65',df[(df['ma_bullish']==1)&(df['rsi'].between(30,65))].index))
    if 'ma_bullish' in df.columns and 'macd_golden_cross' in df.columns and 'volume_surge' in df.columns:
        conds.append(('MA多+金叉+放量',df[(df['ma_bullish']==1)&(df['macd_golden_cross']==1)&(df['volume_surge']==1)].index))

    print('\n  条件胜率(持5天):')
    print('  {:>20} | {:>6} | {:>7} | {:>9} | {:>7}'.format('条件','次数','胜率','均收益','盈亏比'))
    print('  '+'-'*58)

    best_wr=0;best_name='';best_idx=None
    for cn,idx in conds:
        total=len(idx)
        if total<3:continue
        sub=df.loc[idx].dropna(subset=['fwd5'])
        if len(sub)<3:continue
        wr=sub['win5'].mean()*100
        avg=sub['fwd5'].mean()*100
        w_avg=sub[sub['fwd5']>0]['fwd5'].mean()*100 if (sub['fwd5']>0).any() else 0
        l_avg=abs(sub[sub['fwd5']<=0]['fwd5'].mean()*100) if (sub['fwd5']<=0).any() else 1
        pf=w_avg/l_avg if l_avg>0 else 99
        print('  {:>20} | {:>6} | {:>6.1f}% | {:>+8.2f}% | {:>6.1f}'.format(cn,total,wr,avg,pf))
        if wr>best_wr:
            best_wr=wr;best_name=cn;best_idx=idx

    print('\n  >>> 最优: {} (WR={:.1f}%)'.format(best_name,best_wr))

    # 最优条件的持股天曲线
    if best_idx is not None:
        print('\n  {}在不同持股天下的表现:'.format(best_name))
        print('  {:>6} | {:>7} | {:>9} | {:>9} | {:>9}'.format('持有','胜率','均收益','最佳','最差'))
        print('  '+'-'*45)
        sub=df.loc[best_idx].copy()
        for hd in [1,2,3,5,7,10,15,20,30,40,60]:
            col2='hd_x'+str(hd)
            sub[col2]=sub['close'].shift(-hd)/sub['close']-1
            vl=sub[col2].dropna()
            if len(vl)<3:continue
            print('  {:>6} | {:>6.1f}% | {:>+8.2f}% | {:>+8.1f}% | {:>8.1f}%'.format(
                '{}天'.format(hd),(vl>0).mean()*100,vl.mean()*100,vl.max()*100,vl.min()*100))
