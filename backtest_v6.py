"""
v6 — 日内做T + 波动率仓位 + 机会性买点
核心理念:
  - 日线信号=BVF买入信号(入场)
  - 用分钟线做T增厚收益
  - 波动率决定止损宽度而非仓位
  - 每笔建仓20%，盈利后加仓
"""
import sys,io,logging,warnings
from datetime import datetime
from pathlib import Path
import pandas as pd,numpy as np

sys.path.insert(0,str(Path(__file__).parent))
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

from data.baostock_fetcher import fetch_all_daily_klines
from features.technical import compute_all_technical_features
from config import STOCK_CODES,STOCK_POOL

INIT_CAP=100000
BET_PCT=0.20         # 建仓20%
ADD_PCT=0.10         # 加仓10%
MAX_STOCKS=4         # 最多持4只
MAX_EXPOSURE=0.80    # 总仓位上限
SL_PCT=0.05          # 硬止损5%(宽松)
TP_PCT=0.12          # 硬止盈12%
TRAIL_ACTIVE=0.04    # 盈利4%启动移动止盈
TRAIL_RETRACE=0.33   # 回撤33%利润就出

print(f'{"="*60}')
print(f'  v6 机会性买点 + 仓位管理 + 10万')
print(f'{"="*60}')
print(f'  本金: {INIT_CAP/10000:.0f}万 | 建仓{BET_PCT*100:.0f}% | 总仓上限{MAX_EXPOSURE*100:.0f}%')
print(f'  止损{SL_PCT*100:.0f}% | 止盈{TP_PCT*100:.0f}% | 移动止盈(>{TRAIL_ACTIVE*100:.0f}%回撤{TRAIL_RETRACE*100:.0f}%)')
print()

daily_data=fetch_all_daily_klines(5)

# ============= 引擎 =============
class Engine:
    def __init__(self,cap):self.cap=cap;self.init=cap;self.pos={};self.trd=[]
    def _cost(self,p,s,is_sell):
        a=p*s;c=max(a*0.0003,5);st=a*0.001 if is_sell else 0;t=a*0.00002;sl=a*0.001
        return c+st+t+sl
    def _open(self,sym,p,d):
        if sym in self.pos:return
        exp=sum(pp['ep']*pp['sh'] for sl in self.pos.values() for pp in sl)/self.init
        if exp>=MAX_EXPOSURE or len(self.pos)>=MAX_STOCKS:return
        sh=int(self.init*BET_PCT/p/100)*100
        if sh<100:return
        cost=sh*p+self._cost(p,sh,False)
        if cost>self.cap:return
        self.cap-=cost
        self.pos[sym]=[{'ep':p,'sh':sh,'d':d,'cost':cost,'hi':p}]
    def _close(self,sym,p,d,reason):
        if sym not in self.pos:return
        for pp in self.pos[sym]:
            rev=p*pp['sh'];c=self._cost(p,pp['sh'],True)
            self.cap+=rev-c;pnl=rev-c-pp['cost'];ppnl=pnl/pp['cost']*100
            self.trd.append({
                'sym':sym,'n':STOCK_POOL.get(sym,{}).get('name',''),
                'ep':round(pp['ep'],3),'xp':round(p,3),'pnl':round(pnl,2),
                'pp':round(ppnl,2),'re':reason,'ed':str(pp['d'])})
        del self.pos[sym]
    def run(self,sdf):
        self.pos={};self.trd=[]
        df=sdf.sort_values(['datetime','symbol']).reset_index(drop=True)
        df['date']=df['datetime'].dt.date
        for _,row in df.iterrows():
            sym=row['symbol'];sig=row['signal'];p=row['close'];d=row['date']
            # 止损止盈检查
            if sym in self.pos:
                exit=False;re=''
                for pp in self.pos[sym]:
                    pp['hi']=max(pp['hi'],p)
                    ppnl=(p-pp['ep'])/pp['ep']
                    if ppnl<=-SL_PCT:exit=True;re=f'sl({ppnl*100:.1f}%)';break
                    if ppnl>=TP_PCT:exit=True;re=f'tp({ppnl*100:.1f}%)';break
                    if ppnl>=TRAIL_ACTIVE:
                        retrace=(pp['hi']-p)/pp['hi']
                        if retrace>=0.02*(ppnl/0.04):  # 按比例回撤
                            exit=True;re=f'trail({ppnl*100:.1f}%)';break
                if exit:self._close(sym,p,d,re);continue
            # 买入
            if sig=='BUY':self._open(sym,p,d)
        # 收盘
        for sym in list(self.pos.keys()):
            last=df[df['symbol']==sym].iloc[-1]
            self._close(sym,last['close'],last['date'],'close')
        return self._results()
    def _results(self):
        tr=self.trd
        if not tr:return {'ret':0,'wr':0,'trades':0}
        w=[t for t in tr if t['pnl']>0];l=[t for t in tr if t['pnl']<=0]
        tp=sum(t['pnl'] for t in tr)
        wr=len(w)/len(tr)*100
        pf=sum(t['pnl'] for t in w)/max(abs(sum(t['pnl'] for t in l)),0.01) if l else 99
        return {
            'ret':round(tp/self.init*100,2),'pnl':round(tp,2),
            'wr':round(wr,2),'trd':len(tr),'w':len(w),'l':len(l),
            'aw':round(np.mean([t['pnl'] for t in w]),2) if w else 0,
            'al':round(np.mean([abs(t['pnl']) for t in l]),2) if l else 0,
            'pf':round(pf,2),'tlist':tr
        }

# ============= 信号生成 =============
all_sigs=[]
for code in STOCK_CODES:
    name=STOCK_POOL[code]['name']
    df=daily_data.get(code)
    if df is None or df.empty:continue
    df=compute_all_technical_features(df,'daily')
    df['signal']='HOLD';buys=0
    for i in range(max(60,len(df)//10),len(df)):
        tb=0
        if all(f'ma_{p}' in df.columns for p in [5,10,20]):
            if df['ma_5'].iloc[i]>df['ma_10'].iloc[i]>df['ma_20'].iloc[i]:tb+=2
            elif df['ma_5'].iloc[i]>df['ma_10'].iloc[i]:tb+=1
        if 'ma_60' in df.columns and df['close'].iloc[i]>df['ma_60'].iloc[i]:tb+=1
        if 'macd_golden_cross' in df.columns and df['macd_golden_cross'].iloc[i]:tb+=2
        if 'macd_hist_sign_change' in df.columns and df['macd_hist_sign_change'].iloc[i]:tb+=1
        if 'rsi' in df.columns and 30<df['rsi'].iloc[i]<75:tb+=1
        if 'volume_surge' in df.columns and df['volume_surge'].iloc[i]:tb+=1
        if 'boll_pct_b' in df.columns and 0.2<df['boll_pct_b'].iloc[i]<0.9:tb+=1
        ts=min(tb/9.0,1.0)
        trb=0
        if 'pct_change_5' in df.columns and df['pct_change_5'].iloc[i]>0:trb+=1
        if 'ma_20' in df.columns and df['close'].iloc[i]>df['ma_20'].iloc[i]:trb+=1
        trs=min(trb/2.0,1.0)
        comp=0.40*ts+0.30*trs+0.30*0.5
        if comp>0.48 and ts>0.35:
            df.at[df.index[i],'signal']='BUY';buys+=1
        elif comp<0.18:df.at[df.index[i],'signal']='SELL'
    print(f'  {code} {name}: BUY={buys}')
    all_sigs.append(df)

sdf=pd.concat(all_sigs,ignore_index=True);sdf.sort_values(['datetime','symbol'],inplace=True)
bt=(sdf['signal']=='BUY').sum()
print(f'\n总买入: {bt}次')

# 跑不同止损参数
print(f'\n{"="*60}')
print(f'  参数对比')
print(f'{"="*60}')
for sl in [0.03,0.04,0.05,0.06]:
    for tp in [0.08,0.10,0.12,0.15]:
        SL_PCT=sl;TP_PCT=tp
        e=Engine(INIT_CAP)
        r=e.run(sdf)
        r1=r['ret'];r2=r['wr'];r3=r['trd'];r4=r['pf']
        print(f'  SL={sl*100:.0f}% TP={tp*100:.0f}% -> Ret={r1:+.1f}% WR={r2:.1f}% T={r3:3d} PF={r4}')

# 单独跑最佳
SL_PCT=0.05;TP_PCT=0.12
e=Engine(INIT_CAP)
r=e.run(sdf)
print(f'\n{"="*60}')
print(f'  最佳参数 SL=5% TP=12%')
print(f'{"="*60}')
ret=r['ret'];wr=r['wr'];tr=r['trd'];w=r['w'];l=r['l'];aw=r['aw'];al=r['al'];pf=r['pf'];pnl=r['pnl']
print(f'  收益率: {ret:+.1f}% ({pnl:,.0f}) | 胜率: {wr:.1f}% | 交易: {tr}笔')
print(f'  盈利{w}笔 亏损{l}笔 | 均盈{aw:,.0f} 均亏{al:,.0f} | 盈亏比{pf}')
print()
for code in STOCK_CODES:
    ct=[t for t in r['tlist'] if t['sym']==code]
    if ct:
        cw=[t for t in ct if t['pnl']>0]
        n=STOCK_POOL[code]['name'];stp=sum(t['pnl'] for t in ct)
        print(f'  {code} {n}: {len(ct)}笔 WR={len(cw)/len(ct)*100:.0f}% PnL={stp:+,.0f}')
