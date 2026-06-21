"""
v7 — 逐票独立信号 + 机会性买点 + 10万本金
基于历史概率分析:
  000933 神火: MACD金叉+震荡 → 80%胜率(5次/5年) — 稀有高胜率
  002497 雅化: MA多头+MACD金叉 → 87.5%胜率(8次/5年) — 稀有
  000960 锡业: RSI<30抄底 → 91.7%胜率(12次/5年) — 稀有反弹
  000893 亚钾: MA多头+RSI30-65 → 73%胜率(248次/5年) — 主力信号!
"""
import sys,io,logging,warnings
from pathlib import Path
import pandas as pd,numpy as np

sys.path.insert(0,str(Path(__file__).parent))
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

from data.baostock_fetcher import fetch_all_daily_klines
from features.technical import compute_all_technical_features
from config import STOCK_CODES,STOCK_POOL

INIT=100000; BET=0.20; MAX_S=4; MAX_EXP=0.80; SL=0.05; TP=0.15; TRAIL=0.04

# 每票独立信号规则
SIGNALS={
    '000933':('神火','MACD金叉',lambda row,df:row.get('macd_golden_cross',0)==1),
    '002497':('雅化','MA多+金叉',lambda row,df:row.get('ma_bullish',0)==1 and row.get('macd_golden_cross',0)==1),
    '000960':('锡业','RSI<30',lambda row,df:row.get('rsi',99)<30),
    '000893':('亚钾','MA多+RSI30-65',lambda row,df:row.get('ma_bullish',0)==1 and 30<row.get('rsi',99)<65),
}

sep='='*60
print(sep)
print('  v7 — 逐票独立概率策略 + 10万')
print(sep)
for c,(n,r,_) in SIGNALS.items():print('  {} {}: {}'.format(c,n,r))
print('  本金:{}万 | 建仓:{}% | 止损:{}% | 止盈:{}%'.format(INIT//10000,int(BET*100),int(SL*100),int(TP*100)))
print()

daily=fetch_all_daily_klines(5)

class Engine:
    def __init__(self,cap):self.cap=cap;self.init=cap;self.pos={};self.trd=[]
    def _cost(self,p,s,is_sell):
        a=p*s;c=max(a*0.0003,5);st=a*0.001 if is_sell else 0;t=a*0.00002;sl2=a*0.001
        return c+st+t+sl2
    def _open(self,sym,p,d):
        if sym in self.pos:return
        exp=sum(pp['ep']*pp['sh'] for sl in self.pos.values() for pp in sl)/self.init
        if exp>=MAX_EXP or len(self.pos)>=MAX_S:return
        sh=int(self.init*BET/p/100)*100
        if sh<100:return
        cost=sh*p+self._cost(p,sh,False)
        if cost>self.cap:return
        self.cap-=cost
        self.pos[sym]=[{'ep':p,'sh':sh,'d':d,'cost':cost,'hi':p}]
    def _close(self,sym,p,d,reason):
        if sym not in self.pos:return
        for pp in self.pos[sym]:
            rev=p*pp['sh'];c=self._cost(p,pp['sh'],True)
            self.cap+=rev-c;pnl=rev-c-pp['cost'];pc=pnl/pp['cost']*100
            self.trd.append({'s':sym,'n':STOCK_POOL.get(sym,{}).get('name',''),'ep':round(pp['ep'],3),'xp':round(p,3),'pnl':round(pnl,2),'pp':round(pc,2),'r':reason,'ed':str(pp['d'])})
        del self.pos[sym]
    def run(self,sdf):
        self.pos={};self.trd=[]
        df=sdf.sort_values(['datetime','symbol']).reset_index(drop=True)
        df['date']=df['datetime'].dt.date
        for _,row in df.iterrows():
            sym=row['symbol'];sig=row['signal'];p=row['close'];d=row['date']
            if sym in self.pos:
                out=False;rs=''
                for pp in self.pos[sym]:
                    pp['hi']=max(pp['hi'],p)
                    pc2=(p-pp['ep'])/pp['ep']
                    if pc2<=-SL:out=True;rs='sl({:.1f}%)'.format(pc2*100);break
                    if pc2>=TP:out=True;rs='tp({:.1f}%)'.format(pc2*100);break
                    if pc2>=TRAIL:
                        retrace=(pp['hi']-p)/pp['hi']
                        if retrace>=0.02*(pc2/TRAIL):out=True;rs='trail({:.1f}%)'.format(pc2*100);break
                if out:self._close(sym,p,d,rs);continue
            if sig=='BUY':self._open(sym,p,d)
        for sym in list(self.pos.keys()):
            last=df[df['symbol']==sym].iloc[-1]
            self._close(sym,last['close'],last['date'],'close')
        return self._res()
    def _res(self):
        tr=self.trd
        if not tr:return {'ret':0,'wr':0,'trades':0}
        w=[t for t in tr if t['pnl']>0];l=[t for t in tr if t['pnl']<=0]
        tp2=sum(t['pnl'] for t in tr);wr2=len(w)/len(tr)*100
        pf=sum(t['pnl'] for t in w)/max(abs(sum(t['pnl'] for t in l)),0.01) if l else 99
        return {'ret':round(tp2/self.init*100,2),'pnl':round(tp2,2),'wr':round(wr2,2),'trd':len(tr),'w':len(w),'l':len(l),'aw':round(np.mean([t['pnl'] for t in w]),2) if w else 0,'al':round(np.mean([abs(t['pnl']) for t in l]),2) if l else 0,'pf':round(pf,2),'tlist':tr}

# 给每只票按独立规则打信号
all_sigs=[]
for code in STOCK_CODES:
    name,desc,rule=SIGNALS[code]
    df=daily.get(code)
    if df is None or df.empty:continue
    df=compute_all_technical_features(df,'daily')
    df['signal']='HOLD';b=0
    for i in range(max(60,len(df)//10),len(df)):
        row=df.iloc[i]
        if rule(row,df):
            # 附加: 不追高, RSI不超买
            rsi=row.get('rsi',50)
            boll=row.get('boll_pct_b',0.5)
            if rsi<75 and boll<0.9 and row['close']>0:
                df.at[df.index[i],'signal']='BUY';b+=1
    print('  {} {}: {} → BUY={}次'.format(code,name,desc,b))
    all_sigs.append(df)

sdf=pd.concat(all_sigs,ignore_index=True);sdf.sort_values(['datetime','symbol'],inplace=True)
bt=(sdf['signal']=='BUY').sum()
print('\n总买入信号: {}次'.format(bt))

# 跑多组参数
print('\n'+sep)
print('  参数扫描')
print(sep)
best_ret=-999;best_p=None
for sl in [0.04,0.05,0.06]:
    for tp in [0.10,0.12,0.15,0.18]:
        SL=sl;TP=tp
        e=Engine(INIT);r=e.run(sdf)
        print('  SL={:.0f}% TP={:.0f}% -> Ret={:+.1f}% WR={:.1f}% T={:3d} PF={}'.format(sl*100,tp*100,r['ret'],r['wr'],r['trd'],r['pf']))
        if r['ret']>best_ret:best_ret=r['ret'];best_p={'sl':sl,'tp':tp,'r':r}

print('\n'+sep)
print('  最佳参数: SL={:.0f}% TP={:.0f}%'.format(best_p['sl']*100,best_p['tp']*100))
print(sep)
r=best_p['r']
print('  收益率: {:+5.1f}% ({:+,.0f}) | 胜率: {:.1f}% | 交易: {}笔'.format(r['ret'],r['pnl'],r['wr'],r['trd']))
print('  盈利:{} 亏损:{} | 均盈:{:+,.0f} 均亏:{:,.0f} | 盈亏比:{}'.format(r['w'],r['l'],r['aw'],r['al'],r['pf']))
print()
for code in STOCK_CODES:
    ct=[t for t in r['tlist'] if t['s']==code]
    if ct:
        cw=[t for t in ct if t['pnl']>0];n=STOCK_POOL[code]['name']
        print('  {} {}: {}笔 WR={:.0f}% PnL={:+,.0f}'.format(code,n,len(ct),len(cw)/len(ct)*100,sum(t['pnl'] for t in ct)))
