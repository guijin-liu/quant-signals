"""
v5 — 波动率自适应 + 机会性买点 + 多票并行
核心改动:
  1. 止损=每只票的ATR×倍数，不固定%
  2. 高波动票(锡业/神火)用宽止损抓大波
  3. 低波动票(雅化/亚钾)用紧止损提效率
  4. 做T用日内波动率触发
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

# 逐票独立参数(基于5年历史波动率)
CONFIG={
    '000933':{'name':'神火股份','vol':3.01,'sl_atr':1.8,'tp_atr':3.0,'bt':0.48,'rsi_lo':30,'rsi_hi':75},
    '002497':{'name':'雅化集团','vol':2.95,'sl_atr':1.5,'tp_atr':2.5,'bt':0.48,'rsi_lo':30,'rsi_hi':75},
    '000960':{'name':'锡业股份','vol':3.02,'sl_atr':1.8,'tp_atr':3.5,'bt':0.50,'rsi_lo':30,'rsi_hi':75},
    '000893':{'name':'亚钾国际','vol':2.91,'sl_atr':1.5,'tp_atr':2.5,'bt':0.48,'rsi_lo':30,'rsi_hi':75},
}

BET_PCT=0.20       # 每笔仓位
MAX_STOCKS=4        # 最多同时持股
MAX_EXPOSURE=0.80   # 总仓位上限
T_PCT=0.10          # 做T仓位

print('='*60)
print('  v5 波动率自适应策略')
print('='*60)
for c,cfg in CONFIG.items():
    n2=cfg['name']; v2=cfg['vol']; sa=cfg['sl_atr']; ta=cfg['tp_atr']
    print(f'  {c} {n2}: vol={v2:.1f}% sl={sa}xATR tp={ta}xATR')
print()

daily_data=fetch_all_daily_klines(5)

# ==================== 回测引擎 ====================
class Engine:
    def __init__(self,cap=INIT_CAP):
        self.cap=cap;self.init=cap;self.pos={};self.trd=[]
    def reset(self):
        self.cap=self.init;self.pos={};self.trd=[]
    def _cost(self,p,s,is_sell):
        a=p*s;c=max(a*0.0003,5);st=a*0.001 if is_sell else 0;tr=a*0.00002;sl=a*0.001
        return c+st+tr+sl
    def _open(self,sym,p,d,atr):
        if sym in self.pos:return
        exp=sum(pp['ep']*pp['sh'] for sl in self.pos.values() for pp in sl)/self.init
        if exp>=MAX_EXPOSURE or len(self.pos)>=MAX_STOCKS:return
        shares=int(self.init*BET_PCT/p/100)*100
        if shares<100:return
        cost=shares*p+self._cost(p,shares,False)
        if cost>self.cap:return
        self.cap-=cost
        self.pos[sym]=[{'ep':p,'sh':shares,'d':d,'atr':atr,'cost':cost,'high':p}]
    def _close(self,sym,p,d,reason):
        if sym not in self.pos:return
        for pp in self.pos[sym]:
            rev=p*pp['sh'];c=self._cost(p,pp['sh'],True)
            self.cap+=rev-c;pnl=rev-c-pp['cost'];ppnl=pnl/pp['cost']*100
            self.trd.append({
                'symbol':sym,'name':STOCK_POOL.get(sym,{}).get('name',''),
                'ep':round(pp['ep'],3),'xp':round(p,3),'pnl':round(pnl,2),
                'pnl_pct':round(ppnl,2),'reason':reason,'entry':str(pp['d'])
            })
        del self.pos[sym]
    def _should_exit(self,sym,p,cfg):
        if sym not in self.pos:return (False,'')
        ex=False;re=''
        for pp in self.pos[sym]:
            pp['high']=max(pp['high'],p)
            atr=pp.get('atr',p*0.02) or p*0.02
            pnl_pct=(p-pp['ep'])/pp['ep']
            # 止损: 价格跌破入场价-ATR倍
            if p<=pp['ep']-cfg['sl_atr']*atr:
                ex=True;re=f'sl({pnl_pct*100:.1f}%)';break
            # 止盈: 价格涨超ATR倍后回撤1ATR就出
            if pnl_pct>=cfg['tp_atr']*atr/pp['ep']:
                ex=True;re=f'tp({pnl_pct*100:.1f}%)';break
        return (ex,re)
    def run(self,sdf):
        self.reset()
        df=sdf.sort_values(['datetime','symbol']).reset_index(drop=True)
        df['date']=df['datetime'].dt.date
        for i,row in df.iterrows():
            sym=row['symbol'];sig=row['signal'];p=row['close'];d=row['date']
            atr=row.get('atr',p*0.02) or p*0.02
            cfg=CONFIG.get(sym,{})
            if sym in self.pos:
                exit,reason=self._should_exit(sym,p,cfg)
                if exit:
                    self._close(sym,p,d,reason)
                    continue
            if sig=='BUY':
                self._open(sym,p,d,atr)
        for sym in list(self.pos.keys()):
            last=df[df['symbol']==sym].iloc[-1]
            self._close(sym,last['close'],last['date'],'close')
        return self._results()
    def _results(self):
        tr=self.trd
        if not tr:return {'total_return':0,'win_rate':0,'trades':0}
        w=[t for t in tr if t['pnl']>0]
        l=[t for t in tr if t['pnl']<=0]
        total_pnl=sum(t['pnl'] for t in tr)
        wr=len(w)/len(tr)*100
        pf=sum(t['pnl'] for t in w)/max(abs(sum(t['pnl'] for t in l)),0.01) if l else 9
        return {
            'ret':round(total_pnl/self.init*100,2),
            'pnl':round(total_pnl,2),
            'wr':round(wr,2),
            'trades':len(tr),
            'wins':len(w),'losses':len(l),
            'avg_win':round(np.mean([t['pnl'] for t in w]),2) if w else 0,
            'avg_loss':round(np.mean([abs(t['pnl']) for t in l]),2) if l else 0,
            'pf':round(pf,2),
            'trades_list':tr
        }

# ==================== 信号生成 ====================
all_sigs=[]
for code in STOCK_CODES:
    cfg=CONFIG[code]
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
        if 'rsi' in df.columns and cfg['rsi_lo']<df['rsi'].iloc[i]<cfg['rsi_hi']:tb+=1
        if 'volume_surge' in df.columns and df['volume_surge'].iloc[i]:tb+=1
        if 'boll_pct_b' in df.columns and 0.2<df['boll_pct_b'].iloc[i]<0.9:tb+=1
        ts=min(tb/9.0,1.0)
        trb=0
        if 'pct_change_5' in df.columns and df['pct_change_5'].iloc[i]>0:trb+=1
        if 'ma_20' in df.columns and df['close'].iloc[i]>df['ma_20'].iloc[i]:trb+=1
        trs=min(trb/2.0,1.0)
        comp=0.40*ts+0.30*trs+0.30*0.5
        if comp>cfg['bt'] and ts>0.35:
            df.at[df.index[i],'signal']='BUY';buys+=1
        elif comp<0.18:df.at[df.index[i],'signal']='SELL'
    n3=cfg['name']
    print(f'  {code} {n3}: BUY={buys}')
    all_sigs.append(df)

sdf=pd.concat(all_sigs,ignore_index=True);sdf.sort_values(['datetime','symbol'],inplace=True)
btotal=(sdf['signal']=='BUY').sum()
print(f'\n总信号: BUY={btotal}')

eng=Engine(INIT_CAP)
r=eng.run(sdf)

print()
print('='*60)
print('  回测结果')
print('='*60)
ret=r['ret'];wr=r['wr'];tr=r['trades'];w=r['wins'];l=r['losses'];aw=r['avg_win'];al=r['avg_loss'];pf=r['pf']
print(f'  收益率: {ret:+.1f}%  胜率: {wr:.1f}%  交易: {tr}笔')
print(f'  盈利: {w}笔  亏损: {l}笔')
print(f'  平均盈利: {aw:,.0f}  平均亏损: {al:,.0f}')
print(f'  盈亏比: {pf:.2f}')

print(f'\n按股票:')
for code in STOCK_CODES:
    ct=[t for t in r['trades_list'] if t['symbol']==code]
    if ct:
        cw=[t for t in ct if t['pnl']>0]
        names=STOCK_POOL[code]['name']
        tp=sum(t['pnl'] for t in ct)
        wr2=len(cw)/len(ct)*100
        print(f'  {code} {names}: {len(ct)}笔 胜率{wr2:.0f}% 累计{tp:+,.0f}')
