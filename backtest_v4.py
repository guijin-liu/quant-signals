"""
v4 — 机会性买点 + 多票同时 + 做T
核心理念: 信号来了就买，不追求高胜率，靠盈亏比和仓位管理赚钱
"""
import sys,io,json,logging,warnings
from datetime import datetime
from pathlib import Path
import pandas as pd, numpy as np

sys.path.insert(0,str(Path(__file__).parent))
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

from data.baostock_fetcher import fetch_all_daily_klines,fetch_all_minute_klines
from features.technical import compute_all_technical_features
from config import STOCK_CODES,STOCK_POOL

PC={
    'base_bet_pct':0.20,     # 每笔建仓20%
    't_pct':0.10,            # 做T仓位10%
    'max_stocks':4,          # 最多同时4只
    'total_max':0.80,        # 总仓位上限80%
    'stop_loss':0.05,        # 硬止损-5% (放宽, A股日波动2-3%)
    'take_profit':0.12,      # 止盈+12% (大波段)
    'trail_atr':1.5,         # ATR追踪
    'trail_active':0.04,     # 盈利>4%启动追踪
    't_trigger':0.015,       # 日内波动>1.5%触发做T
}

print('='*60)
print('  v4 — 机会性买点 + 多票 + 做T')
print('='*60)
base_pct=PC['base_bet_pct']*100; t_pct=PC['t_pct']*100; ms=PC['max_stocks']; tm=PC['total_max']*100
print(f'\n仓位: 每笔{base_pct:.0f}%建仓 + {t_pct:.0f}%做T, 最多{ms}只, 总仓{tm:.0f}%')
sl_pct = PC['stop_loss']*100; tp_pct = PC['take_profit']*100; ta = PC['trail_atr']
print(f'风控: 止损{sl_pct:.0f}% 止盈{tp_pct:.0f}% ATR追踪{ta}x')
print()

daily_data=fetch_all_daily_klines(5)
min5_data=fetch_all_minute_klines('5',60)

class EngineV4:
    def __init__(self,cap=50000):
        self.cap=cap;self.init=cap;self.pos={};self.trd=[];self.eq=[]
    def reset(self):
        self.cap=self.init;self.pos={};self.trd=[];self.eq=[]
    def _cost(self,p,s,is_sell):
        a=p*s;c=max(a*0.0003,5);st=a*0.001 if is_sell else 0;tr=a*0.00002;sl=a*0.001
        return c+st+tr+sl
    def _open(self,sym,p,d,t,atr,reason):
        if sym in self.pos:return
        exp=sum(pos['entry_price']*pos['shares'] for slist in self.pos.values() for pos in slist)/self.init
        if exp>=PC['total_max']:return
        if len(self.pos)>=PC['max_stocks']:return
        shares=int(self.init*PC['base_bet_pct']/p/100)*100
        if shares<100:return
        cost=shares*p+self._cost(p,shares,False)
        if cost>self.cap:return
        self.cap-=cost
        self.pos[sym]=[{'ep':p,'shares':shares,'d':d,'t':t,'cost':cost,'high':p,'low':p,'atr':atr or p*0.02,'reason':reason,'tot_shares':0}]
    def _close(self,sym,p,d,reason):
        if sym not in self.pos:return
        for pos in self.pos[sym]:
            rev=p*pos['shares'];cost=self._cost(p,pos['shares'],True)
            self.cap+=rev-cost;pnl=rev-cost-pos['cost'];pp=pnl/pos['cost']*100
            self.trd.append({'symbol':sym,'name':STOCK_POOL.get(sym,{}).get('name',''),'entry_date':str(pos['d']),'exit_date':str(d),'entry_price':round(pos['ep'],3),'exit_price':round(p,3),'shares':pos['shares'],'pnl':round(pnl,2),'pnl_pct':round(pp,2),'exit_reason':reason,'t_shares':pos['tot_shares']})
        del self.pos[sym]
    def _do_t(self,sym,p,d,hi,lo):
        """做T: 日内拉升>1.5%卖10%仓位，回落买回"""
        if sym not in self.pos:return
        for pos in self.pos[sym]:
            day_pct=(hi-lo)/lo
            if day_pct<PC['t_trigger']:continue
            t_shares=int(pos['shares']*PC['t_pct']/100)*100
            if t_shares<100:continue
            sell_p=hi
            sell_rev=sell_p*t_shares-self._cost(sell_p,t_shares,True)
            buy_p=lo
            buy_cost=buy_p*t_shares+self._cost(buy_p,t_shares,False)
            t_pnl=sell_rev-buy_cost
            if t_pnl>0:
                self.cap+=t_pnl
                pos['tot_shares']+=t_shares
    def _record(self,i,d,t):
        pv=sum(pos['ep']*pos['shares'] for slist in self.pos.values() for pos in slist)
        self.eq.append({'bar':i,'date':d,'time':t,'cash':round(self.cap,2),'pv':round(pv,2),'eq':round(self.cap+pv,2)})
    def run(self,sdf):
        self.reset()
        df=sdf.sort_values(['datetime','symbol']).reset_index(drop=True)
        df['date']=df['datetime'].dt.date;df['time_str']=df['datetime'].dt.strftime('%H:%M')
        for i,row in df.iterrows():
            sym=row['symbol'];sig=row['signal'];p=row['close'];d=row['date'];ti=row['time_str'];atr=row.get('atr',p*0.02);hi=row.get('high',p);lo=row.get('low',p)
            if sym in self.pos:
                for pos in self.pos[sym]:
                    pos['high']=max(pos['high'],p);pos['low']=min(pos['low'],p)
                exit=False;reason=''
                for pos in self.pos[sym]:
                    pp=(p-pos['ep'])/pos['ep']
                    if pp<=-PC['stop_loss']:exit=True;reason=f'止损({pp*100:.1f}%)';break
                    if pp>=PC['take_profit']:exit=True;reason=f'止盈({pp*100:.1f}%)';break
                    if pp>PC['trail_active'] and atr>0:
                        ts=pos['high']-PC['trail_atr']*atr
                        if p<=ts:exit=True;reason=f'ATR追踪({pp*100:.1f}%)';break
                if not exit:self._do_t(sym,p,d,hi,lo)
                if exit:self._close(sym,p,d,reason);self._record(i,d,ti);continue
            if sig=='BUY':
                self._open(sym,p,d,ti,atr,'信号入场')
            self._record(i,d,ti)
        for sym in list(self.pos.keys()):
            last=df[df['symbol']==sym].iloc[-1]
            self._close(sym,last['close'],last['date'],'收盘平仓')
        self._record(len(df)-1,df['date'].iloc[-1],df['time_str'].iloc[-1])
        return self._results()
    def _results(self):
        tr=self.trd
        if not tr:return {'total_return':0,'win_rate':0,'total_trades':0}
        w=[t for t in tr if t['pnl']>0];l=[t for t in tr if t['pnl']<=0]
        wr=len(w)/len(tr)*100;tp=sum(t['pnl'] for t in tr);trr=tp/self.init*100
        pf=sum(t['pnl'] for t in w)/max(abs(sum(t['pnl'] for t in l)),0.01)
        eq=pd.DataFrame(self.eq)
        if not eq.empty and 'eq' in eq.columns and len(eq)>1:
            eq['pk']=eq['eq'].cummax();eq['dd']=(eq['eq']-eq['pk'])/eq['pk'];mdd=eq['dd'].min()*100
        else:mdd=0
        return {'total_return':round(trr,2),'total_pnl':round(tp,2),'win_rate':round(wr,2),'total_trades':len(tr),'wins':len(w),'losses':len(l),'avg_win':round(np.mean([t['pnl'] for t in w]),2) if w else 0,'avg_loss':round(np.mean([abs(t['pnl']) for t in l]),2) if l else 0,'profit_factor':round(pf,2),'max_drawdown':round(mdd,2),'trades':tr}
    def print(self):
        r=self._results()
        print(f'总收益:{r[\"total_return\"]:.1f}% 胜率:{r[\"win_rate\"]:.1f}% 交易:{r[\"total_trades\"]}笔')
        print(f'盈亏比:{r[\"profit_factor\"]:.2f} 最大回撤:{r[\"max_drawdown\"]:.1f}%')
        print(f'均盈:{r[\"avg_win\"]:,.0f} 均亏:{r[\"avg_loss\"]:,.0f}')

# ============================================================
# 机会性信号: 降低门槛
# ============================================================
print('信号规则: MACD金叉或MA排列改善+RSI不超买+放量确认 = BUY')
print()

SIGNAL_RULES={
    '000933':{'bt':0.50,'trend':None,'rsi_lo':30,'rsi_hi':70},
    '002497':{'bt':0.48,'trend':None,'rsi_lo':30,'rsi_hi':70},
    '000960':{'bt':0.50,'trend':None,'rsi_lo':30,'rsi_hi':70},
    '000893':{'bt':0.50,'trend':None,'rsi_lo':30,'rsi_hi':70},
}

all_sigs=[]
for code in STOCK_CODES:
    s=SIGNAL_RULES[code]
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
        if 'rsi' in df.columns and s['rsi_lo']<df['rsi'].iloc[i]<s['rsi_hi']:tb+=1
        if 'volume_surge' in df.columns and df['volume_surge'].iloc[i]:tb+=1
        if 'boll_pct_b' in df.columns and 0.2<df['boll_pct_b'].iloc[i]<0.9:tb+=1
        ts=min(tb/9.0,1.0)
        trb=0
        if 'pct_change_5' in df.columns and df['pct_change_5'].iloc[i]>0:trb+=1
        if 'ma_20' in df.columns and df['close'].iloc[i]>df['ma_20'].iloc[i]:trb+=1
        trs=min(trb/2.0,1.0)
        comp=0.40*ts+0.30*trs+0.30*0.5
        if comp>s['bt'] and ts>0.35:
            df.at[df.index[i],'signal']='BUY';buys+=1
        elif comp<0.18:df.at[df.index[i],'signal']='SELL'
    print(f'  {code} {STOCK_POOL[code][\"name\"]}: BUY={buys} 次')
    all_sigs.append(df)

sdf=pd.concat(all_sigs,ignore_index=True);sdf.sort_values(['datetime','symbol'],inplace=True)
btotal=(sdf['signal']=='BUY').sum()
stotal=(sdf['signal']=='SELL').sum()
print(f'\n总信号: {len(sdf)}行 BUY={btotal} SELL={stotal}')
print(f'(比之前的29个买入信号多了{btotal//29 - 1}倍)')

print(f'\n运行回测...')
e=EngineV4(50000)
r=e.run(sdf)
print()
e.print()

tr=r['trades']
print(f'\n按股票:')
for code in STOCK_CODES:
    ct=[t for t in tr if t['symbol']==code]
    if ct:
        cw=[t for t in ct if t['pnl']>0]
        names=STOCK_POOL[code]['name']
        print(f'  {code} {names}: {len(ct)}笔 胜率{len(cw)/len(ct)*100:.0f}% 累计{sum(t[\"pnl\"] for t in ct):+,.0f}')

t_pnl=sum(t['pnl'] for t in tr if t.get('t_shares',0)>0)
print(f'\n做T贡献: {t_pnl:+,.0f}')
