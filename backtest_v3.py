"""
v3.0 仓位管理 + 建仓逻辑 + 逐票独立策略

仓位管理:
  原始仓位: 总资金×position_pct (默认25%)
  金字塔加仓: 信号确认后回调-1%加第2仓，再跌-1%加第3仓
  每仓1/3，总上限单票30%

建仓条件(叠加):
  1. 信号出现(MACD金叉+MA多头+趋势匹配)
  2. MA5在MA10之上(短期确认)
  3. 当日不是开盘跳空(开盘价在MA5附近)
  4. RSI不超买(<70)

出场:
  - 固定止损-2%
  - 移动止盈: 8%触发，回撤1/3就出
  - ATR追踪止损在盈利>3%后启用
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

# ============================================================
# 仓位管理参数
# ============================================================
POSITION_CONFIG = {
    'base_pct': 0.20,        # 首次建仓比例 (总资金%)
    'pyramid_pct': 0.10,     # 金字塔加仓比例
    'max_per_stock_pct': 0.35,  # 单票总仓位上限
    'max_total_positions': 4,   # 最多同时持4只
    'atr_multiplier': 1.5,      # ATR止损倍数
    'trailing_activate': 0.03,  # 盈利>3%启动移动止盈
    'trailing_retrace': 0.33,   # 回撤33%的利润就出
}

# ============================================================
# 逐票建仓条件
# ============================================================
PER_STOCK_ENTRY = {
    '000933': {
        'name': '神火股份',
        'buy_threshold': 0.58,
        'require_macd_golden': True,
        'require_ma_bullish': True,
        'require_trend': 'sideways',  # 震荡市MACD金叉最有效(80%胜率)
        'rsi_min': 35, 'rsi_max': 65,
        'boll_pct_min': 0.2,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.08,
    },
    '002497': {
        'name': '雅化集团',
        'buy_threshold': 0.55,
        'require_macd_golden': True,
        'require_ma_bullish': True,
        'require_trend': 'bull',  # MACD金叉+MA多头+牛市=100%
        'rsi_min': 35, 'rsi_max': 65,
        'boll_pct_min': 0.25,
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.10,
    },
    '000960': {
        'name': '锡业股份',
        'buy_threshold': 0.62,
        'require_macd_golden': True,
        'require_ma_bullish': True,
        'require_trend': 'bull',
        'rsi_min': 35, 'rsi_max': 60,
        'boll_pct_min': 0.25,
        'stop_loss_pct': 0.015,
        'take_profit_pct': 0.10,
    },
    '000893': {
        'name': '亚钾国际',
        'buy_threshold': 0.60,
        'require_macd_golden': True,
        'require_ma_bullish': True,
        'require_trend': 'bull',
        'rsi_min': 35, 'rsi_max': 62,
        'boll_pct_min': 0.25,
        'stop_loss_pct': 0.015,
        'take_profit_pct': 0.08,
    },
}

# ============================================================
# 回测引擎 v3 — 带仓位管理
# ============================================================
class BacktestEngineV3:
    def __init__(self, initial_capital=50000):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.positions = {}  # {symbol: {'entry_price':..., 'shares':..., 'entry_date':..., 'pyramid_lots':[...], 'high_since_entry':..., 'atr':...}}
        self.trades = []
        self.equity_curve = []
        self.pc = POSITION_CONFIG

    def reset(self):
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

    def _calc_cost(self, price, shares, is_sell):
        amount = price * shares
        commission = max(amount * 0.0003, 5)
        stamp = amount * 0.001 if is_sell else 0
        transfer = amount * 0.00002
        slippage = amount * 0.001
        return commission + stamp + transfer + slippage

    def _can_open_new(self, symbol, date):
        if len(self.positions) >= self.pc['max_total_positions']:
            return False
        # 同一天同一只票不重复开
        for pos_list in self.positions.values():
            for pos in pos_list:
                if pos['symbol'] == symbol and pos['entry_date'] == date:
                    return False
        return True

    def _get_stock_exposure(self, symbol):
        total = 0
        if symbol in self.positions:
            for pos in self.positions[symbol]:
                total += pos['entry_price'] * pos['shares']
        return total / self.initial_capital

    def _enter(self, symbol, price, date, atr, reason):
        """建仓/加仓"""
        exposure = self._get_stock_exposure(symbol)

        if exposure >= self.pc['max_per_stock_pct']:
            return

        # 首次建仓用base_pct，加仓用pyramid_pct
        if symbol not in self.positions or not self.positions[symbol]:
            bet_pct = self.pc['base_pct']
        else:
            bet_pct = self.pc['pyramid_pct']

        bet_amount = self.initial_capital * bet_pct
        shares = int(bet_amount / price / 100) * 100
        if shares < 100:
            return

        cost = price * shares + self._calc_cost(price, shares, False)
        if cost > self.capital:
            return

        self.capital -= cost
        pos = {
            'symbol': symbol,
            'entry_price': price,
            'shares': shares,
            'entry_date': date,
            'cost': cost,
            'high_since_entry': price,
            'low_since_entry': price,
            'atr': atr or price * 0.02,
            'reason': reason,
            'pyramid_level': 0 if symbol not in self.positions else len(self.positions[symbol]),
        }

        if symbol not in self.positions:
            self.positions[symbol] = []
        self.positions[symbol].append(pos)

    def _exit(self, symbol, price, date, reason):
        """全部平仓"""
        if symbol not in self.positions:
            return
        for pos in self.positions[symbol]:
            revenue = price * pos['shares']
            cost = self._calc_cost(price, pos['shares'], True)
            self.capital += revenue - cost
            pnl = revenue - cost - pos['cost']
            pnl_pct = pnl / pos['cost'] * 100
            self.trades.append({
                'symbol': symbol,
                'name': STOCK_POOL.get(symbol, {}).get('name', ''),
                'entry_date': str(pos['entry_date']),
                'exit_date': str(date),
                'entry_price': round(pos['entry_price'], 3),
                'exit_price': round(price, 3),
                'shares': pos['shares'],
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'exit_reason': reason,
                'pyramid_level': pos['pyramid_level'],
            })
        del self.positions[symbol]

    def run(self, signal_df):
        self.reset()
        df = signal_df.sort_values(['datetime', 'symbol']).reset_index(drop=True)
        df['date'] = df['datetime'].dt.date
        df['time_str'] = df['datetime'].dt.strftime('%H:%M')

        for i, row in df.iterrows():
            symbol = row['symbol']
            signal = row['signal']
            price = row['close']
            d = row['date']
            t = row['time_str']
            atr = row.get('atr', price * 0.02)

            # 更新持仓
            if symbol in self.positions:
                for pos in self.positions[symbol]:
                    pos['high_since_entry'] = max(pos['high_since_entry'], price)
                    pos['low_since_entry'] = min(pos['low_since_entry'], price)

                # 止损检查
                should_exit = False
                exit_reason = ''
                for pos in self.positions[symbol]:
                    pnl_pct = (price - pos['entry_price']) / pos['entry_price']
                    if pnl_pct <= -PER_STOCK_ENTRY[symbol]['stop_loss_pct']:
                        should_exit = True
                        exit_reason = f'止损({pnl_pct*100:.1f}%)'
                        break
                    if pnl_pct >= PER_STOCK_ENTRY[symbol]['take_profit_pct']:
                        should_exit = True
                        exit_reason = f'止盈({pnl_pct*100:.1f}%)'
                        break
                    # ATR追踪止盈
                    if pnl_pct > self.pc['trailing_activate']:
                        if atr > 0:
                            trail_stop = pos['high_since_entry'] - self.pc['atr_multiplier'] * atr
                            if price <= trail_stop:
                                should_exit = True
                                exit_reason = f'ATR追踪({pnl_pct*100:.1f}%)'
                                break

                if should_exit:
                    self._exit(symbol, price, d, exit_reason)
                    self._record_equity(i, d, t)
                    continue

            # 买入信号
            if signal == 'BUY' and self._can_open_new(symbol, d):
                self._enter(symbol, price, d, atr, '信号入场')

            self._record_equity(i, d, t)

        # 收盘平仓
        for symbol in list(self.positions.keys()):
            last = df[df['symbol'] == symbol].iloc[-1]
            self._exit(symbol, last['close'], last['date'], '收盘平仓')

        self._record_equity(len(df)-1, df['date'].iloc[-1], df['time_str'].iloc[-1])
        return self._get_results()

    def _record_equity(self, bar, date, time):
        pos_val = sum(p['entry_price'] * p['shares'] for pl in self.positions.values() for p in pl)
        self.equity_curve.append({
            'bar': bar, 'date': date, 'time': time,
            'cash': round(self.capital, 2),
            'position_value': round(pos_val, 2),
            'total_equity': round(self.capital + pos_val, 2),
        })

    def _get_results(self):
        trades = self.trades
        if not trades:
            return {'total_return': 0, 'win_rate': 0, 'total_trades': 0, 'sharpe': 0, 'max_drawdown': 0, 'profit_factor': 0, 'trades': []}

        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        wr = len(wins) / len(trades) * 100
        total_pnl = sum(t['pnl'] for t in trades)
        total_ret = total_pnl / self.initial_capital * 100
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t['pnl']) for t in losses]) if losses else 0
        pf = sum(t['pnl'] for t in wins) / max(abs(sum(t['pnl'] for t in losses)), 0.01)

        eq = pd.DataFrame(self.equity_curve)
        if not eq.empty and 'total_equity' in eq.columns and len(eq) > 1:
            eq['peak'] = eq['total_equity'].cummax()
            eq['dd'] = (eq['total_equity'] - eq['peak']) / eq['peak']
            max_dd = eq['dd'].min() * 100
            eq['daily_ret'] = eq['total_equity'].pct_change()
            std = eq['daily_ret'].std()
            sharpe = eq['daily_ret'].mean() / std * np.sqrt(252) if std and std > 0 else 0
        else:
            max_dd = 0; sharpe = 0

        return {
            'total_return': round(total_ret, 2), 'total_pnl': round(total_pnl, 2),
            'win_rate': round(wr, 2), 'total_trades': len(trades),
            'wins': len(wins), 'losses': len(losses),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'profit_factor': round(pf, 2), 'sharpe': round(sharpe, 2),
            'max_drawdown': round(max_dd, 2), 'trades': trades, 'equity_curve': eq,
        }

    def print_summary(self):
        r = self._get_results()
        print(f'{"="*50}')
        print(f'回测结果摘要')
        print(f'{"="*50}')
        print(f'初始资金: RMB {self.initial_capital:,.0f}')
        print(f'最终权益: RMB {self.capital:,.0f}')
        print(f'总收益率: {r["total_return"]:.2f}%')
        print(f'总盈亏: RMB {r["total_pnl"]:,.2f}')
        print(f'交易次数: {r["total_trades"]}')
        print(f'胜率: {r["win_rate"]:.1f}%')
        print(f'平均盈利: RMB {r["avg_win"]:,.2f}')
        print(f'平均亏损: RMB {r["avg_loss"]:,.2f}')
        print(f'盈亏比: {r["profit_factor"]:.2f}')
        print(f'夏普比率: {r["sharpe"]:.2f}')
        print(f'最大回撤: {r["max_drawdown"]:.2f}%')
        print(f'{"="*50}')


# ============================================================
# 主流程
# ============================================================
print('='*70)
print('  v3.0 仓位管理 + 逐票独立策略')
print('='*70)

print('\n仓位规则:')
print(f'  首次建仓: {POSITION_CONFIG["base_pct"]*100:.0f}%资金')
print(f'  金字塔加仓: +{POSITION_CONFIG["pyramid_pct"]*100:.0f}% (每级)')
print(f'  单票上限: {POSITION_CONFIG["max_per_stock_pct"]*100:.0f}%')
print(f'  同时持仓上限: {POSITION_CONFIG["max_total_positions"]}只')
print(f'  ATR追踪止损: {POSITION_CONFIG["atr_multiplier"]}x ATR (盈利>{POSITION_CONFIG["trailing_activate"]*100:.0f}%后)')

daily_data = fetch_all_daily_klines(5)

all_signals = []
for code in STOCK_CODES:
    cfg = PER_STOCK_ENTRY[code]
    name = STOCK_POOL[code]['name']
    df = daily_data.get(code)
    if df is None or df.empty:
        continue

    df = compute_all_technical_features(df, 'daily')

    # 趋势
    df['ma20'] = df['close'].rolling(20).mean()
    df['daily_trend'] = 'sideways'
    df.loc[df['close'] > df['ma20'] * 1.05, 'daily_trend'] = 'bull'
    df.loc[df['close'] < df['ma20'] * 0.95, 'daily_trend'] = 'bear'

    # 信号
    df['signal'] = 'HOLD'
    buys = 0
    for i in range(max(60, len(df)//10), len(df)):
        # 打分
        tech = 0
        if all(f'ma_{p}' in df.columns for p in [5,10,20]):
            if df['ma_5'].iloc[i]>df['ma_10'].iloc[i]>df['ma_20'].iloc[i]: tech+=2
            elif df['ma_5'].iloc[i]>df['ma_10'].iloc[i]: tech+=1
        if 'ma_60' in df.columns and df['close'].iloc[i]>df['ma_60'].iloc[i]: tech+=1
        if 'macd_golden_cross' in df.columns and df['macd_golden_cross'].iloc[i]: tech+=1
        if 'macd_hist_sign_change' in df.columns and df['macd_hist_sign_change'].iloc[i]: tech+=1
        if 'rsi' in df.columns and cfg['rsi_min']<df['rsi'].iloc[i]<cfg['rsi_max']: tech+=1
        if 'volume_surge' in df.columns and df['volume_surge'].iloc[i]: tech+=1
        if 'boll_pct_b' in df.columns and df['boll_pct_b'].iloc[i]>cfg['boll_pct_min']: tech+=1
        ts = min(tech/8.0, 1.0)

        trend = 0
        if 'pct_change_5' in df.columns and df['pct_change_5'].iloc[i]>0: trend+=1
        if 'ma_20' in df.columns and df['close'].iloc[i]>df['ma_20'].iloc[i]: trend+=1
        if 'pct_change_3' in df.columns and df['pct_change_3'].iloc[i]>0: trend+=1
        trs = min(trend/3.0, 1.0)

        composite = 0.35*ts + 0.20*trs + 0.45*0.5

        # 条件
        ok = True
        if cfg['require_macd_golden'] and 'macd_golden_cross' in df.columns:
            ok &= bool(df['macd_golden_cross'].iloc[i])
        if cfg['require_ma_bullish'] and 'ma_bullish' in df.columns:
            ok &= bool(df['ma_bullish'].iloc[i])
        if cfg['require_trend'] and 'daily_trend' in df.columns:
            ok &= (df['daily_trend'].iloc[i] == cfg['require_trend'])

        if composite > cfg['buy_threshold'] and ok:
            df.at[df.index[i], 'signal'] = 'BUY'
            buys += 1
        elif composite < 0.20:
            df.at[df.index[i], 'signal'] = 'SELL'

    print(f'  {code} {name}: BUY={buys} 次')
    all_signals.append(df)

sdf = pd.concat(all_signals, ignore_index=True)
sdf.sort_values(['datetime', 'symbol'], inplace=True)

print(f'\n运行回测...')
engine = BacktestEngineV3(initial_capital=50000)
results = engine.run(sdf)

print()
engine.print_summary()

# 逐票
trades = results['trades']
print(f'\n按股票:')
for code in STOCK_CODES:
    ct = [t for t in trades if t['symbol']==code]
    if ct:
        cw = [t for t in ct if t['pnl']>0]
        names = STOCK_POOL[code]['name']
        print(f'  {code} {names}: {len(ct)}笔 胜率{len(cw)/len(ct)*100:.0f}% 累计{sum(t["pnl"] for t in ct):+,.0f}')

# 显示仓位使用情况
pyramid = [t for t in trades if t.get('pyramid_level', 0) > 0]
if pyramid:
    print(f'\n金字塔加仓: {len(pyramid)}笔')
    for lv in sorted(set(t['pyramid_level'] for t in trades)):
        lt = [t for t in trades if t['pyramid_level']==lv]
        lw = [t for t in lt if t['pnl']>0]
        print(f'  级别{lv}: {len(lt)}笔 胜率{len(lw)/len(lt)*100:.0f}%')
