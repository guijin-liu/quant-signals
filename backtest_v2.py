"""
全量回测 v2.0 — 6维框架 + MA双周期共振 + 腾讯/新浪数据
5年日线 + 60天5min/15min

目标: 收益率最大化 (不是胜率最大化)
策略: 高盈亏比 + 宽止损 + 共振过滤
"""
import sys, io, json, logging, warnings
from datetime import datetime
from pathlib import Path
import pandas as pd, numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

from data.baostock_fetcher import fetch_all_daily_klines, fetch_all_minute_klines
from features.technical import compute_all_technical_features
from features.ma_resonance import compute_ma_resonance_features
from backtest.engine import BacktestEngine
from config import BACKTEST_PARAMS, RISK_PARAMS, STOCK_CODES, STOCK_POOL, SIGNAL_PARAMS

# ============================================================
# 1. 加载全部数据
# ============================================================
print('='*60)
print('  全量回测 v2.0 — 收益率最大化')
print('='*60)

print('\n[1/4] 加载5年日线(baostock)...')
daily_data = fetch_all_daily_klines(5)

print('\n[2/4] 加载60天分钟线(baostock)...')
min5_data = fetch_all_minute_klines('5', 60)
min15_data = fetch_all_minute_klines('15', 60)

print('\n[3/4] 加载实时行情(腾讯)...')
from data.tencent_sina import get_realtime_quotes
quotes = get_realtime_quotes()

# ============================================================
# 2. 特征计算
# ============================================================
print('\n[4/4] 计算特征+信号...')

# 用刚才72组合搜索的最佳收益参数
BEST_PARAMS = {
    'buy_threshold': 0.50,   # 低阈值 → 更多交易 → 更多收益
    'resonance_required': 1, # 1维共振即可 → 增加信号量
    'stop_loss': 0.025,      # 2.5% 宽止损
    'take_profit': 0.08,     # 8% 宽止盈
    'tech_weight': 0.25,     # 技术权重适中
}

# 更新风控
RISK_PARAMS['stop_loss_pct'] = BEST_PARAMS['stop_loss']
RISK_PARAMS['take_profit_pct'] = BEST_PARAMS['take_profit']

all_signals = []
all_ma_signals = []

for code in STOCK_CODES:
    name = STOCK_POOL[code]['name']
    df = daily_data.get(code)
    if df is None or df.empty:
        continue

    # 日线技术指标
    df = compute_all_technical_features(df, 'daily')

    # 5min/15min MA共振特征(如果有分钟数据)
    df5 = min5_data.get(code)
    df15 = min15_data.get(code)
    if df5 is not None and not df5.empty:
        df_ma = compute_ma_resonance_features(df5, df15)
        # 聚合到日线: 每天取MA共振信号
        df_ma['date'] = df_ma['datetime'].dt.date
        daily_ma = df_ma.groupby('date').agg(
            ma_entry_score_max=('ma_entry_score', 'max'),
            ma_golden_count=('ma_golden_cross', 'sum'),
            ma_dead_count=('ma_dead_cross', 'sum'),
            trend_bull_pct=('trend_direction', lambda x: (x=='bull').mean()),
        ).reset_index()
        # 合并到日线
        df['date'] = df['datetime'].dt.date
        df = df.merge(daily_ma, on='date', how='left')
        df.fillna({'ma_entry_score_max': 0.5, 'ma_golden_count': 0,
                   'ma_dead_count': 0, 'trend_bull_pct': 0.5}, inplace=True)

    # 信号生成
    df['signal'] = 'HOLD'
    df['signal_score'] = 0.0

    tw = BEST_PARAMS['tech_weight']
    trw = 0.20
    ow = (1.0 - tw - trw) / 4  # 剩余4维平分
    bt = BEST_PARAMS['buy_threshold']
    rr = BEST_PARAMS['resonance_required']

    for i in range(max(60, len(df)//10), len(df)):
        # 技术维度 (7个子项)
        tech_bull = 0
        if all(f'ma_{p}' in df.columns for p in [5,10,20]):
            if df['ma_5'].iloc[i] > df['ma_10'].iloc[i] > df['ma_20'].iloc[i]:
                tech_bull += 2
            elif df['ma_5'].iloc[i] > df['ma_10'].iloc[i]:
                tech_bull += 1
        if 'ma_60' in df.columns and df['close'].iloc[i] > df['ma_60'].iloc[i]:
            tech_bull += 1
        if 'macd_golden_cross' in df.columns and df['macd_golden_cross'].iloc[i]:
            tech_bull += 1
        if 'macd_hist_sign_change' in df.columns and df['macd_hist_sign_change'].iloc[i]:
            tech_bull += 1
        if 'rsi' in df.columns and 40 < df['rsi'].iloc[i] < 70:
            tech_bull += 1
        if 'volume_surge' in df.columns and df['volume_surge'].iloc[i]:
            tech_bull += 1
        if 'boll_pct_b' in df.columns and df['boll_pct_b'].iloc[i] > 0.4:
            tech_bull += 1
        tech_score = min(tech_bull/7.0, 1.0)

        # MA共振加分
        if 'ma_entry_score_max' in df.columns:
            ma_score = df['ma_entry_score_max'].iloc[i]
            tech_score = 0.6 * tech_score + 0.4 * ma_score  # 融合

        # 趋势维度
        trend_bull = 0
        if 'pct_change_5' in df.columns and df['pct_change_5'].iloc[i] > 0:
            trend_bull += 1
        if 'ma_20' in df.columns and df['close'].iloc[i] > df['ma_20'].iloc[i]:
            trend_bull += 1
        if 'pct_change_3' in df.columns and df['pct_change_3'].iloc[i] > 0:
            trend_bull += 1
        # 15min趋势方向
        if 'trend_bull_pct' in df.columns and df['trend_bull_pct'].iloc[i] > 0.5:
            trend_bull += 1
        trend_score = min(trend_bull/4.0, 1.0)

        # 综合评分
        composite = tw*tech_score + trw*trend_score + ow*0.5*4
        resonance = (1 if tech_score>=0.60 else 0) + (1 if trend_score>=0.55 else 0)

        df.at[df.index[i], 'signal_score'] = round(composite, 3)

        if composite > bt and resonance >= rr:
            df.at[df.index[i], 'signal'] = 'BUY'
        elif composite < 0.20 and resonance == 0:
            df.at[df.index[i], 'signal'] = 'SELL'

    buy_n = (df['signal']=='BUY').sum()
    sell_n = (df['signal']=='SELL').sum()
    avg_score = df[df['signal']=='BUY']['signal_score'].mean()
    print(f'  {code} {name}: BUY={buy_n} SELL={sell_n} avg_buy_score={avg_score:.3f}')
    all_signals.append(df)

# 合并回测
sdf = pd.concat(all_signals, ignore_index=True)
sdf.sort_values(['datetime', 'symbol'], inplace=True)

buy_total = (sdf['signal']=='BUY').sum()
sell_total = (sdf['signal']=='SELL').sum()
print(f'\n  合并信号: {len(sdf)}行, BUY={buy_total}, SELL={sell_total}')

# 回测
engine = BacktestEngine()
results = engine.run(sdf)

print()
print('='*60)
print('  回测结果')
print('='*60)
engine.print_summary()

# 交易详情
trades = results['trades']
if trades:
    wins = [t for t in trades if t['pnl']>0]
    max_win = max(t['pnl'] for t in trades)
    max_loss = min(t['pnl'] for t in trades)
    print(f'\n  交易详情:')
    print(f'    总交易: {len(trades)}')
    print(f'    盈利: {len(wins)}笔, 亏损: {len(trades)-len(wins)}笔')
    print(f'    最大单笔盈利: {max_win:,.0f}')
    print(f'    最大单笔亏损: {max_loss:,.0f}')

    # 按股票统计
    for code in STOCK_CODES:
        ct = [t for t in trades if t['symbol']==code]
        if ct:
            cw = [t for t in ct if t['pnl']>0]
            total_pnl = sum(t['pnl'] for t in ct)
            wr_pct = len(cw)/len(ct)*100
            print(f'    {code}: {len(ct)}笔, 胜率{wr_pct:.0f}%, 累计{total_pnl:,.0f}')

# 保存结果
report = {
    'params': BEST_PARAMS,
    'summary': {k:v for k,v in results.items() if k != 'trades' and k != 'equity_curve'},
    'trades': trades,
    'timestamp': datetime.now().isoformat(),
}
out = Path(__file__).parent / 'reports' / f'backtest_v2_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)
print(f'\n报告: {out}')
