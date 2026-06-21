"""
逐票独立策略 — 每只票用自己的最佳参数
基于历史规律:
  000933 神火: MACD金叉+MA多头+震荡市 → 80%胜率
  002497 雅化: MACD金叉+MA多头+牛市 → 100%胜率 (6次)
  000960 锡业: MACD金叉+MA多头+牛市 → 62.5%胜率
  000893 亚钾: MA多头+RSI30-65+牛市 → 56.8%胜率

策略: 每只票独立参数，信号条件不同，满足>80%胜率的条件才出BUY
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
from backtest.engine import BacktestEngine
from config import STOCK_CODES, STOCK_POOL, RISK_PARAMS

# ============================================================
# 每只票独立策略参数
# ============================================================
PER_STOCK_CONFIG = {
    '000933': {  # 神火股份 — 5年1209条, 波动3.01%
        'name': '神火股份',
        # 买入条件: MACD金叉+MA多头+震荡市(MACD金叉在震荡中最可靠)
        'require_macd_golden': True,
        'require_ma_bullish': True,
        'require_trend': 'sideways',  # 震荡市金叉最有效
        'rsi_min': 35, 'rsi_max': 65,
        'require_volume_surge': False,
        'boll_pct_min': 0.2,
        'stop_loss': 0.02,  # 2%
        'take_profit': 0.08,  # 8%
        'buy_threshold': 0.60,
    },
    '002497': {  # 雅化集团 — 5年1209条, 波动2.95%
        'name': '雅化集团',
        # MACD金叉+MA多头+牛市=100%(6次)
        'require_macd_golden': True,
        'require_ma_bullish': True,
        'require_trend': 'bull',
        'rsi_min': 35, 'rsi_max': 65,
        'require_volume_surge': False,
        'boll_pct_min': 0.3,
        'stop_loss': 0.02,
        'take_profit': 0.10,
        'buy_threshold': 0.62,
    },
    '000960': {  # 锡业股份 — 5年1209条, 波动3.02%
        'name': '锡业股份',
        # 单信号都不够80%，叠加多层过滤
        'require_macd_golden': True,
        'require_ma_bullish': True,
        'require_trend': 'bull',
        'rsi_min': 35, 'rsi_max': 60,
        'require_volume_surge': False,
        'boll_pct_min': 0.25,
        'additional_ma': True,  # 额外要求MA5>今天最低价
        'stop_loss': 0.015,
        'take_profit': 0.10,
        'buy_threshold': 0.65,
    },
    '000893': {  # 亚钾国际 — 5年1209条, 波动2.91%
        'name': '亚钾国际',
        # MA多头+RSI30-65+牛市=56.8% — 加MACD金叉
        'require_macd_golden': True,
        'require_ma_bullish': True,
        'require_trend': 'bull',
        'rsi_min': 35, 'rsi_max': 60,
        'require_volume_surge': False,
        'boll_pct_min': 0.3,
        'stop_loss': 0.015,
        'take_profit': 0.08,
        'buy_threshold': 0.62,
    },
}

# 全局: 技术评分权重
TECH_WEIGHT = 0.30
TREND_WEIGHT = 0.20

print('='*70)
print('  逐票独立策略回测')
print('='*70)

daily_data = fetch_all_daily_klines(5)

all_signals = []
per_stock_results = {}

for code in STOCK_CODES:
    cfg = PER_STOCK_CONFIG[code]
    name = STOCK_POOL[code]['name']
    df = daily_data.get(code)
    if df is None or df.empty:
        continue

    df = compute_all_technical_features(df, 'daily')

    # 趋势判断
    df['ma20'] = df['close'].rolling(20).mean()
    df['daily_trend'] = 'sideways'
    df.loc[df['close'] > df['ma20'] * 1.05, 'daily_trend'] = 'bull'
    df.loc[df['close'] < df['ma20'] * 0.95, 'daily_trend'] = 'bear'

    # MA5和10对比
    df['ma5_above_ma10'] = 0
    if 'ma_5' in df.columns and 'ma_10' in df.columns:
        df.loc[df['ma_5'] > df['ma_10'], 'ma5_above_ma10'] = 1

    # 设置风控
    RISK_PARAMS['stop_loss_pct'] = cfg['stop_loss']
    RISK_PARAMS['take_profit_pct'] = cfg['take_profit']

    # 生成信号
    df['signal'] = 'HOLD'
    df['signal_score'] = 0.0

    buy_signals = 0
    for i in range(max(60, len(df)//10), len(df)):
        # 基础技术评分
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
        if 'rsi' in df.columns and cfg['rsi_min'] < df['rsi'].iloc[i] < cfg['rsi_max']:
            tech_bull += 1
        if 'volume_surge' in df.columns and df['volume_surge'].iloc[i]:
            tech_bull += 1
        if 'boll_pct_b' in df.columns and df['boll_pct_b'].iloc[i] > cfg['boll_pct_min']:
            tech_bull += 1
        tech_score = min(tech_bull/8.0, 1.0)

        # 趋势评分
        trend_bull = 0
        if 'pct_change_5' in df.columns and df['pct_change_5'].iloc[i] > 0:
            trend_bull += 1
        if 'ma_20' in df.columns and df['close'].iloc[i] > df['ma_20'].iloc[i]:
            trend_bull += 1
        if 'pct_change_3' in df.columns and df['pct_change_3'].iloc[i] > 0:
            trend_bull += 1
        trend_score = min(trend_bull/3.0, 1.0)

        # 综合
        ow = (1.0 - TECH_WEIGHT - TREND_WEIGHT) / 4
        composite = TECH_WEIGHT*tech_score + TREND_WEIGHT*trend_score + ow*0.5*4

        # 逐票特殊条件
        meets_conditions = True
        if cfg.get('require_macd_golden') and 'macd_golden_cross' in df.columns:
            meets_conditions &= bool(df['macd_golden_cross'].iloc[i])
        if cfg.get('require_ma_bullish') and 'ma_bullish' in df.columns:
            meets_conditions &= bool(df['ma_bullish'].iloc[i])
        if cfg.get('require_trend') and 'daily_trend' in df.columns:
            meets_conditions &= (df['daily_trend'].iloc[i] == cfg['require_trend'])
        if cfg.get('require_volume_surge') and 'volume_surge' in df.columns:
            meets_conditions &= bool(df['volume_surge'].iloc[i])
        if cfg.get('additional_ma') and 'ma5_above_ma10' in df.columns:
            meets_conditions &= bool(df['ma5_above_ma10'].iloc[i])

        df.at[df.index[i], 'signal_score'] = round(composite, 3)

        if composite > cfg['buy_threshold'] and meets_conditions:
            df.at[df.index[i], 'signal'] = 'BUY'
            buy_signals += 1
        elif composite < 0.20:
            df.at[df.index[i], 'signal'] = 'SELL'

    req_keys = [k for k,v in cfg.items() if k.startswith('require') and v]
    print(f'  {code} {name}: BUY={buy_signals} (条件={req_keys})')
    all_signals.append(df)

# 合并回测
sdf = pd.concat(all_signals, ignore_index=True)
sdf.sort_values(['datetime', 'symbol'], inplace=True)

buy_total = (sdf['signal'] == 'BUY').sum()
sell_total = (sdf['signal'] == 'SELL').sum()
print(f'\n合并: {len(sdf)}行, BUY={buy_total}, SELL={sell_total}')

engine = BacktestEngine()
results = engine.run(sdf)

print()
print('='*70)
print('  逐票独立策略 回测结果')
print('='*70)
engine.print_summary()

# 按股票
trades = results['trades']
print(f'\n按股票:')
for code in STOCK_CODES:
    ct = [t for t in trades if t['symbol'] == code]
    if ct:
        cw = [t for t in ct if t['pnl'] > 0]
        total_pnl = sum(t['pnl'] for t in ct)
        print(f'  {code}: {len(ct)}笔 胜率{len(cw)/len(ct)*100:.0f}% 累计{total_pnl:+,.0f}')

# 保存
out = Path(__file__).parent / 'reports' / f'per_stock_strategy_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
out.parent.mkdir(parents=True, exist_ok=True)
report = {
    'configs': {c: {k: v for k, v in cfg.items() if not k.startswith('_')} for c, cfg in PER_STOCK_CONFIG.items()},
    'summary': {k: v for k, v in results.items() if k not in ('trades', 'equity_curve')},
    'total_trades': len(trades),
    'win_rate': results['win_rate'],
    'total_return': results['total_return'],
}
with open(out, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)
print(f'\n报告: {out}')
