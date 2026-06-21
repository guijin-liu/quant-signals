#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模拟回测演示脚本 - 不需要联网，用模拟数据展示完整流程
运行: python run_demo.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd, numpy as np, logging, warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

print("""
╔══════════════════════════════════════════════════════════╗
║       量化交易系统 - 模拟回测演示                          ║
║       标的: 神火股份 雅化集团 锡业股份 亚钾国际              ║
║       策略: 均线+MACD+RSI+量价+多维度共振                   ║
╚══════════════════════════════════════════════════════════╝
""")

# ===== 第1步: 生成模拟数据 =====
print("[Step 1/5] 生成3年模拟日线数据...")
np.random.seed(123)
stocks = {"000933":"神火股份","002497":"雅化集团","000960":"锡业股份","000893":"亚钾国际"}
all_data = []
dates = pd.date_range("2023-01-01","2026-06-19",freq="B")
for code, name in stocks.items():
    price = float(np.random.uniform(8, 25))
    trend = np.sin(np.linspace(0, 8*np.pi, len(dates))) * 5
    for i, date in enumerate(dates):
        price += trend[i] * 0.03 + np.random.normal(0, 0.3)
        price = max(price, 5.0)
        all_data.append({"datetime":date,"symbol":code,"name":name,
            "open":price*(1+np.random.uniform(-0.005,0.005)),
            "high":price*(1+np.random.uniform(0,0.015)),
            "low":price*(1-np.random.uniform(0,0.015)),
            "close":price,"volume":np.random.randint(5e6,5e7),
            "amount":np.random.randint(5e7,5e8),"turnover_rate":np.random.uniform(0.5,5)})
sim_df = pd.DataFrame(all_data)
print(f"  -> 生成 {len(sim_df)} 条数据 (4只股票 x 约900个交易日)")

# ===== 第2步: 特征工程 =====
print("\n[Step 2/5] 计算技术指标特征...")
from features.technical import compute_all_technical_features
all_feat = [compute_all_technical_features(sim_df[sim_df["symbol"]==c].copy(), "daily") for c in stocks]
feature_df = pd.concat(all_feat, ignore_index=True)

# 补充其他维度特征
feature_df["sh_pct_change"]=np.random.normal(0.03,0.4,len(feature_df))
feature_df["sector_strength_score"]=np.random.uniform(0.4,0.85,len(feature_df))
feature_df["flow_capital_score"]=np.random.uniform(-0.3,0.7,len(feature_df))
feature_df["news_sentiment_score"]=np.random.uniform(-0.2,0.5,len(feature_df))
feature_df["us_weighted_return"]=np.random.normal(0.001,0.008,len(feature_df))

from features.pipeline import build_training_target
feature_df = build_training_target(feature_df, horizon=3, threshold=0.015)
exclude = {"datetime","date","symbol","name","stock_name","target_3class","target_binary","target_regression","future_return","index_code","index_name","is_open_period","is_close_period"}
feature_cols = [c for c in feature_df.columns if c not in exclude and str(feature_df[c].dtype) in ('float64','int64','int32','float32','bool')]

print(f"  -> 共 {len(feature_cols)} 个特征:")
print(f"     均线类: MA5/MA10/MA20/MA60, 均线多头排列, 均线粘合度")
print(f"     MACD类: DIF/DEA/柱状图, 金叉死叉, 背离检测")
print(f"     RSI类: RSI值, 超买超卖, RSI趋势")
print(f"     布林带: 上轨/中轨/下轨, %B, 带宽, 突破信号")
print(f"     成交量: 量比, 放量缩量, 量价配合")
print(f"     大盘/板块/资金/情绪/美股映射")

# ===== 第3步: 模型训练 =====
print("\n[Step 3/5] 训练 GradientBoosting 模型...")
df_clean = feature_df.dropna(subset=feature_cols+["target_3class"])
df_clean = df_clean[~df_clean["target_3class"].isna()]
split = int(len(df_clean)*0.8)
train_df, test_df = df_clean.iloc[:split], df_clean.iloc[split:]
X_train = train_df[feature_cols].fillna(0).astype(float); y_train = train_df["target_3class"].astype(int)
X_test = test_df[feature_cols].fillna(0).astype(float); y_test = test_df["target_3class"].astype(int)

from models.train import train_model, save_model
model, results = train_model(X_train, y_train, X_test, y_test)
save_model(model, "xgboost_latest")
print(f"  -> 模型: {results['model_type']}")
print(f"  -> 准确率: {results['accuracy']:.1%} (预测次日涨/跌/平)")
print(f"  -> 最重要特征: {', '.join(results['feature_importance'].head(5)['feature'].values)}")

# ===== 第4步: 信号生成 =====
print("\n[Step 4/5] 生成交易信号 (多维共振过滤)...")
from models.signal import SignalGenerator
gen = SignalGenerator(model=model)
gen.buy_threshold = 0.45
gen.resonance_required = 2
signals = gen.generate_signals(test_df, feature_cols)

bc = (signals["signal"]=="BUY").sum()
sc = (signals["signal"]=="SELL").sum()
hc = (signals["signal"]=="HOLD").sum()
print(f"  -> BUY={bc}  SELL={sc}  HOLD={hc}")
print(f"  -> 信号机制: 6个维度(技术+大盘+板块+资金+情绪+美股)中至少2个同时看多才出BUY")

# ===== 第5步: 回测 =====
print("\n[Step 5/5] 运行回测 (5万初始资金, T+1, 手续费)...")
from backtest.engine import BacktestEngine
engine = BacktestEngine()
bt = engine.run(signals)

print(f"""
  ┌─────────────────────────────────────────┐
  │           回 测 结 果                     │
  ├─────────────────────────────────────────┤
  │  初始资金:   RMB {engine.initial_capital:>10,.0f}              │
  │  最终权益:   RMB {engine.capital:>10,.0f}              │
  │  总收益率:   {bt['total_return']:>12.2f}%              │
  │  总盈亏:     RMB {bt['total_pnl']:>10,.0f}              │
  │                                         │
  │  交易次数:   {bt['total_trades']:>10} 笔              │
  │  胜率:       {bt['win_rate']:>11.1f}%              │
  │  盈亏比:     {bt['profit_factor']:>12.2f}              │
  │  夏普比率:   {bt['sharpe']:>12.2f}              │
  │  最大回撤:   {bt['max_drawdown']:>11.2f}%              │
  │  卡玛比率:   {bt['calmar']:>12.2f}              │
  └─────────────────────────────────────────┘
""")

# 打印交易明细
if bt['trades']:
    print("  最近交易明细:")
    print(f"  {'时间':<20s} {'股票':<10s} {'入场':>8s} {'出场':>8s} {'盈亏':>10s} {'收益率':>8s} {'原因':<15s}")
    print("  " + "-"*85)
    for t in bt['trades'][-10:]:
        print(f"  {str(t['entry_date']):<20s} {t.get('name',t['symbol']):<10s} {t['entry_price']:>8.2f} {t['exit_price']:>8.2f} RMB{t['pnl']:>7.0f} {t['pnl_pct']:>7.2f}% {t['reason']:<15s}")

print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║  运行方式:                                                ║
  ║                                                          ║
  ║  模拟回测(不需联网):  python run_demo.py                  ║
  ║  真实数据回测(需联网): python main.py backtest             ║
  ║  查看最新信号(需联网): python main.py signal               ║
  ║  完整流程(需联网):     python main.py all                  ║
  ║  图形菜单:              run.bat                           ║
  ║                                                          ║
  ║  注: 公司网络屏蔽了东方财富数据源,真实数据需                      ║
  ║      在家网络或手机热点下运行                                 ║
  ╚══════════════════════════════════════════════════════════╝
""")
