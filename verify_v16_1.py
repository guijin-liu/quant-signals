"""v16.1 全面终检验证"""
import os, sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

print("=" * 60)
print("v16.1 全面终检")
print("=" * 60)

# 1. 股票池
from stock_pool import STOCK_POOL, STOCK_PARAMS, DEFAULT_SELL_PCT
p = set(STOCK_POOL.keys())
a = set(STOCK_PARAMS.keys())
assert len(p) == len(a) == 59, f"池子{len(p)}≠参数{len(a)}"
assert p == a, f"池子参数不匹配"
assert "159941" not in p and "513100" not in p, "纳指未剔除"
print(f"1. stock_pool: {len(p)}只（已剔除纳指）✓")

# 2. mx_fetcher
import mx_fetcher
fns = ['fetch_kline','get_quotes','get_financial_quality','get_market_brief','score_financial_quality']
for fn in fns:
    assert hasattr(mx_fetcher, fn), f"mx_fetcher缺{fn}"
print(f"2. mx_fetcher: {len(fns)}个函数 ✓")

# 3. cloud_function
import cloud_function
assert hasattr(cloud_function, 'main_loop')
assert hasattr(cloud_function, 'score_buy')
assert hasattr(cloud_function, 'score_sell')
assert hasattr(cloud_function, 'compute_features')
assert hasattr(cloud_function, 'batch_pre_screen')
src = open('cloud_function.py', encoding='utf-8').read()
assert 'STOCK_PARAMS.get(code' in src
assert 'DEFAULT_SELL_PCT' in src
assert 'predicted_gain = atr_pct' not in src, "残留旧公式"
assert 'gain_pct * 0.7' not in src, "残留二次7折"
print(f"3. cloud_function: ✓ 旧代码已清理")

# 4. 测试卖点策略
import numpy as np
fake = {
    'close': 25.0, 'ma5': 24.5, 'ma10': 24.0, 'ma20': 24.5, 'ma60': 22.0,
    'golden': True, 'bb_pct': 0.2, 'rsi': 50, 'j': 25,
    'vol_ratio': 1.2, 'obv_up': True, 'above_ma60': True,
    'bb_width': 0.05, 'atr': 0.5, 'pos': 0.5, 'dead': False,
}
# 数据驱动（47指标回溯卖点）
b1, r1, t1, g1 = cloud_function.score_buy('000630', fake)
assert b1 and g1 == 8.0, f"铜陵卖点错误: {g1}"
b2, r2, t2, g2 = cloud_function.score_buy('600497', fake)
assert b2 and g2 == 12.9, f"驰宏卖点错误: {g2}"
b3, r3, t3, g3 = cloud_function.score_buy('603986', fake)
assert b3 and g3 == 15.7, f"兆易卖点错误: {g3}"
# 纳指已剔除，不应有参数
assert "159941" not in STOCK_PARAMS and "513100" not in STOCK_PARAMS, "纳指还在参数里"
print(f"4. 卖点策略: 数据驱动 铜陵8.0% 驰宏12.9% 兆易15.7% ✓")

# 5. GitHub Actions workflow（拆分上午/下午两段）
for wf_name in ('quant-scan-am.yml', 'quant-scan-pm.yml'):
    wf = open(f'.github/workflows/{wf_name}', encoding='utf-8').read()
    assert 'MX_APIKEY' in wf, f"{wf_name}缺MX_APIKEY"
    assert 'cloud_function.py' in wf
assert 'SCAN_DEADLINE' in open('.github/workflows/quant-scan-am.yml', encoding='utf-8').read()
print("5. GitHub Actions: 上午/下午两段 + SCAN_DEADLINE ✓")

# 6. live_watcher
import live_watcher
assert hasattr(live_watcher, 'STOCKS')
assert len(live_watcher.STOCKS) == 59, f"live_watcher池子: {len(live_watcher.STOCKS)}"
print("6. live_watcher: 59只池 ✓")

# 7. backtest_miner
import backtest_miner
assert backtest_miner.POOL == STOCK_POOL
print("7. backtest_miner: 使用统一池 ✓")

print()
print("=" * 60)
print("全部7项检查通过! v16.1 终检完成")
print("=" * 60)
