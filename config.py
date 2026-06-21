"""
量化交易系统 - 全局配置
股票池: 神火股份(000933), 雅化集团(002497), 锡业股份(000960), 亚钾国际(000893)
时间框架: 5分钟/15分钟短线
"""

from pathlib import Path

# ==================== 项目路径 ====================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"
BACKTEST_DIR = BASE_DIR / "backtest"
RISK_DIR = BASE_DIR / "risk"

# 确保关键目录存在
for d in [CACHE_DIR, SAVED_MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==================== 股票池 ====================
from stock_pool import STOCK_POOL, STOCK_CODES

# ==================== 时间框架 ====================
# akshare周期参数: '5'=5min, '15'=15min, '30'=30min, '60'=60min
TIMEFRAMES = {
    "5min": "5",
    "15min": "15",
}

PRIMARY_TF = "15min"   # 主周期：定方向
ENTRY_TF = "5min"       # 入场周期：找买点

# ==================== A股大盘指数 ====================
A_MARKET_INDICES = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
}

# ==================== 美股映射配置 ====================
# 美股标的 → 影响的A股及权重
US_MAPPING = {
    "000933": {  # 神火股份 → 美国铝业 + 煤炭ETF
        "AA": {"weight": 0.6, "name": "美国铝业"},
        "BTU": {"weight": 0.4, "name": "皮博迪能源"},
    },
    "002497": {  # 雅化集团 → 特斯拉 + 锂矿ETF
        "TSLA": {"weight": 0.4, "name": "特斯拉"},
        "ALB": {"weight": 0.4, "name": "雅保(锂矿)"},
        "LIT": {"weight": 0.2, "name": "锂矿ETF"},
    },
    "000960": {  # 锡业股份 → 费城半导体 + 锡相关
        "SOXX": {"weight": 0.6, "name": "费城半导体ETF"},
        "INTC": {"weight": 0.4, "name": "英特尔"},
    },
    "000893": {  # 亚钾国际 → 化肥相关美股
        "MOS": {"weight": 0.5, "name": "美盛(化肥)"},
        "NTR": {"weight": 0.5, "name": "Nutrien(钾肥)"},
    },
}

US_INDICES = {
    "^GSPC": "标普500",
    "^IXIC": "纳斯达克",
    "^DJI": "道琼斯",
}

# ==================== 板块配置 ====================
SECTORS = ["有色金属", "化工", "煤炭", "锂电池", "化肥"]
SECTOR_INDICES = {
    "有色金属": "bk0473",
    "化工": "bk0477",
    "煤炭": "bk0468",
    "锂电池": "bk0574",
    "化肥": "bk0478",
}

# ==================== 技术指标参数 ====================
TECH_PARAMS = {
    "ma_periods": [5, 10, 20, 60],       # MA周期（对应5min/15min K线根数）
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "rsi_period": 14,
    "boll_period": 20, "boll_std": 2,
    "volume_ma_periods": [5, 20],         # 量比计算周期
    "atr_period": 14,
}

# ==================== 训练参数 ====================
TRAIN_PARAMS = {
    "lookback_days": 180,          # 训练数据回看天数
    "retrain_frequency": 20,        # 每N个交易日重新训练
    "train_test_split": 0.8,        # 训练/测试比例
    "target_horizon": 3,            # 预测未来N根K线后的涨跌 (15min*3=45min)
    "target_threshold": 0.015,      # 涨/跌分类阈值 (1.5%以上算涨)
    "xgboost_params": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "random_state": 42,
        "n_jobs": -1,
    },
}

# ==================== 回测参数 ====================
BACKTEST_PARAMS = {
    "initial_capital": 100_000,       # 初始资金10万
    "commission_rate": 0.0003,      # 佣金 万分之三
    "stamp_tax_rate": 0.001,        # 印花税 千分之一(仅卖出)
    "transfer_fee_rate": 0.00002,   # 过户费 十万分之二
    "slippage": 0.001,              # 滑点 千分之一
    "min_hold_bars": 3,             # 最短持有K线数 (T+1约束 + 最小持仓)
    "position_pct": 0.25,           # 单只股票仓位比例
    "max_positions": 4,             # 最大同时持仓数
    "max_daily_trades": 6,          # 单日最大交易次数
}

# ==================== 风控参数 ====================
RISK_PARAMS = {
    "stop_loss_pct": 0.04,          # 固定止损 -4% (v7放宽)
    "take_profit_pct": 0.12,        # 固定止盈 +12% (v7拉大)
    "trailing_stop_atr": 1.5,       # ATR追踪止损倍数
    "max_daily_loss_pct": 0.05,     # 单日最大亏损5%
    "max_consecutive_losses": 5,    # 连续止损后暂停交易
    "kelly_fraction": 0.5,          # 半凯利仓位
}

# ==================== 信号参数 ====================
SIGNAL_PARAMS = {
    "ml_weight": 0.5,               # ML模型权重
    "factor_weight": 0.5,           # 多因子规则权重
    "buy_threshold": 0.62,          # 买入阈值 (综合评分 > 0.62)
    "sell_threshold": 0.25,         # 卖出阈值 (综合评分 < 0.25)
    "min_confidence": 0.70,         # ML最低置信度
    "resonance_required": 1,        # v7降低到1维共振,增加信号量
}

# ==================== 数据缓存 ====================
CACHE_TTL = {
    "a_stock_kline": 300,           # 5分钟
    "a_index_kline": 300,
    "us_index_kline": 3600,         # 美股1小时（非交易时段不变）
    "us_stock_kline": 3600,
    "sector_data": 600,
    "capital_flow": 600,
    "news_data": 1800,              # 新闻30分钟
}

# ==================== 日志 ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
