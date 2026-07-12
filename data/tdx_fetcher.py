"""
通达信(mootdx)数据适配器 — A股数据主力
mootdx 直连通达信行情服务器，免费、稳定、数据全
频率码: 0=5分钟, 1=15分钟, 4=日线, 7=1分钟

优势:
  - 无需登录/登出，连接即用
  - 数据更新更及时（实时行情级别）
  - 覆盖全市场（主板+创业板+科创板）
  - 支持1/5/15/30/60分钟+日/周/月线
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from mootdx.quotes import Quotes

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STOCK_CODES, STOCK_POOL, DATA_DIR, CACHE_DIR

logger = logging.getLogger(__name__)

# 频率映射: 系统标签 → mootdx频率码
FREQ_MAP = {
    '1min': 7,
    '5min': 0,
    '15min': 1,
    '30min': None,  # mootdx 不直接支持，用 15min 聚合
    '60min': None,
    'daily': 4,
    'weekly': 5,
    'monthly': 6,
}

# 反向映射: mootdx频率码 → 标签
FREQ_LABEL = {0: '5min', 1: '15min', 4: 'daily', 5: 'weekly', 6: 'monthly', 7: '1min'}

# 全局客户端（单例，避免重复连接）
_client = None


def _get_client() -> Quotes:
    """获取或创建通达信客户端（单例）"""
    global _client
    if _client is None:
        _client = Quotes.factory(market='std')
        logger.info(f"通达信客户端已连接: {_client.server}")
    return _client


def _code_to_tdx(code: str) -> str:
    """股票代码转通达信格式 (纯数字，mootdx 自动识别市场)"""
    return code


def _tdx_to_df(df: pd.DataFrame, code: str, freq: str = 'daily') -> pd.DataFrame:
    """
    将 mootdx 返回的 DataFrame 转换为系统兼容格式
    mootdx 列: open, close, high, low, volume, amount, datetime
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # mootdx 可能将 datetime 同时设为 index 和 column，reset 解决冲突
    df = df.reset_index(drop=True).copy()

    # 确保数值类型
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    else:
        df['amount'] = df['close'] * df['volume']

    # 确保 datetime 列是 datetime 类型
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])

    # 添加标准字段
    df['symbol'] = code
    df['name'] = STOCK_POOL.get(code, {}).get('name', '')

    # 计算涨跌幅
    if 'close' in df.columns and len(df) > 1:
        df['pct_change'] = df['close'].pct_change() * 100

    # 去重 + 排序
    if 'datetime' in df.columns:
        df = df.drop_duplicates('datetime').sort_values('datetime').reset_index(drop=True)

    return df


def fetch_kline(code: str, freq: str = 'daily', count: int = 800) -> pd.DataFrame:
    """
    获取单只股票K线数据

    Args:
        code: 股票代码 (如 '000933')
        freq: '1min' | '5min' | '15min' | 'daily' | 'weekly' | 'monthly'
        count: 获取最近N根K线 (mootdx 每次最多约800根)

    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume, amount, symbol, name, pct_change
    """
    tdx_freq = FREQ_MAP.get(freq)
    if tdx_freq is None:
        logger.error(f"不支持的频率: {freq}")
        return pd.DataFrame()

    client = _get_client()

    try:
        # mootdx bars: start=0 最新, offset=count
        df = client.bars(symbol=code, frequency=tdx_freq, start=0, offset=min(count, 800))

        if df is None or df.empty:
            logger.warning(f"{code}: 无{freq}数据")
            return pd.DataFrame()

        result = _tdx_to_df(df, code, freq)
        return result

    except Exception as e:
        logger.error(f"获取 {code} {freq}失败: {e}")
        return pd.DataFrame()


def fetch_daily_kline(code: str, years: int = 5) -> pd.DataFrame:
    """
    获取单只股票日线数据（多年历史）

    Args:
        code: 股票代码
        years: 回看年数

    Returns:
        DataFrame
    """
    # 日线每次最多~800条，5年约1260个交易日，需要分批
    days_needed = years * 252
    all_dfs = []

    for offset in range(0, days_needed, 700):
        batch = fetch_kline(code, 'daily', count=min(700, days_needed - offset))
        if batch.empty:
            break
        all_dfs.append(batch)
        if len(batch) < 700:
            break

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.drop_duplicates('datetime').sort_values('datetime').reset_index(drop=True)

    # 截取所需年份
    if not result.empty:
        cutoff = datetime.now() - timedelta(days=365 * years)
        result = result[result['datetime'] >= cutoff]

    return result


def fetch_minute_kline(code: str, freq: str = '15', days: int = 30) -> pd.DataFrame:
    """
    获取单只股票分钟K线数据

    Args:
        code: 股票代码
        freq: '5' (5分钟), '15' (15分钟)
        days: 回看天数

    Returns:
        DataFrame
    """
    freq_key = '5min' if freq == '5' else '15min'

    # 分钟线每次最多~800根，一天5min有48根，15min有16根
    bars_per_day = 48 if freq == '5' else 16
    needed_bars = days * bars_per_day

    all_dfs = []
    for offset in range(0, needed_bars, 700):
        batch = fetch_kline(code, freq_key, count=min(700, needed_bars - offset))
        if batch.empty:
            break
        all_dfs.append(batch)
        if len(batch) < 700:
            break

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.drop_duplicates('datetime').sort_values('datetime').reset_index(drop=True)

    return result


def fetch_all_daily_klines(years: int = 5) -> dict:
    """获取所有股票池的日线数据"""
    result = {}
    for code in STOCK_CODES:
        name = STOCK_POOL[code]['name']
        logger.info(f"TDX 日线 {code} {name}...")
        df = fetch_daily_kline(code, years)
        if not df.empty:
            logger.info(f"  -> {len(df)}条, {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")
        else:
            logger.warning(f"  -> {code} 无数据")
        result[code] = df
    return result


def fetch_all_minute_klines(freq: str = '15', days: int = 30) -> dict:
    """获取所有股票池的分钟数据"""
    result = {}
    freq_label = '5min' if freq == '5' else '15min'
    for code in STOCK_CODES:
        name = STOCK_POOL[code]['name']
        logger.info(f"TDX {freq_label} {code} {name}...")
        df = fetch_minute_kline(code, freq, days)
        if not df.empty:
            logger.info(f"  -> {len(df)}条, {df['datetime'].min()} ~ {df['datetime'].max()}")
        else:
            logger.warning(f"  -> {code} 无数据")
        result[code] = df
    return result


# ==================== 缓存辅助 ====================

def fetch_daily_with_cache(code: str, years: int = 5, ttl_hours: int = 4) -> pd.DataFrame:
    """带缓存的日线数据获取"""
    from data.fetcher import cache

    cache_key = f"tdx_daily_{code}_{years}y"
    df = cache.get(cache_key, ttl_hours * 3600)
    if df is not None:
        logger.info(f"  {code} 命中缓存: {len(df)}条")
        return df

    df = fetch_daily_kline(code, years)
    if not df.empty:
        cache.set(cache_key, df)
    return df


def fetch_minute_with_cache(code: str, freq: str = '15', days: int = 30, ttl_minutes: int = 10) -> pd.DataFrame:
    """带缓存的分钟数据获取"""
    from data.fetcher import cache

    cache_key = f"tdx_{freq}min_{code}_{days}d"
    df = cache.get(cache_key, ttl_minutes * 60)
    if df is not None:
        logger.info(f"  {code} {freq}min 命中缓存: {len(df)}条")
        return df

    df = fetch_minute_kline(code, freq, days)
    if not df.empty:
        cache.set(cache_key, df)
    return df


# ==================== 云函数兼容接口 ====================

def fetch_minute_data_cloud(code: str, freq: str = '15', days: int = 180) -> pd.DataFrame:
    """
    云端数据获取 — mootdx直连，无需登录/登出

    Args:
        code: 股票代码
        freq: '5' 或 '15'
        days: 回看天数

    Returns:
        DataFrame with columns: date, time, open, high, low, close, volume
    """
    tdx_freq = 0 if freq == '5' else 1  # 0=5min, 1=15min

    client = _get_client()

    # 分钟线每天约16根(15min)，180天约2880根，需要分批
    bars_per_day = 48 if freq == '5' else 16
    needed = days * bars_per_day

    all_data = []
    for offset in range(0, needed, 700):
        try:
            df = client.bars(symbol=code, frequency=tdx_freq, start=offset, offset=min(700, needed - offset))
            if df is not None and not df.empty:
                all_data.append(df)
            if df is None or len(df) < 700:
                break
        except Exception as e:
            logger.error(f"{code} TDX获取失败: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)

    # 转换为 cloud_function 期望的格式: date, time, open, high, low, close, volume
    if 'datetime' in df.columns:
        dt = pd.to_datetime(df['datetime'])
        df['date'] = dt.dt.strftime('%Y-%m-%d')
        df['time'] = dt.dt.strftime('%H%M%S%f')[:6]  # HHMMSS
    elif 'date' not in df.columns:
        df['date'] = ''
    if 'time' not in df.columns:
        df['time'] = ''

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 确保关键列存在
    cols_needed = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    for col in cols_needed:
        if col not in df.columns:
            df[col] = 0

    return df[cols_needed]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== TDX 日线测试 (000933 神火股份) ===")
    df = fetch_daily_kline('000933', 3)
    if not df.empty:
        print(f"条数: {len(df)}")
        print(f"范围: {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")
        print(df.tail(5)[['datetime', 'open', 'close', 'volume', 'pct_change']])

    print("\n=== TDX 15分钟测试 (000933) ===")
    df = fetch_minute_kline('000933', '15', 7)
    if not df.empty:
        print(f"条数: {len(df)}")
        print(df.tail(5)[['datetime', 'open', 'close', 'volume']])

    print("\n=== TDX 5分钟测试 (000933) ===")
    df = fetch_minute_kline('000933', '5', 3)
    if not df.empty:
        print(f"条数: {len(df)}")
        print(df.tail(10)[['datetime', 'open', 'close', 'volume']])
