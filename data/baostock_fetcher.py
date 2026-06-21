"""
Baostock 数据适配器
替代 akshare (东方财富被公司网络屏蔽)，使用 baostock 获取A股数据
baostock 官网: http://baostock.com/
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import baostock as bs

from config import STOCK_CODES, STOCK_POOL, DATA_DIR, CACHE_DIR

logger = logging.getLogger(__name__)


def _ensure_login():
    """确保 baostock 已登录"""
    bs.login()


def _ensure_logout():
    """登出"""
    try:
        bs.logout()
    except:
        pass


def _code_to_baostock(code: str) -> str:
    """将股票代码转换为 baostock 格式 (sz.000933 / sh.600123)"""
    if code.startswith(('0', '3', '2')):
        return f'sz.{code}'
    elif code.startswith(('6', '9')):
        return f'sh.{code}'
    return f'sz.{code}'


def _baostock_to_akshare_format(df: pd.DataFrame, code: str, freq: str = 'daily') -> pd.DataFrame:
    """
    将 baostock 返回的 DataFrame 转换为与 akshare 兼容的格式
    baostock 列: date, time, code, open, high, low, close, volume, amount
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # 转换数据类型
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    # 处理时间列
    if freq == 'daily':
        df['datetime'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    else:
        # 分钟数据有 time 列
        if 'time' in df.columns:
            # baostock time format: 20260615093500000
            df['time_clean'] = df['time'].str[:12]  # 取 YYYYMMDDHHMM
            df['datetime'] = pd.to_datetime(df['time_clean'], format='%Y%m%d%H%M')
        else:
            df['datetime'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    df['symbol'] = code
    df['name'] = STOCK_POOL.get(code, {}).get('name', '')

    # 计算涨跌幅
    if 'close' in df.columns and len(df) > 1:
        df['pct_change'] = df['close'].pct_change() * 100
        df['change'] = df['close'].diff()

    # 排序
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def fetch_daily_kline(code: str, years: int = 5) -> pd.DataFrame:
    """
    获取单只股票日线数据

    Args:
        code: 股票代码 (如 '000933')
        years: 回看年数

    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume, amount, symbol, name, pct_change
    """
    _ensure_login()

    bs_code = _code_to_baostock(code)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')

    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,code,open,high,low,close,volume,amount,turn,pctChg',
            start_date=start_date,
            end_date=end_date,
            frequency='d',
            adjustflag='2'  # 前复权
        )

        data = []
        while rs.next():
            data.append(rs.get_row_data())

        if not data:
            logger.warning(f"{code}: 无日线数据")
            return pd.DataFrame()

        cols = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
        df = pd.DataFrame(data, columns=cols)

        # 用 baostock 自带的 pctChg
        df['pct_change'] = pd.to_numeric(df['pctChg'], errors='coerce')

        return _baostock_to_akshare_format(df, code, 'daily')

    except Exception as e:
        logger.error(f"获取 {code} 日线失败: {e}")
        return pd.DataFrame()


def fetch_all_daily_klines(years: int = 5) -> dict:
    """
    获取所有股票池的日线数据

    Returns:
        {code: DataFrame}
    """
    result = {}
    for code in STOCK_CODES:
        name = STOCK_POOL[code]['name']
        logger.info(f"获取 {code} {name} 日线数据...")
        df = fetch_daily_kline(code, years)
        if not df.empty:
            logger.info(f"  -> {len(df)} 条, {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")
        result[code] = df
    return result


def fetch_minute_kline(code: str, freq: str = '15', days: int = 30) -> pd.DataFrame:
    """
    获取单只股票分钟K线数据

    Args:
        code: 股票代码
        freq: '5' (5分钟), '15' (15分钟), '30', '60'
        days: 回看天数

    Returns:
        DataFrame
    """
    _ensure_login()

    bs_code = _code_to_baostock(code)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,time,code,open,high,low,close,volume,amount',
            start_date=start_date,
            end_date=end_date,
            frequency=freq,
            adjustflag='2'
        )

        data = []
        while rs.next():
            data.append(rs.get_row_data())

        if not data:
            logger.warning(f"{code}: 无{freq}分钟数据")
            return pd.DataFrame()

        cols = ['date', 'time', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
        df = pd.DataFrame(data, columns=cols)

        return _baostock_to_akshare_format(df, code, freq)

    except Exception as e:
        logger.error(f"获取 {code} {freq}min失败: {e}")
        return pd.DataFrame()


def fetch_all_minute_klines(freq: str = '15', days: int = 30) -> dict:
    """
    获取所有股票池的分钟数据

    Args:
        freq: '5' 或 '15'
        days: 回看天数

    Returns:
        {code: DataFrame}
    """
    result = {}
    freq_label = '5min' if freq == '5' else '15min'
    for code in STOCK_CODES:
        name = STOCK_POOL[code]['name']
        logger.info(f"获取 {code} {name} {freq_label}数据...")
        df = fetch_minute_kline(code, freq, days)
        if not df.empty:
            logger.info(f"  -> {len(df)} 条, {df['datetime'].min()} ~ {df['datetime'].max()}")
        result[code] = df
    return result


def fetch_latest_daily_data(code: str, days: int = 1) -> pd.DataFrame:
    """获取最新几天的日线数据 (快速查询)"""
    return fetch_daily_kline(code, years=max(1, days // 250 + 1))


# ==================== 缓存辅助 ====================

def fetch_daily_with_cache(code: str, years: int = 5, ttl_hours: int = 4) -> pd.DataFrame:
    """带缓存的日线数据获取"""
    from data.fetcher import cache

    cache_key = f"baostock_daily_{code}_{years}y"
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

    cache_key = f"baostock_{freq}min_{code}_{days}d"
    df = cache.get(cache_key, ttl_minutes * 60)
    if df is not None:
        logger.info(f"  {code} {freq}min 命中缓存: {len(df)}条")
        return df

    df = fetch_minute_kline(code, freq, days)
    if not df.empty:
        cache.set(cache_key, df)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试
    print("=== 日线测试 ===")
    dailies = fetch_all_daily_klines(3)
    for code, df in dailies.items():
        if not df.empty:
            print(f"{code}: {len(df)}条, {df['datetime'].min().date()} ~ {df['datetime'].max().date()}")
            print(df.tail(3)[['datetime', 'open', 'close', 'volume', 'pct_change']])

    print("\n=== 15分钟测试 ===")
    minutes = fetch_all_minute_klines('15', 7)
    for code, df in minutes.items():
        if not df.empty:
            print(f"{code}: {len(df)}条")

    _ensure_logout()
