"""
A股数据获取：个股日线K线 + 大盘指数
数据源: 腾讯财经 (稳定, 不限IP)
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from config import (
    STOCK_CODES, STOCK_POOL, TIMEFRAMES, A_MARKET_INDICES,
    CACHE_TTL,
)
from data.fetcher import cache

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
}


def _market_prefix(symbol: str) -> str:
    """腾讯接口市场前缀: sz=深圳, sh=上海"""
    return "sz" if symbol.startswith(("0", "3")) else "sh"


def get_stock_daily(symbol: str, years: int = 5, use_cache: bool = True) -> pd.DataFrame:
    """
    从腾讯财经获取个股日线K线
    每次请求最多返回~400条，需要分批获取多年数据
    """
    cache_key = f"daily_qq_{symbol}_{years}y"
    if use_cache:
        df = cache.get(cache_key, 86400)
        if df is not None:
            logger.debug(f"{symbol} 命中缓存")
            return df

    prefix = _market_prefix(symbol)
    all_rows = []

    # 腾讯接口一次最多约400条，需要分批
    days_needed = years * 252
    for offset in range(0, days_needed, 380):
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        params = {
            "param": f"{prefix}{symbol},day,,,{380},qfq",
        }
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            data = r.json()
            if data.get("code") != 0:
                logger.warning(f"腾讯接口返回异常: {data}")
                break

            stock_key = f"{prefix}{symbol}"
            stock_data = data.get("data", {}).get(stock_key, {})
            klines = stock_data.get("qfqday") or stock_data.get("day") or []

            if not klines:
                break

            for line in klines:
                all_rows.append({
                    "datetime": line[0],
                    "open": float(line[1]),
                    "close": float(line[2]),
                    "high": float(line[3]),
                    "low": float(line[4]),
                    "volume": int(float(line[5])),
                })

            if len(klines) < 380:
                break  # 已获取全部数据

        except Exception as e:
            logger.error(f"获取 {symbol} K线失败: {e}")
            break

    if not all_rows:
        return pd.DataFrame()

    # 去重排序
    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.drop_duplicates("datetime", inplace=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 计算衍生字段
    df["pct_change"] = df["close"].pct_change() * 100
    df["symbol"] = symbol
    df["name"] = STOCK_POOL.get(symbol, {}).get("name", "")

    # 截取所需年份
    cutoff = datetime.now() - timedelta(days=365 * years)
    df = df[df["datetime"] >= cutoff]

    if not df.empty and use_cache:
        cache.set(cache_key, df)

    return df


def get_all_stocks_daily(years: int = 5, use_cache: bool = True) -> dict:
    """获取所有股票日线"""
    result = {}
    for code in STOCK_CODES:
        df = get_stock_daily(code, years, use_cache)
        if not df.empty:
            result[code] = df
            logger.info(f"{STOCK_POOL[code]['name']}({code}): {len(df)}条, "
                       f"{df['datetime'].min().date()}~{df['datetime'].max().date()}, "
                       f"价格 {df['close'].iloc[0]:.2f}~{df['close'].iloc[-1]:.2f}")
        else:
            logger.warning(f"{code} 获取失败")
    return result


def get_market_indices(tf: str = "daily", use_cache: bool = True) -> dict:
    """获取大盘指数日线（使用腾讯接口）"""
    result = {}
    index_map = {
        "sh000001": "sh000001",  # 上证
        "sz399001": "sz399001",  # 深证
        "sz399006": "sz399006",  # 创业板
    }

    for idx_code, name in A_MARKET_INDICES.items():
        cache_key = f"index_qq_{idx_code}"
        if use_cache:
            df = cache.get(cache_key, 86400)
            if df is not None:
                result[idx_code] = df
                continue

        qq_code = index_map.get(idx_code, idx_code)
        prefix = qq_code[:2]
        code = qq_code[2:]

        try:
            url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            params = {"param": f"{prefix}{code},day,,,400,qfq"}
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            data = r.json()

            stock_key = f"{prefix}{code}"
            stock_data = data.get("data", {}).get(stock_key, {})
            klines = stock_data.get("qfqday") or stock_data.get("day") or []

            if klines:
                rows = [{"datetime": l[0], "open": float(l[1]), "close": float(l[2]),
                         "high": float(l[3]), "low": float(l[4]), "volume": int(float(l[5]))}
                        for l in klines]
                df = pd.DataFrame(rows)
                df["datetime"] = pd.to_datetime(df["datetime"])
                df["pct_change"] = df["close"].pct_change() * 100
                df["index_code"] = idx_code
                df["index_name"] = name
                df.sort_values("datetime", inplace=True)
                df.reset_index(drop=True, inplace=True)
                if use_cache:
                    cache.set(cache_key, df)
                result[idx_code] = df
                logger.info(f"{name}: {len(df)}条")
        except Exception as e:
            logger.warning(f"获取指数 {name} 失败: {e}")

    return result


def get_stock_kline(symbol: str, tf: str = "daily", use_cache: bool = True) -> pd.DataFrame:
    """统一接口"""
    return get_stock_daily(symbol, 5, use_cache)


def get_all_stocks_kline(tf: str = "daily", use_cache: bool = True) -> dict:
    """统一接口"""
    return get_all_stocks_daily(5, use_cache)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== 测试腾讯数据源 ===")
    daily = get_all_stocks_daily(3)
    for code, df in daily.items():
        print(f"\n{code} {STOCK_POOL[code]['name']}:")
        print(f"  条数: {len(df)}")
        print(f"  范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
        print(f"  最新: open={df['open'].iloc[-1]}, close={df['close'].iloc[-1]}")
        print(f"  最近5天涨跌幅: {df['pct_change'].tail(5).tolist()}")
