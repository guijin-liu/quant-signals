"""
美股数据获取：三大指数 + 映射个股
使用 akshare + yfinance 互补
"""

import logging
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from config import US_INDICES, US_MAPPING, STOCK_CODES, CACHE_TTL
from data.fetcher import cache, retry_on_fail

logger = logging.getLogger(__name__)


@retry_on_fail
def _fetch_us_index_ak(index_code: str, period: str = "15") -> pd.DataFrame:
    """
    通过akshare获取美股指数分钟K线
    index_code: e.g. '.IXIC' (纳斯达克), '.DJI' (道琼斯), '.INX' (标普500)
    """
    try:
        df = ak.index_us_stock_sina(symbol=index_code)
        return df if df is not None else pd.DataFrame()
    except Exception:
        pass

    # 备选：用 yfinance
    try:
        import yfinance as yf
        ticker_map = {
            "^GSPC": "^GSPC",
            "^IXIC": "^IXIC",
            "^DJI": "^DJI",
        }
        ticker = ticker_map.get(index_code, index_code)
        t = yf.Ticker(ticker)
        df = t.history(period="5d", interval="15m")
        if df is not None and not df.empty:
            df.reset_index(inplace=True)
            df.rename(columns={
                "Datetime": "datetime", "Open": "open", "Close": "close",
                "High": "high", "Low": "low", "Volume": "volume",
            }, inplace=True)
            return df
    except Exception as e:
        logger.warning(f"yfinance获取美股指数失败: {e}")

    return pd.DataFrame()


def get_us_indices(tf: str = "15min", use_cache: bool = True) -> dict:
    """获取美股三大指数近期数据"""
    result = {}
    for code, name in US_INDICES.items():
        cache_key = f"us_index_{code}_{tf}"
        ttl = CACHE_TTL["us_index_kline"]

        if use_cache:
            df = cache.get(cache_key, ttl)
            if df is not None:
                result[code] = df
                continue

        df = _fetch_us_index_ak(code)
        if df is not None and not df.empty:
            if use_cache:
                cache.set(cache_key, df)
            result[code] = df
            logger.info(f"获取 {name} {tf}: {len(df)} 条")

    return result


@retry_on_fail
def _fetch_us_stock_yf(ticker: str) -> pd.DataFrame:
    """通过yfinance获取美股个股日线（用于隔夜复盘）"""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        df = t.history(period="1mo", interval="1d")
        if df is not None and not df.empty:
            df.reset_index(inplace=True)
            df.rename(columns={
                "Date": "datetime", "Open": "open", "Close": "close",
                "High": "high", "Low": "low", "Volume": "volume",
            }, inplace=True)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            df["ticker"] = ticker
            return df
    except Exception as e:
        logger.warning(f"yfinance获取美股 {ticker} 失败: {e}")
    return pd.DataFrame()


def get_us_mapping_data(use_cache: bool = True) -> dict:
    """
    获取所有映射美股数据
    返回: {a_stock_code: {us_ticker: DataFrame}}
    """
    result = {}
    for a_code, us_stocks in US_MAPPING.items():
        result[a_code] = {}
        for us_ticker, info in us_stocks.items():
            cache_key = f"us_stock_{us_ticker}_daily"
            ttl = CACHE_TTL["us_stock_kline"]

            if use_cache:
                df = cache.get(cache_key, ttl)
                if df is not None:
                    result[a_code][us_ticker] = df
                    continue

            df = _fetch_us_stock_yf(us_ticker)
            if df is not None and not df.empty:
                if use_cache:
                    cache.set(cache_key, df)
                result[a_code][us_ticker] = df
                logger.info(f"获取美股 {info['name']}({us_ticker}): {len(df)} 条")

    return result


def compute_us_overnight_impact(us_data: dict) -> dict:
    """
    计算隔夜美股对A股的预期影响
    返回: {a_stock_code: impact_score}
    impact_score: -1到1，正=利好，负=利空
    """
    impact = {}
    for a_code, us_stocks_data in us_data.items():
        if a_code not in US_MAPPING:
            continue

        score = 0.0
        for us_ticker, weight_info in US_MAPPING[a_code].items():
            weight = weight_info["weight"]
            df = us_stocks_data.get(us_ticker)
            if df is None or df.empty or "close" not in df.columns:
                continue
            # 用最近2日涨跌幅做信号
            if len(df) >= 2:
                ret = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1)
                score += weight * min(max(ret * 5, -1), 1)  # 缩放并裁剪

        impact[a_code] = round(score, 4)

    return impact


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== 测试美股数据获取 ===")
    us_data = get_us_mapping_data()
    for a_code, stocks in us_data.items():
        print(f"\n{a_code} 映射美股:")
        for ticker, df in stocks.items():
            print(f"  {ticker}: {len(df)} 条")

    impact = compute_us_overnight_impact(us_data)
    print(f"\n隔夜影响评分: {impact}")
