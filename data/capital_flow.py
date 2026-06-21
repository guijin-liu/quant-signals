"""
资金流向数据获取 — 修复版
- 主力资金净流入(大单/超大单)
- 北向资金(沪股通+深股通)
- 个股资金流向
"""
import logging
import pandas as pd
import akshare as ak
from config import STOCK_CODES, STOCK_POOL, CACHE_TTL
from data.fetcher import cache, retry_on_fail

logger = logging.getLogger(__name__)


@retry_on_fail
def get_north_flow() -> pd.DataFrame:
    """北向资金流向(日级别)"""
    try:
        df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
        if df is not None and not df.empty:
            col_map = {}
            for col in df.columns:
                if "日期" in col: col_map[col] = "datetime"
                elif "净流入" in col and "成交" in col: col_map[col] = "net_inflow"
                elif "买入" in col: col_map[col] = "buy_amount"
                elif "卖出" in col: col_map[col] = "sell_amount"
            df = df.rename(columns=col_map)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)
            logger.info(f"北向资金: {len(df)}条")
            return df
    except Exception as e:
        logger.warning(f"北向资金: {e}")
    return pd.DataFrame()


@retry_on_fail
def get_market_fund_flow() -> pd.DataFrame:
    """市场整体资金流向(主力/超大单/大单/中单/小单)"""
    try:
        df = ak.stock_market_fund_flow()
        if df is not None and not df.empty:
            col_map = {}
            for col in df.columns:
                if "日期" in col: col_map[col] = "datetime"
                elif "主力" in col and "净额" in col: col_map[col] = "main_net_inflow"
                elif "主力" in col and "占比" in col: col_map[col] = "main_net_pct"
                elif "超大单" in col and "净额" in col: col_map[col] = "super_large_net"
                elif "大单" in col and "净额" in col: col_map[col] = "large_net"
            df = df.rename(columns=col_map)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            logger.info(f"市场资金: {len(df)}条")
            return df
    except Exception as e:
        logger.warning(f"市场资金: {e}")
    return pd.DataFrame()


@retry_on_fail
def get_stock_fund_flow_individual(symbol: str) -> pd.DataFrame:
    """个股资金流向"""
    try:
        market = "sz" if symbol.startswith(("0", "2", "3")) else "sh"
        df = ak.stock_individual_fund_flow(stock=symbol, market=market)
        if df is not None and not df.empty:
            col_map = {}
            for col in df.columns:
                if "日期" in col: col_map[col] = "datetime"
                elif "主力" in col and "净额" in col: col_map[col] = "main_net"
                elif "主力" in col and "占比" in col: col_map[col] = "main_net_pct"
                elif "超大单" in col and "净额" in col: col_map[col] = "super_large_net"
                elif "大单" in col and "净额" in col: col_map[col] = "large_net"
                elif "中单" in col and "净额" in col: col_map[col] = "mid_net"
                elif "小单" in col and "净额" in col: col_map[col] = "small_net"
            df = df.rename(columns=col_map)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            df["symbol"] = symbol
            return df
    except Exception as e:
        logger.warning(f"个股{symbol}资金: {e}")
    return pd.DataFrame()


@retry_on_fail
def get_sector_fund_flow() -> pd.DataFrame:
    """板块资金流向排名"""
    try:
        df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流向")
        if df is not None and not df.empty:
            col_map = {}
            for col in df.columns:
                if "名称" in col: col_map[col] = "sector_name"
                elif "主力" in col and "净额" in col: col_map[col] = "main_net"
            df = df.rename(columns=col_map)
            logger.info(f"板块资金: {len(df)}条")
            return df
    except Exception as e:
        logger.warning(f"板块资金: {e}")
    return pd.DataFrame()


def get_all_capital_flow(use_cache: bool = True) -> dict:
    """获取所有资金流向数据"""
    result = {
        "north_flow": pd.DataFrame(),
        "market_flow": pd.DataFrame(),
        "stock_flows": {},
        "sector_flow": pd.DataFrame(),
    }
    ttl = CACHE_TTL.get("capital_flow", 600)

    # 北向
    key = "north_flow"
    if use_cache:
        result["north_flow"] = cache.get(key, ttl) or pd.DataFrame()
    if result["north_flow"].empty:
        result["north_flow"] = get_north_flow()
        if not result["north_flow"].empty:
            cache.set(key, result["north_flow"])

    # 市场
    key = "market_flow"
    if use_cache:
        result["market_flow"] = cache.get(key, ttl) or pd.DataFrame()
    if result["market_flow"].empty:
        result["market_flow"] = get_market_fund_flow()
        if not result["market_flow"].empty:
            cache.set(key, result["market_flow"])

    # 板块
    key = "sector_flow"
    if use_cache:
        result["sector_flow"] = cache.get(key, ttl) or pd.DataFrame()
    if result["sector_flow"].empty:
        result["sector_flow"] = get_sector_fund_flow()
        if not result["sector_flow"].empty:
            cache.set(key, result["sector_flow"])

    # 个股
    for code in STOCK_CODES:
        key = f"stock_flow_{code}"
        if use_cache:
            df = cache.get(key, ttl)
            if df is not None:
                result["stock_flows"][code] = df
                continue
        df = get_stock_fund_flow_individual(code)
        if not df.empty:
            cache.set(key, df)
            result["stock_flows"][code] = df

    return result


def compute_capital_flow_trend(stock_flow_df: pd.DataFrame) -> dict:
    """计算个股资金流向趋势"""
    if stock_flow_df.empty or "main_net" not in stock_flow_df.columns:
        return {"main_net_3d": 0, "main_net_5d": 0, "trend": "flat"}

    df = stock_flow_df.sort_values("datetime")
    net_col = "main_net"
    net_3d = df[net_col].tail(3).sum() if len(df) >= 3 else df[net_col].sum()
    net_5d = df[net_col].tail(5).sum() if len(df) >= 5 else df[net_col].sum()

    if net_5d > 0 and net_3d > 0:
        trend = "inflow"
    elif net_5d < 0 and net_3d < 0:
        trend = "outflow"
    else:
        trend = "diverging"

    return {
        "main_net_3d": round(net_3d, 2),
        "main_net_5d": round(net_5d, 2),
        "trend": trend,
    }
