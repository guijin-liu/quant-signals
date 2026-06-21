"""
资金流向数据获取
- 主力资金净流入（大单/超大单）
- 北向资金（沪深股通）流向
- 个股资金流向
"""

import logging
import pandas as pd
import akshare as ak
from datetime import datetime
from config import STOCK_CODES, STOCK_POOL, CACHE_TTL
from data.fetcher import cache, retry_on_fail

logger = logging.getLogger(__name__)


@retry_on_fail
def get_north_flow() -> pd.DataFrame:
    """
    获取北向资金（沪股通+深股通）流向
    日级别数据
    """
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        if df is not None and not df.empty:
            col_map = {
                "日期": "datetime", "当日成交净买额": "net_inflow",
                "买入成交额": "buy_amount", "卖出成交额": "sell_amount",
                "沪股通流入": "sh_inflow", "深股通流入": "sz_inflow",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
    except Exception as e:
        logger.warning(f"获取北向资金失败: {e}")
    return pd.DataFrame()


@retry_on_fail
def get_market_fund_flow() -> pd.DataFrame:
    """
    获取市场整体资金流向（主力/超大单/大单/中单/小单）
    """
    try:
        df = ak.stock_market_fund_flow()
        if df is not None and not df.empty:
            col_map = {
                "日期": "datetime",
                "主力净流入-净额": "main_net_inflow",
                "主力净流入-净占比": "main_net_pct",
                "超大单净流入-净额": "super_large_net",
                "超大单净流入-净占比": "super_large_pct",
                "大单净流入-净额": "large_net",
                "大单净流入-净占比": "large_pct",
                "中单净流入-净额": "mid_net",
                "中单净流入-净占比": "mid_pct",
                "小单净流入-净额": "small_net",
                "小单净流入-净占比": "small_pct",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            return df
    except Exception as e:
        logger.warning(f"获取市场资金流向失败: {e}")
    return pd.DataFrame()


@retry_on_fail
def get_stock_fund_flow_individual(symbol: str) -> pd.DataFrame:
    """
    获取个股资金流向（近期）
    包含主力净流入、超大单、大单、中单、小单
    """
    try:
        df = ak.stock_individual_fund_flow(stock=symbol, market="sz" if symbol.startswith(("0", "3")) else "sh")
        if df is not None and not df.empty:
            col_map = {
                "日期": "datetime",
                "主力净流入-净额": "main_net",
                "主力净流入-净占比": "main_net_pct",
                "超大单净流入-净额": "super_large_net",
                "超大单净流入-净占比": "super_large_pct",
                "大单净流入-净额": "large_net",
                "大单净流入-净占比": "large_pct",
                "中单净流入-净额": "mid_net",
                "中单净流入-净占比": "mid_pct",
                "小单净流入-净额": "small_net",
                "小单净流入-净占比": "small_pct",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            df["symbol"] = symbol
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
    except Exception as e:
        logger.warning(f"获取个股 {symbol} 资金流向失败: {e}")
    return pd.DataFrame()


@retry_on_fail
def get_sector_fund_flow() -> pd.DataFrame:
    """获取板块资金流向排名"""
    try:
        df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流向")
        if df is not None and not df.empty:
            col_map = {
                "板块名称": "sector_name",
                "主力净流入-净额": "main_net",
                "主力净流入-净占比": "main_net_pct",
                "超大单净流入-净额": "super_large_net",
                "大单净流入-净额": "large_net",
                "中单净流入-净额": "mid_net",
                "小单净流入-净额": "small_net",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
            return df
    except Exception as e:
        logger.warning(f"获取板块资金流向失败: {e}")
    return pd.DataFrame()


def get_all_capital_flow(use_cache: bool = True) -> dict:
    """获取所有资金流向数据"""
    result = {
        "north_flow": pd.DataFrame(),
        "market_flow": pd.DataFrame(),
        "stock_flows": {},
        "sector_flow": pd.DataFrame(),
    }

    ttl = CACHE_TTL["capital_flow"]

    # 北向资金
    cache_key = "north_flow"
    if use_cache:
        df = cache.get(cache_key, ttl)
        if df is not None:
            result["north_flow"] = df
    if result["north_flow"].empty:
        result["north_flow"] = get_north_flow()
        if not result["north_flow"].empty and use_cache:
            cache.set(cache_key, result["north_flow"])

    # 市场资金流向
    cache_key = "market_flow"
    if use_cache:
        df = cache.get(cache_key, ttl)
        if df is not None:
            result["market_flow"] = df
    if result["market_flow"].empty:
        result["market_flow"] = get_market_fund_flow()
        if not result["market_flow"].empty and use_cache:
            cache.set(cache_key, result["market_flow"])

    # 个股资金流向
    for code in STOCK_CODES:
        cache_key = f"stock_flow_{code}"
        if use_cache:
            df = cache.get(cache_key, ttl)
            if df is not None:
                result["stock_flows"][code] = df
                continue
        df = get_stock_fund_flow_individual(code)
        if not df.empty:
            if use_cache:
                cache.set(cache_key, df)
            result["stock_flows"][code] = df
            logger.info(f"获取 {STOCK_POOL[code]['name']} 资金流向: {len(df)} 条")

    # 板块资金流向
    cache_key = "sector_flow"
    if use_cache:
        df = cache.get(cache_key, ttl)
        if df is not None:
            result["sector_flow"] = df
    if result["sector_flow"].empty:
        result["sector_flow"] = get_sector_fund_flow()
        if not result["sector_flow"].empty and use_cache:
            cache.set(cache_key, result["sector_flow"])

    return result


def compute_capital_flow_trend(stock_flow_df: pd.DataFrame) -> dict:
    """
    计算个股资金流向趋势指标
    返回近N日主力净流入趋势
    """
    if stock_flow_df.empty or "main_net" not in stock_flow_df.columns:
        return {"main_net_3d": 0, "main_net_5d": 0, "trend": "flat"}

    df = stock_flow_df.sort_values("datetime")
    net_col = "main_net"

    # 近3日/5日累计净流入
    net_3d = df[net_col].tail(3).sum() if len(df) >= 3 else df[net_col].sum()
    net_5d = df[net_col].tail(5).sum() if len(df) >= 5 else df[net_col].sum()

    # 趋势判断
    if net_5d > 0 and net_3d > 0:
        trend = "inflow_accelerating" if net_3d > net_5d / 2 else "inflow"
    elif net_5d < 0 and net_3d < 0:
        trend = "outflow_accelerating" if abs(net_3d) > abs(net_5d) / 2 else "outflow"
    else:
        trend = "diverging"  # 短期和中期方向不一致

    return {
        "main_net_3d": round(net_3d, 2),
        "main_net_5d": round(net_5d, 2),
        "main_net_pct": round(df["main_net_pct"].tail(5).mean(), 2) if "main_net_pct" in df.columns else 0,
        "trend": trend,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== 测试资金流向数据 ===")
    flows = get_all_capital_flow()
    print(f"北向资金: {len(flows['north_flow'])} 条")
    print(f"市场资金: {len(flows['market_flow'])} 条")
    print(f"板块资金: {len(flows['sector_flow'])} 条")
    for code, df in flows["stock_flows"].items():
        trend = compute_capital_flow_trend(df)
        print(f"{code}: {len(df)} 条, 趋势={trend}")
