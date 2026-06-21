"""
板块数据获取：行业板块/概念板块指数及排名
"""

import logging
import pandas as pd
import akshare as ak
from config import SECTORS, SECTOR_INDICES, CACHE_TTL
from data.fetcher import cache, retry_on_fail

logger = logging.getLogger(__name__)


@retry_on_fail
def get_sector_rank() -> pd.DataFrame:
    """
    获取行业板块实时排名（涨跌幅排序）
    """
    try:
        df = ak.stock_board_industry_name_em()
        if df is None or df.empty:
            return pd.DataFrame()

        # 标准化列名
        keep_cols = {
            "板块名称": "sector_name",
            "最新价": "price",
            "涨跌幅": "pct_change",
            "总市值": "total_market_cap",
            "换手率": "turnover_rate",
            "上涨家数": "up_count",
            "下跌家数": "down_count",
            "领涨股票": "leading_stock",
            "领涨股票-涨跌幅": "leading_pct",
        }
        cols = {k: v for k, v in keep_cols.items() if k in df.columns}
        df = df[list(cols.keys())].rename(columns=cols)
        df["rank"] = df["pct_change"].rank(ascending=False) if "pct_change" in df.columns else 0
        df.sort_values("pct_change", ascending=False, inplace=True, na_position="last")
        df.reset_index(drop=True, inplace=True)
        return df

    except Exception as e:
        logger.error(f"获取板块排名失败: {e}")
        return pd.DataFrame()


@retry_on_fail
def get_concept_sector_rank() -> pd.DataFrame:
    """获取概念板块排名"""
    try:
        df = ak.stock_board_concept_name_em()
        if df is None or df.empty:
            return pd.DataFrame()

        keep_cols = {
            "板块名称": "sector_name",
            "最新价": "price",
            "涨跌幅": "pct_change",
            "总市值": "total_market_cap",
            "换手率": "turnover_rate",
            "上涨家数": "up_count",
            "下跌家数": "down_count",
            "领涨股票": "leading_stock",
        }
        cols = {k: v for k, v in keep_cols.items() if k in df.columns}
        df = df[list(cols.keys())].rename(columns=cols)
        df["rank"] = df["pct_change"].rank(ascending=False) if "pct_change" in df.columns else 0
        df.sort_values("pct_change", ascending=False, inplace=True, na_position="last")
        df.reset_index(drop=True, inplace=True)
        return df

    except Exception as e:
        logger.error(f"获取概念板块排名失败: {e}")
        return pd.DataFrame()


@retry_on_fail
def get_sector_kline(sector_name: str, tf: str = "15min") -> pd.DataFrame:
    """
    获取指定板块的日K线
    板块指数通过东方财富获取
    """
    try:
        # 尝试直接获取板块指数K线
        df = ak.stock_board_industry_hist_em(symbol=sector_name, period="日k")
        if df is not None and not df.empty:
            col_map = {
                "日期": "datetime", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "涨跌幅": "pct_change",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            df["sector_name"] = sector_name
            df.sort_values("datetime", inplace=True)
            return df
    except Exception as e:
        logger.warning(f"获取板块 {sector_name} K线失败: {e}")

    return pd.DataFrame()


def get_relevant_sectors(use_cache: bool = True) -> dict:
    """获取相关板块数据和排名"""
    result = {"rank": pd.DataFrame(), "concept_rank": pd.DataFrame(), "relevance": {}}

    cache_key = "sector_rank"
    ttl = CACHE_TTL["sector_data"]

    if use_cache:
        df = cache.get(cache_key, ttl)
        if df is not None:
            result["rank"] = df

    if result["rank"].empty:
        result["rank"] = get_sector_rank()
        if not result["rank"].empty and use_cache:
            cache.set(cache_key, result["rank"])

    # 概念板块排名
    concept_key = "concept_sector_rank"
    if use_cache:
        df = cache.get(concept_key, ttl)
        if df is not None:
            result["concept_rank"] = df

    if result["concept_rank"].empty:
        result["concept_rank"] = get_concept_sector_rank()
        if not result["concept_rank"].empty and use_cache:
            cache.set(concept_key, result["concept_rank"])

    return result


def find_stock_sector_rank(sector_rank_df: pd.DataFrame, sector_names: list) -> dict:
    """
    在板块排名中找到指定板块的排名信息
    返回 {sector_name: {rank, pct_change, ...}}
    """
    result = {}
    if sector_rank_df.empty or "sector_name" not in sector_rank_df.columns:
        return result

    for name in sector_names:
        match = sector_rank_df[sector_rank_df["sector_name"].str.contains(name, na=False)]
        if not match.empty:
            row = match.iloc[0]
            result[name] = {
                "rank": row.get("rank", 999),
                "pct_change": row.get("pct_change", 0),
                "turnover_rate": row.get("turnover_rate", 0),
                "up_count": row.get("up_count", 0),
                "down_count": row.get("down_count", 0),
            }

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== 测试板块数据 ===")
    result = get_relevant_sectors()
    print(f"行业板块排名: {len(result['rank'])} 个板块")
    if not result['rank'].empty:
        print(result['rank'].head(10)[["sector_name", "pct_change", "rank"]])

    ranks = find_stock_sector_rank(result['rank'], SECTORS)
    print(f"\n相关板块排名: {ranks}")
