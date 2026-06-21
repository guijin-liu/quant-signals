"""
板块数据获取 — 行业/概念板块指数及排名
修复版：使用IPv4强制+更新akshare API名称
"""
import logging
import pandas as pd
import akshare as ak
from config import SECTORS, CACHE_TTL
from data.fetcher import cache, retry_on_fail

logger = logging.getLogger(__name__)


@retry_on_fail
def get_sector_rank() -> pd.DataFrame:
    """获取行业板块排名（涨跌幅排序）"""
    try:
        # 新版akshare函数名
        df = ak.stock_board_industry_name_em()
        if df is None or df.empty:
            logger.warning("板块排名为空")
            return pd.DataFrame()

        # 重命名列
        col_map = {}
        for col in df.columns:
            if "名称" in col: col_map[col] = "sector_name"
            elif "涨跌幅" in col: col_map[col] = "pct_change"
            elif "最新价" in col: col_map[col] = "price"
            elif "市值" in col: col_map[col] = "total_market_cap"
            elif "换手" in col: col_map[col] = "turnover_rate"
            elif "上涨" in col: col_map[col] = "up_count"
            elif "下跌" in col: col_map[col] = "down_count"
            elif "领涨" in col and "涨跌幅" not in col: col_map[col] = "leading_stock"
            elif "涨跌幅" in col and "领涨" in col: col_map[col] = "leading_pct"

        df = df.rename(columns=col_map)

        if "pct_change" in df.columns:
            df["pct_change"] = pd.to_numeric(df["pct_change"], errors="coerce")
            df["rank"] = df["pct_change"].rank(ascending=False)
            df.sort_values("pct_change", ascending=False, inplace=True, na_position="last")

        df.reset_index(drop=True, inplace=True)
        logger.info(f"板块排名: {len(df)}个行业, 前3: {df.iloc[:3].get('sector_name', []).tolist()}")
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

        col_map = {}
        for col in df.columns:
            if "名称" in col: col_map[col] = "sector_name"
            elif "涨跌幅" in col: col_map[col] = "pct_change"
            elif "最新价" in col: col_map[col] = "price"

        df = df.rename(columns=col_map)

        if "pct_change" in df.columns:
            df["pct_change"] = pd.to_numeric(df["pct_change"], errors="coerce")
            df["rank"] = df["pct_change"].rank(ascending=False)
            df.sort_values("pct_change", ascending=False, inplace=True, na_position="last")

        df.reset_index(drop=True, inplace=True)
        return df

    except Exception as e:
        logger.error(f"获取概念板块排名失败: {e}")
        return pd.DataFrame()


def get_relevant_sectors(use_cache: bool = True) -> dict:
    """获取相关板块数据"""
    result = {"rank": pd.DataFrame(), "concept_rank": pd.DataFrame(), "relevance": {}}
    ttl = CACHE_TTL.get("sector_data", 600)

    # 行业板块
    if use_cache:
        result["rank"] = cache.get("sector_rank", ttl) or pd.DataFrame()
    if result["rank"].empty:
        result["rank"] = get_sector_rank()
        if not result["rank"].empty:
            cache.set("sector_rank", result["rank"])

    # 概念板块
    if use_cache:
        result["concept_rank"] = cache.get("concept_sector_rank", ttl) or pd.DataFrame()
    if result["concept_rank"].empty:
        result["concept_rank"] = get_concept_sector_rank()
        if not result["concept_rank"].empty:
            cache.set("concept_sector_rank", result["concept_rank"])

    return result


def find_stock_sector_rank(sector_rank_df: pd.DataFrame, sector_names: list) -> dict:
    """在板块排名中找到指定板块"""
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
