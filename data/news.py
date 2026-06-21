"""
新闻数据获取与情绪分析 — 修复版
- 个股相关新闻
- 关键词+规则情绪打分
"""
import logging
import pandas as pd
import akshare as ak
from config import STOCK_CODES, STOCK_POOL, CACHE_TTL
from data.fetcher import cache, retry_on_fail

logger = logging.getLogger(__name__)

# ==================== 情绪词典 ====================
POSITIVE_KW = [
    "利好", "大涨", "涨停", "突破", "增长", "预增", "扭亏", "盈利",
    "中标", "签约", "订单", "扩产", "投产", "量产", "研发成功",
    "获批", "专利", "创新高", "增持", "回购", "分红",
    "政策支持", "补贴", "扶持", "行业景气", "供不应求", "涨价",
    "机构看好", "上调评级", "买入评级", "目标价", "超预期",
    "产能释放", "需求旺盛", "量价齐升", "业绩提升",
]

NEGATIVE_KW = [
    "利空", "大跌", "跌停", "亏损", "预亏", "下降", "下滑", "减少",
    "减持", "质押", "冻结", "诉讼", "处罚", "罚款", "调查",
    "停产", "限产", "停工", "裁员", "退市", "ST",
    "违约", "逾期", "债务", "资金链", "流动性",
    "低迷", "供过于求", "降价", "价格战",
    "看空", "下调评级", "卖出评级", "低于预期",
    "产能过剩", "需求疲软", "业绩下滑", "商誉减值", "资产减值", "暴雷",
]


def analyze_sentiment(text: str) -> dict:
    """关键词情绪打分"""
    if not isinstance(text, str) or not text:
        return {"score": 0.0, "label": "neutral", "pos": 0, "neg": 0}

    pos = sum(1 for kw in POSITIVE_KW if kw in text)
    neg = sum(1 for kw in NEGATIVE_KW if kw in text)
    total = pos + neg
    if total == 0:
        score = 0.0
    else:
        score = (pos - neg) / (total + 1) * (total / (total + 3))

    label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
    return {"score": round(score, 4), "label": label, "pos": pos, "neg": neg}


@retry_on_fail
def get_stock_news(symbol: str, limit: int = 30) -> pd.DataFrame:
    """获取个股新闻"""
    try:
        df = ak.stock_news_em(stock=symbol)
        if df is not None and not df.empty:
            col_map = {}
            for c in df.columns:
                if "标题" in c: col_map[c] = "title"
                elif "时间" in c: col_map[c] = "datetime"
                elif "内容" in c: col_map[c] = "content"
            df = df.rename(columns=col_map)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df["symbol"] = symbol
            logger.info(f"{symbol} 新闻: {len(df)}条")
            return df.head(limit)
    except Exception as e:
        logger.warning(f"{symbol} 新闻失败: {e}")
    return pd.DataFrame()


def get_all_news(use_cache: bool = True) -> dict:
    """获取所有新闻情绪"""
    result = {"stock_news": {}, "sentiment": {}}
    ttl = CACHE_TTL.get("news_data", 1800)

    for code in STOCK_CODES:
        key = f"stock_news_{code}"
        if use_cache:
            df = cache.get(key, ttl)
            if df is not None:
                result["stock_news"][code] = df
            else:
                df = get_stock_news(code)
                if not df.empty:
                    cache.set(key, df)
        else:
            df = get_stock_news(code)

        result["stock_news"][code] = df
        if not df.empty and "title" in df.columns:
            sentiments = []
            for _, row in df.iterrows():
                text = str(row.get("title", "")) + " " + str(row.get("content", ""))
                s = analyze_sentiment(text)
                s["title"] = str(row.get("title", ""))[:50]
                sentiments.append(s)
            result["sentiment"][code] = sentiments

    return result
