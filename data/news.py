"""
新闻数据获取与情绪分析
- A股个股相关新闻
- 财经快讯
- 情绪打分（利好/利空/中性）
"""

import logging
import re
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from config import STOCK_CODES, STOCK_POOL, CACHE_TTL
from data.fetcher import cache, retry_on_fail

logger = logging.getLogger(__name__)

# 尝试导入 SnowNLP 做情感分析
try:
    from snownlp import SnowNLP
    HAS_SNOWNLP = True
except ImportError:
    HAS_SNOWNLP = False
    logger.warning("SnowNLP 未安装，使用关键词规则情绪打分")


# ==================== 情绪词典 ====================
POSITIVE_KEYWORDS = [
    "利好", "大涨", "涨停", "突破", "增长", "预增", "扭亏", "盈利",
    "中标", "签约", "订单", "扩产", "投产", "量产", "研发成功",
    "获批复", "获批", "专利", "创新高", "增持", "回购", "分红",
    "政策支持", "补贴", "扶持", "行业景气", "供不应求", "涨价",
    "机构看好", "上调评级", "买入评级", "目标价", "超预期",
    "产能释放", "需求旺盛", "量价齐升", "业绩提升",
]

NEGATIVE_KEYWORDS = [
    "利空", "大跌", "跌停", "亏损", "预亏", "下降", "下滑", "减少",
    "减持", "质押", "冻结", "诉讼", "处罚", "罚款", "调查",
    "停产", "限产", "停工", "裁员", "退市", "ST",
    "违约", "逾期", "债务危机", "流动性危机", "资金链断裂",
    "行业低迷", "供过于求", "降价", "价格战", "竞争加剧",
    "机构看空", "下调评级", "卖出评级", "低于预期",
    "产能过剩", "需求疲软", "订单减少", "业绩下滑",
    "商誉减值", "资产减值", "计提", "暴雷",
]


def _sentiment_by_keywords(text: str) -> dict:
    """基于关键词的情绪打分"""
    if not isinstance(text, str) or not text:
        return {"score": 0.0, "label": "neutral", "positive_hits": 0, "negative_hits": 0}

    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)

    # 加权：连续出现的关键词增加权重
    total = pos_count + neg_count
    if total == 0:
        score = 0.0
        label = "neutral"
    else:
        # 归一化到 [-1, 1]
        raw = (pos_count - neg_count) / (total + 1)
        score = round(raw * (total / (total + 3)), 4)  # 阻尼：信息量小时分数接近0
        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"

    return {
        "score": score,
        "label": label,
        "positive_hits": pos_count,
        "negative_hits": neg_count,
    }


def _sentiment_by_snownlp(text: str) -> dict:
    """基于SnowNLP的情绪打分"""
    if not isinstance(text, str) or not text:
        return {"score": 0.0, "label": "neutral", "method": "snownlp"}

    try:
        s = SnowNLP(text)
        score = s.sentiments  # 0~1，越大越正面
        # 映射到 [-1, 1]
        adjusted = (score - 0.5) * 2
        label = "positive" if score > 0.6 else ("negative" if score < 0.4 else "neutral")
        return {"score": round(adjusted, 4), "label": label, "raw_score": round(score, 4), "method": "snownlp"}
    except Exception:
        return _sentiment_by_keywords(text)


def analyze_sentiment(text: str) -> dict:
    """统一情绪分析入口"""
    if HAS_SNOWNLP and len(text) > 10:
        return _sentiment_by_snownlp(text)
    return _sentiment_by_keywords(text)


@retry_on_fail
def get_stock_news(symbol: str, limit: int = 50) -> pd.DataFrame:
    """获取个股相关新闻"""
    try:
        df = ak.stock_news_em(symbol=symbol)
        if df is not None and not df.empty:
            # 列名可能有差异
            if "新闻标题" in df.columns:
                df.rename(columns={"新闻标题": "title", "发布时间": "datetime", "新闻内容": "content"}, inplace=True)
            elif "标题" in df.columns:
                df.rename(columns={"标题": "title", "时间": "datetime", "内容": "content"}, inplace=True)

            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
            df["symbol"] = symbol
            df = df.head(limit)
            return df
    except Exception as e:
        logger.warning(f"获取 {symbol} 新闻失败: {e}")
    return pd.DataFrame()


@retry_on_fail
def get_financial_news(limit: int = 100) -> pd.DataFrame:
    """获取财经快讯（全市场）"""
    try:
        df = ak.stock_info_global_em()
        if df is not None and not df.empty:
            if "title" in df.columns or "标题" in df.columns:
                pass
            return df.head(limit)
    except Exception:
        pass

    try:
        df = ak.stock_news_main_cx()  # 财新新闻
        if df is not None and not df.empty:
            return df.head(limit)
    except Exception as e:
        logger.warning(f"获取财经新闻失败: {e}")

    return pd.DataFrame()


def get_all_news(use_cache: bool = True) -> dict:
    """获取所有相关新闻并做情绪分析"""
    result = {"stock_news": {}, "market_news": pd.DataFrame(), "sentiment": {}}

    ttl = CACHE_TTL["news_data"]

    # 个股新闻
    for code in STOCK_CODES:
        cache_key = f"stock_news_{code}"
        if use_cache:
            df = cache.get(cache_key, ttl)
            if df is not None:
                result["stock_news"][code] = df
                # 做情绪分析
                if not df.empty and "title" in df.columns:
                    sentiments = []
                    for _, row in df.iterrows():
                        text = str(row.get("title", "")) + " " + str(row.get("content", ""))
                        s = analyze_sentiment(text)
                        s["title"] = row.get("title", "")
                        sentiments.append(s)
                    result["sentiment"][code] = sentiments
                continue

        df = get_stock_news(code)
        if not df.empty:
            if use_cache:
                cache.set(cache_key, df)
            result["stock_news"][code] = df
            # 情绪分析
            sentiments = []
            for _, row in df.iterrows():
                text = str(row.get("title", "")) + " " + str(row.get("content", ""))
                s = analyze_sentiment(text)
                s["title"] = row.get("title", "")
                sentiments.append(s)
            result["sentiment"][code] = sentiments
            logger.info(f"获取 {STOCK_POOL[code]['name']} 新闻: {len(df)} 条")

    return result


def compute_news_sentiment_score(sentiments: list) -> dict:
    """
    汇总一只股票的新闻情绪总分
    返回综合评分和信号
    """
    if not sentiments:
        return {"avg_score": 0.0, "label": "neutral", "signal": "hold"}

    scores = [s["score"] for s in sentiments]
    avg_score = sum(scores) / len(scores)

    # 考虑信息衰减：越新的新闻权重越高
    if len(scores) > 1:
        weighted = sum(s * (i + 1) for i, s in enumerate(scores))
        weighted /= sum(range(1, len(scores) + 1))
    else:
        weighted = avg_score

    if avg_score > 0.2:
        signal = "bullish"
    elif avg_score < -0.2:
        signal = "bearish"
    else:
        signal = "neutral"

    return {
        "avg_score": round(avg_score, 4),
        "weighted_score": round(weighted, 4),
        "label": signal,
        "positive_ratio": round(sum(1 for s in scores if s["label"] == "positive") / len(scores), 3),
        "negative_ratio": round(sum(1 for s in scores if s["label"] == "negative") / len(scores), 3),
        "news_count": len(scores),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== 测试新闻情绪分析 ===")
    # 测试情绪打分
    test_texts = [
        "公司业绩大幅增长，超出市场预期，机构给予买入评级",
        "公司面临巨额亏损，控股股东减持股份，股价承压",
        "今日大盘窄幅震荡，成交清淡",
    ]
    for t in test_texts:
        result = analyze_sentiment(t)
        print(f"文本: {t[:40]}...")
        print(f"  情绪: {result}")
