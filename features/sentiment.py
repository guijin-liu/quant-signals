"""
新闻情绪特征
- 个股新闻情绪评分
- 市场整体情绪
- 情绪变化趋势
"""

import logging
import numpy as np
import pandas as pd
from config import STOCK_CODES

logger = logging.getLogger(__name__)


def extract_sentiment_features(news_sentiment: dict, stock_code: str) -> dict:
    """
    从新闻情绪数据提取特征
    news_sentiment: {stock_code: [sentiment_dicts]}
    """
    feats = {
        "sentiment_score": 0.0,
        "sentiment_weighted": 0.0,
        "sentiment_signal": "neutral",
        "sentiment_pos_ratio": 0.0,
        "sentiment_neg_ratio": 0.0,
        "sentiment_news_count": 0,
        "sentiment_recent_score": 0.0,      # 最近3条
        "sentiment_trend": 0,                # -1=恶化, 0=不变, 1=改善
    }

    sentiments = news_sentiment.get(stock_code, [])
    if not sentiments:
        return feats

    scores = [s.get("score", 0) for s in sentiments]
    labels = [s.get("label", "neutral") for s in sentiments]

    feats["sentiment_score"] = round(np.mean(scores), 4) if scores else 0.0
    feats["sentiment_news_count"] = len(scores)

    # 加权分数（越新权重越高）
    if len(scores) > 1:
        weights = np.arange(1, len(scores) + 1)
        feats["sentiment_weighted"] = round(np.average(scores, weights=weights), 4)
    else:
        feats["sentiment_weighted"] = feats["sentiment_score"]

    # 近期3条
    recent = scores[-3:]
    feats["sentiment_recent_score"] = round(np.mean(recent), 4) if recent else 0.0

    # 正负比例
    total = len(labels)
    feats["sentiment_pos_ratio"] = round(sum(1 for l in labels if l == "positive") / total, 3)
    feats["sentiment_neg_ratio"] = round(sum(1 for l in labels if l == "negative") / total, 3)

    # 趋势
    if len(scores) >= 4:
        earlier = np.mean(scores[:len(scores)//2])
        later = np.mean(scores[len(scores)//2:])
        if later - earlier > 0.1:
            feats["sentiment_trend"] = 1
        elif later - earlier < -0.1:
            feats["sentiment_trend"] = -1

    # 综合信号
    ws = feats["sentiment_weighted"]
    if ws > 0.15:
        feats["sentiment_signal"] = "bullish"
    elif ws < -0.15:
        feats["sentiment_signal"] = "bearish"
    else:
        feats["sentiment_signal"] = "neutral"

    return feats


def compute_market_sentiment(all_stock_sentiments: dict) -> dict:
    """
    汇总全市场情绪
    """
    all_scores = []
    for code, sentiments in all_stock_sentiments.items():
        for s in sentiments:
            all_scores.append(s.get("score", 0))

    if not all_scores:
        return {"market_sentiment": 0.0, "market_sentiment_signal": "neutral"}

    avg = np.mean(all_scores)
    signal = "bullish" if avg > 0.1 else ("bearish" if avg < -0.1 else "neutral")

    return {
        "market_sentiment": round(float(avg), 4),
        "market_sentiment_signal": signal,
    }
