"""
信号生成模块
- ML模型预测 + 多因子规则打分
- 多维共振过滤
- 输出：买入/卖出/持有 + 置信度
"""

import logging
import numpy as np
import pandas as pd
from config import (
    STOCK_CODES, STOCK_POOL, SIGNAL_PARAMS, PRIMARY_TF,
)

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    交易信号生成器
    融合ML预测和多因子规则打分
    """

    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler
        self.ml_weight = SIGNAL_PARAMS["ml_weight"]
        self.factor_weight = SIGNAL_PARAMS["factor_weight"]
        self.buy_threshold = SIGNAL_PARAMS["buy_threshold"]
        self.sell_threshold = SIGNAL_PARAMS["sell_threshold"]
        self.min_confidence = SIGNAL_PARAMS["min_confidence"]
        self.resonance_required = SIGNAL_PARAMS["resonance_required"]

    def predict_ml(self, X: pd.DataFrame) -> np.ndarray:
        """ML模型预测概率"""
        if self.model is None:
            logger.warning("模型未加载，ML预测不可用")
            return np.full((len(X), 3), 0.33)

        try:
            proba = self.model.predict_proba(X)
            return proba
        except Exception as e:
            logger.error(f"ML预测失败: {e}")
            return np.full((len(X), 3), 0.33)

    def score_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        多因子规则打分 (0~1)
        每个维度产生一个0~1分，最后加权平均
        """
        scores = pd.DataFrame(index=df.index)

        # === 1. 技术维度 (权重 0.30) ===
        tech_score = 0.5  # 基础分

        # 均线多头排列
        if "ma_bullish" in df.columns:
            tech_score += df["ma_bullish"] * 0.15

        # MACD金叉或柱状图由负转正
        if "macd_golden_cross" in df.columns and "macd_hist_sign_change" in df.columns:
            tech_score += (df["macd_golden_cross"] | df["macd_hist_sign_change"]) * 0.10

        # RSI适中（不超买不超卖，且在上升）
        if "rsi" in df.columns:
            rsi_ok = ((df["rsi"] > 40) & (df["rsi"] < 70)).astype(float) * 0.05
            rsi_rising = (df["rsi_rising"] == 1).astype(float) * 0.05
            tech_score += rsi_ok + rsi_rising

        # 布林带中轨上方
        if "boll_pct_b" in df.columns:
            tech_score += ((df["boll_pct_b"] > 0.4) & (df["boll_pct_b"] < 0.9)).astype(float) * 0.05

        # 量价配合
        if "vol_price_up" in df.columns:
            tech_score += df["vol_price_up"] * 0.05
        if "volume_surge" in df.columns:
            tech_score += df["volume_surge"] * 0.05

        scores["technical"] = tech_score.clip(0, 1)

        # === 2. 大盘环境维度 (权重 0.20) ===
        market_score = 0.5
        if "sh_pct_change" in df.columns:
            market_score += ((df["sh_pct_change"] > 0).astype(float) - 0.5) * 0.3
            # 大盘涨且个股强于大盘
            market_score += ((df["pct_change"] > df["sh_pct_change"]) & (df["sh_pct_change"] > 0)).astype(float) * 0.2
        scores["market"] = market_score.clip(0, 1)

        # === 3. 美股映射维度 (权重 0.10) ===
        us_score = 0.5
        if "us_weighted_return" in df.columns:
            us_score += (df["us_weighted_return"] * 10).clip(-0.4, 0.4)
        if "us_overnight_signal" in df.columns:
            us_bull = (df["us_overnight_signal"] == "bullish").astype(float)
            us_score += us_bull * 0.2
        scores["us_mapping"] = us_score.clip(0, 1)

        # === 4. 板块维度 (权重 0.15) ===
        sector_score = 0.5
        if "sector_strength_score" in df.columns:
            sector_score = df["sector_strength_score"]
        if "sector_is_leading" in df.columns:
            sector_score += df["sector_is_leading"] * 0.2
        scores["sector"] = sector_score.clip(0, 1)

        # === 5. 资金流向维度 (权重 0.15) ===
        flow_score = 0.5
        if "flow_capital_score" in df.columns:
            flow_score += df["flow_capital_score"] * 0.5
        if "flow_stock_flow_trend" in df.columns:
            flow_score += (df["flow_stock_flow_trend"] > 0).astype(float) * 0.3
        scores["money_flow"] = flow_score.clip(0, 1)

        # === 6. 新闻情绪维度 (权重 0.10) ===
        news_score = 0.5
        if "news_sentiment_score" in df.columns:
            news_score += df["news_sentiment_score"] * 0.5
        if "news_sentiment_signal" in df.columns:
            news_bull = (df["news_sentiment_signal"] == "bullish").astype(float)
            news_score += news_bull * 0.3
        scores["news"] = news_score.clip(0, 1)

        # === 综合打分 ===
        weights = {
            "technical": 0.30,
            "market": 0.20,
            "us_mapping": 0.10,
            "sector": 0.15,
            "money_flow": 0.15,
            "news": 0.10,
        }
        scores["factor_total"] = sum(scores[c] * w for c, w in weights.items())
        scores["factor_total"] = scores["factor_total"].clip(0, 1)

        return scores

    def count_resonance(self, factor_scores: pd.DataFrame) -> pd.Series:
        """统计共振维度数（分数>0.6的维度数）"""
        resonance_cols = ["technical", "market", "us_mapping", "sector", "money_flow", "news"]
        return (factor_scores[resonance_cols] > 0.6).sum(axis=1)

    def generate_signals(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        生成最终交易信号
        融合 ML + 多因子，多维共振过滤
        """
        result = df.copy()
        result["signal"] = "HOLD"
        result["signal_score"] = 0.0
        result["confidence"] = 0.0
        result["resonance_count"] = 0
        result["ml_buy_prob"] = 0.0
        result["ml_probs"] = None

        if result.empty:
            return result

        # 因子打分
        factor_scores = self.score_factors(result)
        resonance = self.count_resonance(factor_scores)

        # ML预测
        X_numeric = result[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        try:
            ml_proba = self.predict_ml(X_numeric)
            ml_buy_prob = ml_proba[:, 2]  # class=2 (大涨) 的概率
            ml_sell_prob = ml_proba[:, 0]  # class=0 (跌) 的概率
        except Exception:
            ml_buy_prob = np.zeros(len(result))
            ml_sell_prob = np.zeros(len(result))

        # === 综合评分 ===
        composite = (
            self.ml_weight * ml_buy_prob +
            self.factor_weight * factor_scores["factor_total"]
        )

        # ML置信度 (概率偏离均匀分布的程度)
        ml_confidence = np.max(ml_proba, axis=1) if ml_proba is not None else np.zeros(len(result))

        # === 信号判定 ===
        for i in range(len(result)):
            score = composite.iloc[i] if hasattr(composite, 'iloc') else composite[i]
            resonance_n = resonance.iloc[i] if hasattr(resonance, 'iloc') else resonance[i]
            ml_conf = ml_confidence[i]
            f_score = factor_scores["factor_total"].iloc[i] if hasattr(factor_scores["factor_total"], 'iloc') else factor_scores["factor_total"][i]

            # 买入条件：ML概率高 + 多因子分数高 + 足够维度共振
            buy_condition = (
                score >= self.buy_threshold and
                ml_conf >= self.min_confidence and
                resonance_n >= self.resonance_required and
                f_score > 0.6
            )

            # 卖出条件
            sell_condition = (
                score <= self.sell_threshold or
                (ml_sell_prob[i] > 0.5 and resonance_n < 2)
            )

            if buy_condition:
                result.at[result.index[i], "signal"] = "BUY"
            elif sell_condition:
                result.at[result.index[i], "signal"] = "SELL"
            else:
                result.at[result.index[i], "signal"] = "HOLD"

            result.at[result.index[i], "signal_score"] = round(float(score), 4)
            result.at[result.index[i], "confidence"] = round(float(ml_conf), 4)
            result.at[result.index[i], "resonance_count"] = int(resonance_n)
            result.at[result.index[i], "ml_buy_prob"] = round(float(ml_buy_prob[i]), 4)

        # 信号统计
        buy_count = (result["signal"] == "BUY").sum()
        sell_count = (result["signal"] == "SELL").sum()
        hold_count = (result["signal"] == "HOLD").sum()
        logger.info(f"信号统计: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count} (共{len(result)}条)")

        return result

    def get_latest_signal(self, df: pd.DataFrame, feature_cols: list) -> dict:
        """
        获取最新一条数据的交易信号
        用于实盘/模拟盘盘中决策
        """
        if df.empty:
            return {"signal": "HOLD", "reason": "无数据"}

        signals = self.generate_signals(df.tail(50), feature_cols)
        latest = signals.iloc[-1]

        return {
            "datetime": str(latest.get("datetime", "")),
            "symbol": latest.get("symbol", ""),
            "stock_name": latest.get("stock_name", ""),
            "signal": latest["signal"],
            "score": latest["signal_score"],
            "confidence": latest["confidence"],
            "resonance": latest["resonance_count"],
            "close": latest.get("close", 0),
        }


def create_signal_generator(model_path: str = None):
    """工厂函数：创建信号生成器"""
    from models.train import load_model

    model = None
    if model_path:
        model = load_model(model_path)
    else:
        model = load_model("xgboost_latest")

    return SignalGenerator(model=model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== 信号生成测试 ===")
    # 用模拟数据测试
    np.random.seed(42)
    dates = pd.date_range("2025-01-01 09:30", periods=100, freq="15min")
    test_df = pd.DataFrame({
        "datetime": dates,
        "close": np.cumsum(np.random.randn(100) * 0.1) + 20,
        "open": np.zeros(100),
        "high": np.zeros(100),
        "low": np.zeros(100),
        "volume": np.random.randint(10000, 100000, 100),
        "symbol": "000933",
        "stock_name": "神火股份",
    })
    # 添加模拟特征
    for feat in ["ma_bullish", "macd_golden_cross", "macd_hist_sign_change",
                  "rsi", "rsi_rising", "boll_pct_b", "vol_price_up",
                  "volume_surge", "sh_pct_change", "pct_change",
                  "us_weighted_return", "us_overnight_signal",
                  "sector_strength_score", "sector_is_leading",
                  "flow_capital_score", "flow_stock_flow_trend",
                  "news_sentiment_score", "news_sentiment_signal"]:
        test_df[feat] = np.random.random(100)

    gen = SignalGenerator()
    signals = gen.generate_signals(test_df, [f for f in test_df.columns if f not in ["datetime", "symbol", "stock_name"]])
    print(signals["signal"].value_counts())
    print(f"\n最近5条信号:")
    print(signals[["datetime", "signal", "signal_score", "resonance_count"]].tail())
