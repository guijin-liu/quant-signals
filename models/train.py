"""
模型训练模块
- 梯度提升三分类（大涨/小涨/跌）
- 滚动训练避免过拟合
- 概率校准
- 特征重要性分析
"""

import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from config import (
    TRAIN_PARAMS, SAVED_MODELS_DIR, STOCK_CODES, STOCK_POOL,
)

logger = logging.getLogger(__name__)


def _create_model(params: dict = None):
    """
    创建模型（优先XGBoost，不可用则用sklearn GradientBoosting）
    """
    if params is None:
        params = TRAIN_PARAMS["xgboost_params"]

    # 尝试XGBoost
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=params.get("random_state", 42),
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 3),
            n_jobs=params.get("n_jobs", -1),
        )
        logger.info("使用 XGBoost 模型")
        return model, "xgboost"
    except ImportError:
        pass

    # 回退到 sklearn GradientBoostingClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        min_samples_split=params.get("min_child_weight", 3) * 2,
        min_samples_leaf=params.get("min_child_weight", 3),
        max_features=params.get("colsample_bytree", 0.8),
        random_state=params.get("random_state", 42),
        validation_fraction=0.1,
        n_iter_no_change=30,
        tol=1e-4,
    )
    logger.info("使用 sklearn GradientBoostingClassifier 模型（XGBoost不可用）")
    return model, "sklearn_gb"


def train_model(X_train, y_train, X_test=None, y_test=None, params: dict = None):
    """
    训练三分类模型（自动选择XGBoost或sklearn）
    """
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

    if params is None:
        params = TRAIN_PARAMS["xgboost_params"]

    model, model_type = _create_model(params)
    logger.info(f"开始训练模型 ({model_type})...")

    # 训练
    model.fit(X_train, y_train)

    # 评估
    results = {"model_type": model_type}
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        results["accuracy"] = accuracy_score(y_test, y_pred)
        results["classification_report"] = classification_report(y_test, y_pred, target_names=["跌", "平", "涨"])
        results["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        results["predictions"] = y_pred
        results["probabilities"] = y_proba

        logger.info(f"测试准确率: {results['accuracy']:.4f}")
        logger.info(f"\n{results['classification_report']}")

    # 特征重要性
    importance = model.feature_importances_
    feature_names = X_train.columns.tolist()

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    results["feature_importance"] = imp_df

    logger.info(f"\nTop 10 重要特征:")
    for _, row in imp_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.6f}")

    return model, results


def train_with_rolling_window(
    dataset: dict,
    window_days: int = 180,
    retrain_days: int = 20,
) -> dict:
    """
    滚动窗口训练（模拟实盘持续更新）
    """
    df = dataset["features"].copy()
    feature_cols = dataset["feature_cols"]

    if df.empty:
        return {}

    df = df.dropna(subset=feature_cols + ["target_3class"])
    df = df.sort_values("datetime")

    dates = df["datetime"].unique()
    if len(dates) < window_days:
        logger.warning("数据不足，无法滚动训练")
        return {}

    models = []
    performances = []

    for i in range(window_days, len(dates), retrain_days):
        train_end = dates[i]
        test_start = dates[i]
        test_end = dates[min(i + retrain_days, len(dates) - 1)]

        train_data = df[df["datetime"] <= train_end]
        test_data = df[(df["datetime"] >= test_start) & (df["datetime"] <= test_end)]

        if train_data.empty or test_data.empty:
            continue

        X_train = train_data[feature_cols].select_dtypes(include=[np.number])
        y_train = train_data["target_3class"]
        X_test = test_data[feature_cols].select_dtypes(include=[np.number])
        y_test = test_data["target_3class"]

        model, results = train_model(X_train, y_train, X_test, y_test)
        models.append({
            "train_end": train_end,
            "model": model,
            "accuracy": results.get("accuracy", 0),
        })
        performances.append({
            "train_end": train_end,
            "accuracy": results.get("accuracy", 0),
        })

    # 保存最新模型
    if models:
        best = max(models, key=lambda x: x["accuracy"])
        save_model(best["model"], "xgboost_latest")
        logger.info(f"最佳模型 准确率: {best['accuracy']:.4f} (训练截止: {best['train_end']})")

    return {
        "models": models,
        "performances": pd.DataFrame(performances),
    }


def save_model(model, name: str) -> str:
    """保存模型到文件"""
    path = SAVED_MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    logger.info(f"模型已保存: {path}")
    return str(path)


def load_model(name: str = "xgboost_latest"):
    """加载模型"""
    path = SAVED_MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        logger.warning(f"模型文件不存在: {path}")
        return None
    return joblib.load(path)


def calibrate_probability(model, X_cal, y_cal):
    """
    Platt Scaling 概率校准
    使模型输出的概率更接近真实概率
    """
    from sklearn.calibration import CalibratedClassifierCV

    calibrator = CalibratedClassifierCV(
        model, method="sigmoid", cv=3
    )
    calibrator.fit(X_cal, y_cal)
    logger.info("概率校准完成 (Platt Scaling)")
    return calibrator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== 模型训练测试 ===")
    # 生成模拟数据测试
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.choice([0, 1, 2], size=1000, p=[0.3, 0.3, 0.4])
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    model, results = train_model(
        pd.DataFrame(X_train, columns=[f"f{i}" for i in range(20)]),
        y_train,
        pd.DataFrame(X_test, columns=[f"f{i}" for i in range(20)]),
        y_test,
    )
    print(f"准确率: {results['accuracy']:.4f}")
