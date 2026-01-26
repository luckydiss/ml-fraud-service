from typing import Any, Dict, Optional, Tuple

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.common import get_logger, settings
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    create_full_pipeline,
    prepare_training_data,
)

logger = get_logger(__name__)


class FraudModelTrainer:

    def __init__(
        self,
        lgbm_params: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ):

        self.lgbm_params = lgbm_params or settings.get_lgbm_params()
        self.threshold = threshold or settings.optimal_threshold
        self.pipeline: Optional[Pipeline] = None
        self.metrics: Optional[Dict[str, float]] = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Pipeline:
        """Обучение пайплайна модели."""
        logger.info(f"Запуск обучения. Размер данных: {X_train.shape}")

        model = LGBMClassifier(**self.lgbm_params)

        self.pipeline = create_full_pipeline(
            model=model,
            numerical_features=NUMERICAL_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )

        self.pipeline.fit(X_train, y_train)

        logger.info("Обучение завершено")
        return self.pipeline

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Оценка обученного пайплайна."""
        if self.pipeline is None:
            raise ValueError("Модель не обучена")

        logger.info("Оценка модели")

        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "threshold": self.threshold,
        }

        logger.info(f"Метрики: {self.metrics}")
        return self.metrics

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Tuple[Pipeline, Dict[str, float]]:
        logger.info("Запуск процесса обучения и оценки")


        X_train, X_test, y_train, y_test = prepare_training_data(df, test_size)

        self.train(X_train, y_train)
        metrics = self.evaluate(X_test, y_test)

        return self.pipeline, metrics

    def get_pipeline(self) -> Pipeline:
        if self.pipeline is None:
            raise ValueError("Модель не обучена")
        return self.pipeline

    def get_metrics(self) -> Dict[str, float]:
        if self.metrics is None:
            raise ValueError("Оценка не проводилась")
        return self.metrics
