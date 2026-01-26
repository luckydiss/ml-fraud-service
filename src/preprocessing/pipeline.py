from typing import List, Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common import get_logger

from .features import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ColumnSelector,
    FeatureEngineeringTransformer,
)

logger = get_logger(__name__)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Частотное кодирование категориальных признаков."""

    def __init__(self):
        self.freq_maps_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        for col in X_df.columns:
            freq_map = X_df[col].value_counts(normalize=True).to_dict()
            self.freq_maps_[col] = freq_map

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        result = pd.DataFrame(index=X_df.index)

        for col in X_df.columns:
            freq_map = self.freq_maps_.get(col, {})
            result[f"{col}_freq"] = X_df[col].map(freq_map).fillna(0)

        return result.values

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return [f"{col}_freq" for col in input_features]
        return []


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """Преобразует входные данные в DataFrame."""

    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.columns)

    def get_feature_names_out(self, input_features=None):
        return self.columns


def create_preprocessing_transformer(
    numerical_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Создает трансформер для препроцессинга столбцов."""
    categorical_high = ["merchant", "job"]
    categorical_low = ["category", "gender"]

    categorical_high_filtered = [
        col for col in categorical_high if col in categorical_features
    ]
    categorical_low_filtered = [
        col for col in categorical_low if col in categorical_features
    ]

    numeric_pipeline = Pipeline([("scaler", StandardScaler(with_mean=False))])

    categorical_low_pipeline = Pipeline(
        [("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    categorical_high_pipeline = Pipeline([("freq_encoder", FrequencyEncoder())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat_low", categorical_low_pipeline, categorical_low_filtered),
            ("cat_high", categorical_high_pipeline, categorical_high_filtered),
        ],
        remainder="drop",
    )

    return preprocessor


def create_full_pipeline(
    model,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> Pipeline:
    """Создает полный ML-пайплайн (признаки + предобработка + модель)."""
    numerical_features = numerical_features or NUMERICAL_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    all_features = numerical_features + categorical_features

    pipeline = Pipeline([
        ("feature_engineering", FeatureEngineeringTransformer()),
        ("select_features", ColumnSelector(all_features)),
        ("preprocessing", create_preprocessing_transformer(
            numerical_features, categorical_features
        )),
        ("model", model),
    ])

    logger.info("Полный пайплайн создан")
    return pipeline


def prepare_training_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Подготавливает данные для обучения с time-split разделением."""
    logger.info("Подготовка тренировочных данных")

    df_sorted = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    split_index = int(len(df_sorted) * (1 - test_size))

    train_df = df_sorted.iloc[:split_index].copy()
    test_df = df_sorted.iloc[split_index:].copy()

    from .features import RAW_INPUT_COLUMNS

    X_train = train_df[RAW_INPUT_COLUMNS]
    y_train = train_df["is_fraud"]
    X_test = test_df[RAW_INPUT_COLUMNS]
    y_test = test_df["is_fraud"]

    logger.info(f"Размер выборки: train={len(X_train)}, test={len(X_test)}")

    return X_train, X_test, y_train, y_test
