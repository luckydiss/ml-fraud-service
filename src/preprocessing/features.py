import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.common import get_logger

logger = get_logger(__name__)


def haversine_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Рассчитывает расстояние Хаверсина между двумя точками."""
    R = 6371

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Извлекает признаки из timestamp транзакции."""

    def __init__(self, timestamp_col: str = "trans_date_trans_time"):
        self.timestamp_col = timestamp_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.debug("Извлечение временных признаков")

        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if self.timestamp_col in X.columns:
            ts = pd.to_datetime(X[self.timestamp_col])
            X["hour"] = ts.dt.hour
            X["day_of_week"] = ts.dt.dayofweek
            X["day_of_month"] = ts.dt.day
            X["month"] = ts.dt.month

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return list(input_features) + ["hour", "day_of_week", "day_of_month", "month"]


class DistanceCalculator(BaseEstimator, TransformerMixin):
    """Рассчитывает расстояние между клиентом и продавцом."""

    def __init__(
        self,
        lat_col: str = "lat",
        long_col: str = "long",
        merch_lat_col: str = "merch_lat",
        merch_long_col: str = "merch_long",
    ):
        self.lat_col = lat_col
        self.long_col = long_col
        self.merch_lat_col = merch_lat_col
        self.merch_long_col = merch_long_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.debug("Расчет расстояния между клиентом и продавцом")

        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if all(
            col in X.columns
            for col in [self.lat_col, self.long_col, self.merch_lat_col, self.merch_long_col]
        ):
            X["distance"] = haversine_distance(
                X[self.lat_col].values,
                X[self.long_col].values,
                X[self.merch_lat_col].values,
                X[self.merch_long_col].values,
            )

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return list(input_features) + ["distance"]


class AgeCalculator(BaseEstimator, TransformerMixin):
    """Рассчитывает возраст клиента на момент транзакции."""

    def __init__(
        self,
        dob_col: str = "dob",
        timestamp_col: str = "trans_date_trans_time",
    ):
        self.dob_col = dob_col
        self.timestamp_col = timestamp_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.debug("Расчет возраста клиента")

        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if self.dob_col in X.columns and self.timestamp_col in X.columns:
            dob = pd.to_datetime(X[self.dob_col])
            trans_time = pd.to_datetime(X[self.timestamp_col])
            X["age"] = ((trans_time - dob).dt.days / 365.25).astype(int)

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return list(input_features) + ["age"]


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Выбирает определенные столбцы для дальнейшей обработки."""

    def __init__(self, columns: list):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return X[self.columns]

    def get_feature_names_out(self, input_features=None):
        return self.columns


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Комбинирует трансформеры для создания всех признаков."""

    def __init__(self):
        self.time_extractor = TimeFeatureExtractor()
        self.distance_calculator = DistanceCalculator()
        self.age_calculator = AgeCalculator()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Применение feature engineering")

        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        X = self.time_extractor.transform(X)
        X = self.distance_calculator.transform(X)
        X = self.age_calculator.transform(X)

        logger.info(f"Feature engineering завершен. Размер: {X.shape}")
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        features = list(input_features)
        features.extend(["hour", "day_of_week", "day_of_month", "month", "distance", "age"])
        return features


NUMERICAL_FEATURES = [
    "amt",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "distance",
    "age",
]

CATEGORICAL_FEATURES = ["merchant", "category", "gender", "job"]

RAW_INPUT_COLUMNS = [
    "amt",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
    "merchant",
    "category",
    "gender",
    "job",
    "trans_date_trans_time",
    "dob",
]
