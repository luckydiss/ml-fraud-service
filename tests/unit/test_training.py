"""
Тесты для сервиса обучения (FraudModelTrainer).
"""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.training.trainer import FraudModelTrainer


@pytest.fixture
def sample_training_data():
    """Минимальный датасет для тестирования обучения."""
    data = {
        "amt": [100.0, 200.0, 50.0, 1000.0, 150.0] * 20,
        "lat": [40.7] * 100,
        "long": [-74.0] * 100,
        "city_pop": [1000000] * 100,
        "merch_lat": [40.8] * 100,
        "merch_long": [-73.9] * 100,
        "merchant": ["shop_A", "shop_B"] * 50,
        "category": ["grocery", "electronics"] * 50,
        "gender": ["M", "F"] * 50,
        "job": ["Engineer", "Doctor"] * 50,
        "trans_date_trans_time": ["2024-01-15 14:30:00"] * 100,
        "dob": ["1985-06-20"] * 100,
        "is_fraud": [0, 0, 0, 1, 0] * 20,  # 20% фрода
    }
    return pd.DataFrame(data)


class TestFraudModelTrainer:
    """Тесты для FraudModelTrainer."""

    def test_evaluate_without_training_raises_error(self):
        """Вызов evaluate без обучения выбрасывает ValueError."""
        trainer = FraudModelTrainer()
        
        with pytest.raises(ValueError, match="Модель не обучена"):
            trainer.evaluate(pd.DataFrame(), pd.Series())

    def test_train_and_evaluate_full_cycle(self, sample_training_data):
        """Полный цикл обучения и оценки работает корректно."""
        trainer = FraudModelTrainer()
        
        pipeline, metrics = trainer.train_and_evaluate(sample_training_data)
        
        assert isinstance(pipeline, Pipeline)
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
