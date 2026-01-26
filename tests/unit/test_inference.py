import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.inference.predictor import FraudPredictor
from src.registry import ModelRegistry
from src.training.trainer import FraudModelTrainer


@pytest.fixture
def real_registry_setup():
    """Создает временный реестр с обученными моделями."""
    tmp_dir = tempfile.mkdtemp()
    registry_path = os.path.join(tmp_dir, "model_registry")
    os.makedirs(registry_path)
    
    data = {
        "amt": [100.0, 50.0, 200.0, 10.0] * 10,
        "lat": [40.7] * 40,
        "long": [-74.0] * 40,
        "city_pop": [1000000] * 40,
        "merch_lat": [40.8] * 40,
        "merch_long": [-73.9] * 40,
        "merchant": ["A", "B", "C", "D"] * 10,
        "category": ["X", "Y", "Z", "W"] * 10,
        "gender": ["M", "F", "M", "F"] * 10,
        "job": ["J1", "J2", "J3", "J4"] * 10,
        "trans_date_trans_time": ["2024-01-15 14:30:00"] * 40,
        "dob": ["1985-06-20"] * 40,
        "is_fraud": [0, 1, 0, 0] * 10,
    }
    df = pd.DataFrame(data)
    
    trainer = FraudModelTrainer()
    pipeline, metrics = trainer.train_and_evaluate(df)
    
    registry = ModelRegistry(registry_path=registry_path)
    
    # Версия 1 (threshold 0.5)
    v1 = registry.register_pipeline(pipeline, metrics, threshold=0.5, version="v1")
    registry.set_active_version(v1)
    
    # Версия 2 (threshold 0.1)
    v2 = registry.register_pipeline(pipeline, metrics, threshold=0.1, version="v2")
    
    yield {
        "registry": registry,
        "v1": v1,
        "v2": v2
    }
    
    shutil.rmtree(tmp_dir)


@pytest.fixture
def sample_transaction():
    """Пример транзакции для предсказания."""
    return {
        "amt": 125.50,
        "lat": 40.7128,
        "long": -74.0060,
        "city_pop": 8336817,
        "merch_lat": 40.7580,
        "merch_long": -73.9855,
        "merchant": "fraud_Kirlin and Sons",
        "category": "grocery_pos",
        "gender": "M",
        "job": "Software Engineer",
        "trans_date_trans_time": "2024-01-15 14:30:00",
        "dob": "1985-06-20",
    }


class TestFraudPredictorReal:
    """Тесты для FraudPredictor с реальными зависимостями."""

    def test_load_active_model(self, real_registry_setup):
        """Предиктор успешно загружает активную модель по умолчанию."""
        predictor = FraudPredictor(registry=real_registry_setup["registry"])
        success = predictor.load_model()
        
        assert success is True
        assert predictor.is_loaded is True
        assert predictor.version == real_registry_setup["v1"]
        assert predictor.threshold == 0.5

    def test_load_specific_version(self, real_registry_setup):
        """Предиктор успешно загружает конкретную версию."""
        predictor = FraudPredictor(registry=real_registry_setup["registry"])
        success = predictor.load_model(version=real_registry_setup["v2"])
        
        assert success is True
        assert predictor.version == real_registry_setup["v2"]
        assert predictor.threshold == 0.1

    def test_predict_real_pipeline(self, real_registry_setup, sample_transaction):
        """Реальный пайплайн выполняет предсказание без ошибок."""
        predictor = FraudPredictor(registry=real_registry_setup["registry"])
        predictor.load_model()
        
        is_fraud, probability = predictor.predict(sample_transaction)
        
        assert isinstance(is_fraud, bool)
        assert isinstance(probability, float)
        assert 0 <= probability <= 1

    def test_predict_batch_real_pipeline(self, real_registry_setup, sample_transaction):
        """Реальный пайплайн выполняет пакетное предсказание."""
        predictor = FraudPredictor(registry=real_registry_setup["registry"])
        predictor.load_model()
        
        transactions = [sample_transaction] * 3
        results = predictor.predict_batch(transactions)
        
        assert len(results) == 3
        for is_fraud, prob in results:
            assert isinstance(is_fraud, bool)
            assert 0 <= prob <= 1

    def test_reload_to_new_active(self, real_registry_setup):
        """Предиктор корректно обновляется при смене активной модели."""
        registry = real_registry_setup["registry"]
        registry.set_active_version(real_registry_setup["v1"])
        
        predictor = FraudPredictor(registry=registry)
        predictor.load_model()
        assert predictor.version == real_registry_setup["v1"]
        
        registry.set_active_version(real_registry_setup["v2"])
        
        success, old_ver, new_ver = predictor.reload_model()
        
        assert success is True
        assert old_ver == real_registry_setup["v1"]
        assert new_ver == real_registry_setup["v2"]
        assert predictor.version == real_registry_setup["v2"]
        assert predictor.threshold == 0.1

    def test_get_model_info_real(self, real_registry_setup):
        """Метод get_model_info возвращает реальные метаданные."""
        predictor = FraudPredictor(registry=real_registry_setup["registry"])
        predictor.load_model()
        
        info = predictor.get_model_info()
        
        assert info["loaded"] is True
        assert info["version"] == real_registry_setup["v1"]
        assert "metrics" in info
        assert "accuracy" in info["metrics"]
        assert "pipeline_steps" in info
        assert len(info["pipeline_steps"]) > 0
