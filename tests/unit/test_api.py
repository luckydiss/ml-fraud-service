import os
import shutil
import tempfile
from datetime import datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.inference.predictor import FraudPredictor
from src.registry import ModelRegistry
from src.training.trainer import FraudModelTrainer


@pytest.fixture(scope="module")
def test_env():
    """Создает временное окружение для тестов: реестр и одну обученную модель."""
    tmp_dir = tempfile.mkdtemp()
    registry_path = os.path.join(tmp_dir, "model_registry")
    os.makedirs(registry_path)
    
    data = {
        "amt": [100.0, 50.0] * 5,
        "lat": [40.7] * 10,
        "long": [-74.0] * 10,
        "city_pop": [1000000] * 10,
        "merch_lat": [40.8] * 10,
        "merch_long": [-73.9] * 10,
        "merchant": ["shop_A"] * 10,
        "category": ["grocery"] * 10,
        "gender": ["M"] * 10,
        "job": ["Engineer"] * 10,
        "trans_date_trans_time": ["2024-01-15 14:30:00"] * 10,
        "dob": ["1985-06-20"] * 10,
        "is_fraud": [0, 1] * 5,
    }
    df = pd.DataFrame(data)
    
    trainer = FraudModelTrainer()
    pipeline, metrics = trainer.train_and_evaluate(df)
    
    registry = ModelRegistry(registry_path=registry_path)
    version = registry.register_pipeline(pipeline, metrics, threshold=0.5)
    registry.set_active_version(version)
    
    yield {
        "registry_path": registry_path,
        "registry": registry,
        "active_version": version
    }
    
    shutil.rmtree(tmp_dir)


@pytest.fixture
def real_predictor(test_env):
    """Создает FraudPredictor, подключенный к временному реестру."""
    predictor = FraudPredictor(registry=test_env["registry"])
    predictor.load_model()
    return predictor


@pytest.fixture
def client(real_predictor):
    """TestClient с реальным предиктором."""
    # патчим предиктор в модуле api.py
    with pytest.MonkeyPatch.context() as m:
        m.setattr("src.inference.api.predictor", real_predictor)
        from src.inference.api import app
        import src.inference.api as api
        api.predictor = real_predictor
        
        yield TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def sample_transaction_payload():
    """Валидный payload для /predict."""
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


class TestApiWithRealPredictor:

    def test_health_check(self, client, test_env):
        """Проверка health-check с загруженной моделью."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["model_version"] == test_env["active_version"]

    def test_predict_real_flow(self, client, sample_transaction_payload):
        """Проверка полного цикла предсказания через API."""
        response = client.post("/predict", json=sample_transaction_payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert isinstance(data["is_fraud"], bool)
        assert 0 <= data["fraud_probability"] <= 1

    def test_model_info_real_data(self, client, test_env):
        """Проверка получения метаданных реальной модели."""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["version"] == test_env["active_version"]
        assert "metrics" in data
        assert "f1" in data["metrics"]
        assert data["threshold"] == 0.5

    def test_predict_validation_error(self, client, sample_transaction_payload):
        """Проверка, что Pydantic всё еще ловит ошибки до предиктора."""
        invalid_payload = sample_transaction_payload.copy()
        invalid_payload["lat"] = 999  # Невалидная широта
        
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422

    def test_reload_to_same_version(self, client, test_env):
        """Проверка перезагрузки"""
        response = client.post("/reload")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["new_version"] == test_env["active_version"]
