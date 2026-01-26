import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference import FraudPredictor
from src.preprocessing import (
    RAW_INPUT_COLUMNS,
)
from src.registry import ModelRegistry
from src.training import FraudModelTrainer, QualityGateValidator


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic transaction data for testing."""
    np.random.seed(42)

    # Generate timestamps
    base_date = datetime(2024, 1, 1)
    timestamps = [
        base_date + timedelta(hours=np.random.randint(0, 24*30))
        for _ in range(n_samples)
    ]
    timestamps.sort()

    # Generate data
    data = {
        "trans_date_trans_time": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        "amt": np.random.exponential(100, n_samples),
        "lat": np.random.uniform(25, 48, n_samples),
        "long": np.random.uniform(-125, -70, n_samples),
        "city_pop": np.random.randint(1000, 1000000, n_samples),
        "merch_lat": np.random.uniform(25, 48, n_samples),
        "merch_long": np.random.uniform(-125, -70, n_samples),
        "merchant": np.random.choice(["merchant_A", "merchant_B", "merchant_C"], n_samples),
        "category": np.random.choice(["grocery_pos", "gas_transport", "shopping_net"], n_samples),
        "gender": np.random.choice(["M", "F"], n_samples),
        "job": np.random.choice(["Engineer", "Doctor", "Teacher", "Nurse"], n_samples),
        "dob": [
            (base_date - timedelta(days=np.random.randint(7000, 25000))).strftime("%Y-%m-%d")
            for _ in range(n_samples)
        ],
        "is_fraud": np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
    }

    return pd.DataFrame(data)


@pytest.fixture
def temp_registry():
    """Create a temporary model registry for testing."""
    temp_dir = tempfile.mkdtemp()
    registry_path = Path(temp_dir) / "model_registry"
    registry_path.mkdir(parents=True, exist_ok=True)

    yield registry_path

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    return generate_sample_data(500)


@pytest.fixture
def sample_transaction():
    """Generate a single sample transaction for prediction."""
    return {
        "amt": 125.50,
        "lat": 40.7128,
        "long": -74.0060,
        "city_pop": 8336817,
        "merch_lat": 40.7580,
        "merch_long": -73.9855,
        "merchant": "merchant_A",
        "category": "grocery_pos",
        "gender": "M",
        "job": "Engineer",
        "trans_date_trans_time": "2024-01-15 14:30:00",
        "dob": "1985-06-20",
    }


class TestFullMLCycle:
    """End-to-end tests for the full ML lifecycle."""

    def test_train_register_reload_predict(
        self, temp_registry, sample_data, sample_transaction
    ):
        """
        Test complete ML lifecycle:
        Train → Register → Reload → Predict
        """
        # Step 1: Train model
        trainer = FraudModelTrainer()
        pipeline, metrics = trainer.train_and_evaluate(sample_data)

        assert pipeline is not None
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

        # Step 2: Register in registry
        registry = ModelRegistry(registry_path=temp_registry)
        version = registry.register_pipeline(
            pipeline=pipeline,
            metrics=metrics,
            threshold=trainer.threshold,
        )
        registry.set_active_version(version)

        assert version is not None
        assert registry.get_active_version() == version

        # Step 3: Load model in predictor (simulates reload)
        predictor = FraudPredictor(registry=registry)
        success = predictor.load_model()

        assert success
        assert predictor.is_loaded
        assert predictor.version == version

        # Step 4: Make prediction
        is_fraud, probability = predictor.predict(sample_transaction)

        assert isinstance(is_fraud, bool)
        assert 0.0 <= probability <= 1.0

    def test_quality_gates_validation(self, sample_data):
        """Test that quality gates correctly validate metrics."""
        trainer = FraudModelTrainer()
        _, metrics = trainer.train_and_evaluate(sample_data)

        # Test with very low thresholds (should pass)
        validator_low = QualityGateValidator(
            min_precision=0.01,
            min_recall=0.01,
            min_f1=0.01,
        )
        result = validator_low.validate(metrics)
        assert result.passed

        # Test with very high thresholds (should fail)
        validator_high = QualityGateValidator(
            min_precision=0.99,
            min_recall=0.99,
            min_f1=0.99,
        )
        result = validator_high.validate(metrics)
        assert not result.passed
        assert len(result.failures) > 0

    def test_pipeline_processes_raw_data(self, temp_registry, sample_data):
        """Test that pipeline correctly processes raw transaction data."""
        # Train and register
        trainer = FraudModelTrainer()
        pipeline, metrics = trainer.train_and_evaluate(sample_data)

        registry = ModelRegistry(registry_path=temp_registry)
        version = registry.register_pipeline(
            pipeline=pipeline,
            metrics=metrics,
            threshold=trainer.threshold,
        )
        registry.set_active_version(version)

        # Load and predict on raw data
        loaded_pipeline, metadata = registry.load_pipeline()

        # Create raw test data (without engineered features)
        raw_test = sample_data[RAW_INPUT_COLUMNS].head(10)

        # Pipeline should handle raw data and produce predictions
        predictions = loaded_pipeline.predict(raw_test)
        probabilities = loaded_pipeline.predict_proba(raw_test)

        assert len(predictions) == 10
        assert probabilities.shape == (10, 2)
        assert all(p in [0, 1] for p in predictions)

    def test_model_reload_preserves_version(self, temp_registry, sample_data):
        """Test that model reload correctly updates version."""
        trainer = FraudModelTrainer()

        # Train and register first version
        pipeline1, metrics1 = trainer.train_and_evaluate(sample_data)
        registry = ModelRegistry(registry_path=temp_registry)
        version1 = registry.register_pipeline(
            pipeline=pipeline1,
            metrics=metrics1,
            threshold=trainer.threshold,
        )
        registry.set_active_version(version1)

        # Load in predictor
        predictor = FraudPredictor(registry=registry)
        predictor.load_model()
        assert predictor.version == version1

        # Train and register second version
        pipeline2, metrics2 = trainer.train_and_evaluate(sample_data)
        version2 = registry.register_pipeline(
            pipeline=pipeline2,
            metrics=metrics2,
            threshold=trainer.threshold,
        )
        registry.set_active_version(version2)

        # Reload should get new version
        success, prev_version, new_version = predictor.reload_model()

        assert success
        assert prev_version == version1
        assert new_version == version2
        assert predictor.version == version2

    def test_registry_lists_versions(self, temp_registry, sample_data):
        """Test that registry correctly lists all versions."""
        trainer = FraudModelTrainer()
        registry = ModelRegistry(registry_path=temp_registry)

        # Register multiple versions
        versions = []
        for i in range(3):
            pipeline, metrics = trainer.train_and_evaluate(sample_data)
            version = registry.register_pipeline(
                pipeline=pipeline,
                metrics=metrics,
                threshold=trainer.threshold,
            )
            versions.append(version)

        # List versions
        listed = registry.list_versions()

        assert len(listed) == 3
        for v in versions:
            assert any(item["version"] == v for item in listed)


class TestFeatureEngineering:
    """Tests for feature engineering inside pipeline."""

    def test_time_features_extracted(self, sample_data):
        """Test that time features are correctly extracted."""
        from src.preprocessing.features import TimeFeatureExtractor

        extractor = TimeFeatureExtractor()
        result = extractor.transform(sample_data)

        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "day_of_month" in result.columns
        assert "month" in result.columns

    def test_distance_calculated(self, sample_data):
        """Test that distance is correctly calculated."""
        from src.preprocessing.features import DistanceCalculator

        calculator = DistanceCalculator()
        result = calculator.transform(sample_data)

        assert "distance" in result.columns
        assert all(result["distance"] >= 0)

    def test_age_calculated(self, sample_data):
        """Test that age is correctly calculated."""
        from src.preprocessing.features import AgeCalculator

        calculator = AgeCalculator()
        result = calculator.transform(sample_data)

        assert "age" in result.columns
        assert all(result["age"] >= 0)
        assert all(result["age"] < 120)  # Reasonable age range
