
import warnings
from typing import Any, Dict, Optional, Tuple

import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

from src.common import get_logger, settings
from src.registry import ModelRegistry

logger = get_logger(__name__)


class FraudPredictor:
    """
    Класс-обертка для управления жизненным циклом ML-модели и выполнения инференса.
    
    Обеспечивает:
    - Загрузку и обновление модели из реестра.
    - Выполнение предсказаний без сохранения состояния (stateless).
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        """ 
        Инициализация предиктора

        Args:
            registry: Экземпляр ModelRegistry для загрузки моделей. Если None, создает новый.
        """
        self.registry = registry or ModelRegistry()
        self.pipeline = None
        self.metadata: Dict[str, Any] = {}
        self.threshold: float = settings.optimal_threshold

        logger.info("FraudPredictor инициализирован")

    def load_model(self, version: Optional[str] = None) -> bool:
        """
        Первичная загрузка модели из реестра
        
        Args:
            version: Версия модели (например, 'v_20240101_120000'). Если None, грузит активную.
            
        Returns:
            bool: True если загрузка прошла успешно, иначе False.
        """
        try:
            logger.info(f"Загрузка версии модели: {version or 'active'}")
            self.pipeline, self.metadata = self.registry.load_pipeline(version)
            self.threshold = self.metadata.get("threshold", settings.optimal_threshold)
            logger.info(f"Модель загружена: {self.metadata.get('version')}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False

    def reload_model(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Перезагрузка модели на последнюю активную версию из реестра.
        
        Используется для обновления модели в работающем сервисе без его остановки.
        
        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (успех, старая версия, новая версия).
        """
        previous_version = self.metadata.get("version")

        try:
            logger.info("Перезагрузка модели из реестра")
            self.pipeline, self.metadata = self.registry.load_pipeline()
            self.threshold = self.metadata.get("threshold", settings.optimal_threshold)
            new_version = self.metadata.get("version")
            logger.info(f"Модель обновлена: {previous_version} -> {new_version}")
            return True, previous_version, new_version
        except Exception as e:
            logger.error(f"Ошибка обновления модели: {e}")
            return False, previous_version, None

    def predict(self, transaction: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Предсказание фрода для одной транзакции.
        
        Args:
            transaction: Словарь с "сырыми" данными транзакции (соответствует TransactionRequest).
            
        Returns:
            Tuple[bool, float]: (флаг фрода, вероятность от 0 до 1).
        """
        if self.pipeline is None:
            raise ValueError("Модель не загружена")

        df = pd.DataFrame([transaction])

        proba = self.pipeline.predict_proba(df)[0, 1]

        is_fraud = proba >= self.threshold

        return bool(is_fraud), float(proba)

    def predict_batch(self, transactions: list) -> list:
        """
        Пакетное предсказание фрода для списка транзакций.
        
        Args:
            transactions: Список словарей с данными транзакций.
            
        Returns:
            list: Список кортежей (флаг фрода, вероятность).
        """
        if self.pipeline is None:
            raise ValueError("Модель не загружена")

        df = pd.DataFrame(transactions)

        probas = self.pipeline.predict_proba(df)[:, 1]

        results = []
        for proba in probas:
            is_fraud = proba >= self.threshold
            results.append((bool(is_fraud), float(proba)))

        return results

    @property
    def is_loaded(self) -> bool:
        """Проверка, инициализирован ли Pipeline и готов ли он к работе."""
        return self.pipeline is not None

    @property
    def version(self) -> Optional[str]:
        """Возвращает строковый идентификатор текущей версии модели."""
        return self.metadata.get("version")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о текущей модели и её параметрах.
        
        Returns:
            Dict[str, Any]: словарь с версией, метриками, порогом и шагами пайплайна.
        """
        if not self.is_loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "version": self.metadata.get("version"),
            "created_at": self.metadata.get("created_at"),
            "threshold": self.threshold,
            "metrics": self.metadata.get("metrics", {}),
            "pipeline_steps": self.metadata.get("pipeline_steps", []),
        }
