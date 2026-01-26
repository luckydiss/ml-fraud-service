import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from src.common import get_logger, settings

logger = get_logger(__name__)


class ModelRegistry:
    """Реестр моделей для хранения и версионирования пайплайнов.

        Структура:
        model_registry/
        ├── models/
        │   ├── v_20240115_120000/
        │   │   ├── pipeline.joblib    # (feature eng + preprocessing + model)
        │   │   └── metadata.json
        │   └── v_20240116_090000/
        │       └── ...
        └── active_version.txt
    """
    def __init__(self, registry_path: Optional[Path] = None):
        """Инициализация реестра моделей."""
        self.registry_path = Path(registry_path or settings.model_registry_path)
        self.models_path = self.registry_path / "models"
        self.active_version_file = self.registry_path / "active_version.txt"

        self.models_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Реестр моделей инициализирован в: {self.registry_path}")

    def _get_version_path(self, version: str) -> Path:
        """Возвращает путь к конкретной версии."""
        return self.models_path / version

    def _generate_version(self) -> str:
        """Генерирует имя версии на основе времени."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v_{timestamp}"

    def register_pipeline(
        self,
        pipeline: Any,
        metrics: Dict[str, float],
        threshold: float,
        version: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Регистрирует полный пайплайн как единый объект."""
        version = version or self._generate_version()
        version_path = self._get_version_path(version)

        version_path.mkdir(parents=True, exist_ok=True)

        pipeline_path = version_path / "pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)

        metadata = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "threshold": threshold,
            "pipeline_steps": [step[0] for step in pipeline.steps],
            **(extra_metadata or {}),
        }

        metadata_path = version_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Пайплайн зарегистрирован: {version}, метрики: {metrics}")
        return version

    def set_active_version(self, version: str) -> None:
        """Устанавливает активную версию для инференса."""
        version_path = self._get_version_path(version)
        if not version_path.exists():
            raise ValueError(f"Версия {version} не найдена в реестре")

        with open(self.active_version_file, "w") as f:
            f.write(version)

        logger.info(f"Активная версия пайплайна установлена: {version}")

    def get_active_version(self) -> Optional[str]:
        """Возвращает текущую активную версию."""
        if not self.active_version_file.exists():
            return None

        with open(self.active_version_file, "r") as f:
            return f.read().strip()

    def load_pipeline(self, version: Optional[str] = None) -> tuple:
        """Загружает пайплайн и его метаданные."""
        if version is None:
            version = self.get_active_version()
            if version is None:
                raise ValueError("Активная версия не установлена")

        version_path = self._get_version_path(version)
        if not version_path.exists():
            raise ValueError(f"Версия {version} не найдена")

        pipeline = joblib.load(version_path / "pipeline.joblib")

        with open(version_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        logger.info(f"Загружен пайплайн версии: {version}")
        return pipeline, metadata

    def list_versions(self) -> List[Dict[str, Any]]:
        """Список всех доступных версий пайплайнов."""
        versions = []
        for version_dir in self.models_path.iterdir():
            if version_dir.is_dir():
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        versions.append(json.load(f))

        versions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return versions

    def delete_version(self, version: str) -> None:
        """Удаляет версию пайплайна."""
        if version == self.get_active_version():
            raise ValueError("Нельзя удалить активную версию")

        version_path = self._get_version_path(version)
        if version_path.exists():
            shutil.rmtree(version_path)
            logger.info(f"Версия удалена: {version}")

    def get_metadata(self, version: Optional[str] = None) -> Dict[str, Any]:
        """Возвращает метаданные для конкретной версии."""
        if version is None:
            version = self.get_active_version()
            if version is None:
                raise ValueError("Активная версия не установлена")

        version_path = self._get_version_path(version)
        metadata_path = version_path / "metadata.json"

        if not metadata_path.exists():
            raise ValueError(f"Метаданные не найдены для версии {version}")

        with open(metadata_path, "r") as f:
            return json.load(f)
