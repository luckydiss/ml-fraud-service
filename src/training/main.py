"""
полный цикл:

1. обучение
2. оценка
3. проверка quality gates
4. регистрация пайплайна в реестре (если прошёл quality gates)"""

import sys
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd

from src.common import get_logger, settings
from src.common.logging import configure_root_logger
from src.registry import ModelRegistry

from .trainer import FraudModelTrainer
from .validator import QualityGateValidator

logger = get_logger(__name__)


def load_training_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Загрузка тренировочных данных из CSV."""
    data_path = data_path or settings.data_path

    if data_path.is_dir():
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"CSV файлы не найдены в {data_path}")
        data_file = csv_files[0]
        logger.info(f"Найден файл данных: {data_file}")
    else:
        data_file = data_path

    logger.info(f"Загрузка данных из: {data_file}")
    df = pd.read_csv(data_file)
    logger.info(f"Загружено строк: {len(df)}")

    return df


def trigger_inference_reload(
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: float = 30.0,
) -> bool:
    """Перезагрузка модели в сервисе инференса."""
    host = host or settings.inference_host
    port = port or settings.inference_port

    if host == "0.0.0.0":
        host = "127.0.0.1"

    url = f"http://{host}:{port}/reload"

    logger.info(f"Запрос на перезагрузку инференса: {url}")

    try:
        response = httpx.post(url, timeout=timeout)
        response.raise_for_status()
        logger.info("Инференс успешно перезагружен")
        return True
    except httpx.ConnectError:
        logger.warning(f"Нет связи с сервисом инференса: {url}")
        return False
    except httpx.HTTPStatusError as e:
        logger.error(f"Ошибка перезагрузки инференса: {e}")
        return False
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при перезагрузке: {e}")
        return False


def run_training_pipeline(
    data_path: Optional[Path] = None,
    skip_reload: bool = False,
    force_register: bool = False,
) -> dict:
    """Запуск полного цикла обучения модели."""
    logger.info("=" * 60)
    logger.info("Запуск пайплайна обучения")
    logger.info("=" * 60)

    result = {
        "success": False,
        "metrics": None,
        "validation_passed": False,
        "version": None,
        "reload_triggered": False,
    }

    try:
        logger.info("Шаг 1: Загрузка данных")
        df = load_training_data(data_path)

        logger.info("Шаг 2: Обучение и оценка модели")
        trainer = FraudModelTrainer()
        pipeline, metrics = trainer.train_and_evaluate(df)
        result["metrics"] = metrics

        logger.info("Шаг 3: Валидация метрик")
        validator = QualityGateValidator()
        validation = validator.validate(metrics)
        result["validation_passed"] = validation.passed

        if not validation.passed and not force_register:
            logger.warning("Валидация не пройдена. Регистрация отменена.")
            logger.warning(f"Ошибки: {validation.failures}")
            result["success"] = False
            return result

        if not validation.passed and force_register:
            logger.warning("Валидация не пройдена, но форсируем регистрацию...")

        logger.info("Шаг 4: Регистрация пайплайна")
        registry = ModelRegistry()
        version = registry.register_pipeline(
            pipeline=pipeline,
            metrics=metrics,
            threshold=trainer.threshold,
            extra_metadata={
                "validation_passed": validation.passed,
                "quality_gates": validation.thresholds,
            },
        )
        registry.set_active_version(version)
        result["version"] = version

        if not skip_reload:
            logger.info("Шаг 5: Перезагрузка инференса")
            result["reload_triggered"] = trigger_inference_reload()
        else:
            logger.info("Шаг 5: Пропуск перезагрузки инференса")

        result["success"] = True
        logger.info("=" * 60)
        logger.info("Пайплайн обучения успешно завершен!")
        logger.info(f"Версия: {version}")
        logger.info(f"Метрики: {metrics}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Ошибка пайплайна обучения: {e}")
        raise

    return result


def main():
    configure_root_logger()

    logger.info("Обучение модели детекции фрода")
    logger.info(f"Окружение: {settings.app_env}")

    try:
        result = run_training_pipeline()

        if result["success"]:
            logger.info("Обучение завершено успешно")
            sys.exit(0)
        else:
            logger.error("Обучение не прошло валидацию")
            sys.exit(1)

    except FileNotFoundError as e:
        logger.error(f"Данные не найдены: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка обучения: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
