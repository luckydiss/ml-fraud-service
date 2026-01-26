from dataclasses import dataclass
from typing import Dict, List, Optional

from src.common import get_logger, settings

logger = get_logger(__name__)


@dataclass
class ValidationResult:

    passed: bool
    metrics: Dict[str, float]
    thresholds: Dict[str, float]
    failures: List[str]
    message: str


class QualityGateValidator:
    """Проверка метрик модели на соответствие порогам (quality gates)."""

    def __init__(
        self,
        min_precision: Optional[float] = None,
        min_recall: Optional[float] = None,
        min_f1: Optional[float] = None,
    ):
        quality_gates = settings.get_quality_gates()

        self.thresholds = {
            "precision": min_precision or quality_gates["precision"],
            "recall": min_recall or quality_gates["recall"],
            "f1": min_f1 or quality_gates["f1"],
        }

        logger.info(f"Пороги качества инициализированы: {self.thresholds}")

    def validate(self, metrics: Dict[str, float]) -> ValidationResult:
        """Проверяет метрики на соответствие порогам."""
        failures = []

        for metric_name, threshold in self.thresholds.items():
            if metric_name not in metrics:
                failures.append(f"Отсутствует метрика: {metric_name}")
                continue

            actual_value = metrics[metric_name]
            if actual_value < threshold:
                failures.append(
                    f"{metric_name}: {actual_value:.4f} < {threshold:.4f} (threshold)"
                )

        passed = len(failures) == 0

        if passed:
            message = "Все пороги качества пройдены"
            logger.info(message)
        else:
            message = f"Проверка качества не пройдена: {'; '.join(failures)}"
            logger.warning(message)

        return ValidationResult(
            passed=passed,
            metrics=metrics,
            thresholds=self.thresholds,
            failures=failures,
            message=message,
        )

    def validate_and_raise(self, metrics: Dict[str, float]) -> ValidationResult:
        result = self.validate(metrics)

        if not result.passed:
            raise ValueError(f"Ошибка валидации качества: {result.message}")

        return result
