from typing import Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """
    Сырые данные транзакции для предсказания фрода.
    """

    amt: float = Field(..., description="Сумма транзакции", ge=0)
    lat: float = Field(..., description="Широта клиента", ge=-90, le=90)
    long: float = Field(..., description="Долгота клиента", ge=-180, le=180)
    city_pop: int = Field(..., description="Население города", ge=0)
    merch_lat: float = Field(..., description="Широта продавца (мерчанта)", ge=-90, le=90)
    merch_long: float = Field(..., description="Долгота продавца (мерчанта)", ge=-180, le=180)
    merchant: str = Field(..., description="Название продавца", min_length=1)
    category: str = Field(..., description="Категория транзакции", min_length=1)
    gender: str = Field(..., description="Пол клиента")
    job: str = Field(..., description="Работа/профессия клиента")
    trans_date_trans_time: str = Field(
        ...,
        description="Время транзакции в формате ISO (YYYY-MM-DD HH:MM:SS)",
    )
    dob: str = Field(..., description="Дата рождения клиента (YYYY-MM-DD)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Ответ эндпоинта предсказания фрода."""

    is_fraud: bool = Field(..., description="Является ли транзакция мошеннической (вердикт)")
    fraud_probability: float = Field(
        ..., description="Вероятность фрода (0-1)", ge=0, le=1
    )
    threshold_used: float = Field(
        ..., description="Использованный порог классификации", ge=0, le=1
    )
    model_version: str = Field(..., description="Версия модели, использованная для предсказания")


class HealthResponse(BaseModel):
    """Ответ эндпоинта проверки состояния (health check)."""

    status: str = Field(..., description="Статус сервиса (healthy/degraded)")
    model_loaded: bool = Field(..., description="Загружена ли модель")
    model_version: Optional[str] = Field(None, description="Текущая версия модели")
    timestamp: str = Field(..., description="Временная метка проверки")


class ModelInfoResponse(BaseModel):
    """Ответ эндпоинта с информацией о модели."""

    version: str = Field(..., description="Версия модели")
    created_at: str = Field(..., description="Дата создания модели")
    threshold: float = Field(..., description="Порог классификации")
    metrics: dict = Field(..., description="Метрики оценки качества модели")
    pipeline_steps: list = Field(..., description="Список этапов (шагов) пайплайна")


class ReloadResponse(BaseModel):
    """Ответ эндпоинта перезагрузки модели."""

    success: bool = Field(..., description="Успешность перезагрузки")
    previous_version: Optional[str] = Field(None, description="Предыдущая версия модели")
    new_version: str = Field(..., description="Новая версия модели после перезагрузки")
    message: str = Field(..., description="Сообщение о статусе")


class ErrorResponse(BaseModel):
    """Стандартный ответ с ошибкой."""

    error: str = Field(..., description="Тип ошибки")
    message: str = Field(..., description="Сообщение об ошибке")
    detail: Optional[str] = Field(None, description="Дополнительные детали ошибки")


class TrainRequest(BaseModel):
    """Запрос на обучение модели."""

    force_register: bool = Field(
        default=False,
        description="Зарегистрировать модель даже если не прошла quality gates"
    )
    skip_reload: bool = Field(
        default=False,
        description="Пропустить автоматическую перезагрузку модели в inference"
    )
    use_database: bool = Field(
        default=True,
        description="Использовать данные из БД (True) или из CSV файла (False)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "force_register": False,
                    "skip_reload": False,
                    "use_database": True,
                }
            ]
        }
    }


class TrainResponse(BaseModel):
    """Ответ эндпоинта обучения модели."""

    success: bool = Field(..., description="Успешность обучения")
    version: Optional[str] = Field(None, description="Версия зарегистрированной модели")
    metrics: Optional[dict] = Field(None, description="Метрики обученной модели")
    validation_passed: bool = Field(..., description="Прошла ли модель quality gates")
    reload_triggered: bool = Field(..., description="Была ли перезагружена модель в inference")
    message: str = Field(..., description="Сообщение о статусе обучения")


class DatabaseStatsResponse(BaseModel):
    """Статистика базы данных."""

    total: int = Field(..., description="Общее количество транзакций")
    fraud: int = Field(..., description="Количество мошеннических транзакций")
    legitimate: int = Field(..., description="Количество легитимных транзакций")
    unlabeled: int = Field(..., description="Количество транзакций без метки")


class DataUploadResponse(BaseModel):
    """Ответ на загрузку данных в БД."""

    success: bool = Field(..., description="Успешность загрузки")
    records_loaded: int = Field(..., description="Количество загруженных записей")
    message: str = Field(..., description="Сообщение о статусе")