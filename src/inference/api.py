"""
FastAPI application for fraud detection service.

Endpoints:
- POST /predict — Transaction fraud prediction (Inference endpoint)
- POST /train — Train new model (Train endpoint)
- POST /reload — Reload model / Hotswap (Hotswap endpoint)
- GET /health — Service health check
- GET /model/info — Current model version & metrics
- GET /database/stats — Database statistics
- POST /database/load — Load data from CSV to database

"""

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.common import get_logger, settings
from src.database import TransactionDatabase, get_database

from .predictor import FraudPredictor
from .schemas import (
    DatabaseStatsResponse,
    DataUploadResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReloadResponse,
    TrainRequest,
    TrainResponse,
    TransactionRequest,
)

logger = get_logger(__name__)

predictor: FraudPredictor = None
database: TransactionDatabase = None
_training_in_progress: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения FastAPI.
    - При старте: инициализирует FraudPredictor, Database и пытается загрузить активную модель.
    - При выключении: логирует завершение работы сервиса.
    """
    global predictor, database

    logger.info("Запуск ML Fraud Detection Service")
    
    # Инициализация базы данных
    database = get_database()
    logger.info("База данных инициализирована")
    
    # Инициализация предиктора
    predictor = FraudPredictor()

    try:
        if predictor.load_model():
            logger.info(f"Модель загружена при старте: {predictor.version}")
        else:
            logger.warning("Модель не загружена при старте. Используйте endpoint /reload.")
    except Exception as e:
        logger.warning(f"Не удалось загрузить модель при старте: {e}")

    yield

    logger.info("Остановка ML Fraud Detection Service")


app = FastAPI(
    title="Credit Fraud Detection API",
    description="""
## ML-сервис для детекции мошеннических транзакций

### Основные возможности:
- **Inference** — предсказание фрода в реальном времени
- **Training** — обучение новых моделей через API
- **Hotswap** — горячая замена модели без остановки сервиса
- **Database** — хранение транзакций в SQLite

### Архитектура:
```
Prediction request → Inference endpoint → Preprocessing → Inference → Response
Train request → Train endpoint → Preprocessing → Training → AI Model
Hotswap request → Hotswap endpoint → Model reload
```
    """,
    version="2.0.0",
    lifespan=lifespan,
)


# =============================================================================
# INFERENCE ENDPOINT
# =============================================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Предсказание фрода",
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        500: {"model": ErrorResponse, "description": "Prediction error"},
    },
)
async def predict(transaction: TransactionRequest) -> PredictionResponse:
    """
    Предсказание вероятности фрода для входящей транзакции.
    
    Эндпоинт принимает "сырые" данные, выполняет feature engineering 
    внутри пайплайна и возвращает вердикт модели.
    
    Returns:
        PredictionResponse: результат классификации, вероятность фрода и версия используемой модели.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена. Сделайте запрос на /reload",
        )

    try:
        transaction_dict = transaction.model_dump()
        is_fraud, fraud_probability = predictor.predict(transaction_dict)

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=fraud_probability,
            threshold_used=predictor.threshold,
            model_version=predictor.version,
        )

    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


# =============================================================================
# TRAIN ENDPOINT
# =============================================================================

def _run_training(
    use_database: bool = True,
    force_register: bool = False,
    skip_reload: bool = False,
) -> dict:
    """Синхронная функция обучения (вызывается в background)."""
    from src.training.main import run_training_pipeline, load_training_data
    from src.registry import ModelRegistry
    from src.training.trainer import FraudModelTrainer
    from src.training.validator import QualityGateValidator
    
    global _training_in_progress
    
    result = {
        "success": False,
        "metrics": None,
        "validation_passed": False,
        "version": None,
        "reload_triggered": False,
        "message": "",
    }
    
    try:
        _training_in_progress = True
        logger.info("=" * 60)
        logger.info("Запуск обучения через API")
        logger.info("=" * 60)
        
        # Шаг 1: Загрузка данных
        if use_database:
            logger.info("Шаг 1: Загрузка данных из БД")
            df = database.get_training_data()
            if len(df) == 0:
                result["message"] = "Нет данных для обучения в БД"
                return result
        else:
            logger.info("Шаг 1: Загрузка данных из CSV")
            df = load_training_data()
        
        logger.info(f"Загружено {len(df)} записей")
        
        # Шаг 2: Обучение
        logger.info("Шаг 2: Обучение модели")
        trainer = FraudModelTrainer()
        pipeline, metrics = trainer.train_and_evaluate(df)
        result["metrics"] = metrics
        
        # Шаг 3: Валидация
        logger.info("Шаг 3: Валидация качества")
        validator = QualityGateValidator()
        validation = validator.validate(metrics)
        result["validation_passed"] = validation.passed
        
        if not validation.passed and not force_register:
            result["message"] = f"Валидация не пройдена: {validation.failures}"
            logger.warning(result["message"])
            return result
        
        # Шаг 4: Регистрация
        logger.info("Шаг 4: Регистрация модели")
        registry = ModelRegistry()
        version = registry.register_pipeline(
            pipeline=pipeline,
            metrics=metrics,
            threshold=trainer.threshold,
            extra_metadata={
                "validation_passed": validation.passed,
                "quality_gates": validation.thresholds,
                "trained_via": "api",
            },
        )
        registry.set_active_version(version)
        result["version"] = version
        
        # Шаг 5: Reload (если нужно)
        if not skip_reload:
            logger.info("Шаг 5: Перезагрузка модели")
            success, _, _ = predictor.reload_model()
            result["reload_triggered"] = success
        
        result["success"] = True
        result["message"] = f"Модель успешно обучена и зарегистрирована: {version}"
        
        logger.info("=" * 60)
        logger.info(result["message"])
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Ошибка обучения: {e}")
        result["message"] = f"Ошибка обучения: {str(e)}"
    finally:
        _training_in_progress = False
    
    return result


@app.post(
    "/train",
    response_model=TrainResponse,
    tags=["Training"],
    summary="Запуск обучения модели",
    responses={
        409: {"model": ErrorResponse, "description": "Training already in progress"},
        500: {"model": ErrorResponse, "description": "Training error"},
    },
)
async def train_model(request: TrainRequest) -> TrainResponse:
    """
    Запуск обучения новой модели.
    
    Обучение выполняется синхронно и может занять несколько минут.
    После успешного обучения модель автоматически регистрируется
    и загружается в inference (если не указан skip_reload).
    
    Returns:
        TrainResponse: результат обучения с метриками и версией модели.
    """
    global _training_in_progress
    
    if _training_in_progress:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Обучение уже выполняется. Дождитесь завершения.",
        )
    
    try:
        result = _run_training(
            use_database=request.use_database,
            force_register=request.force_register,
            skip_reload=request.skip_reload,
        )
        
        return TrainResponse(
            success=result["success"],
            version=result["version"],
            metrics=result["metrics"],
            validation_passed=result["validation_passed"],
            reload_triggered=result["reload_triggered"],
            message=result["message"],
        )
        
    except Exception as e:
        logger.error(f"Ошибка обучения: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}",
        )


# =============================================================================
# HOTSWAP ENDPOINT
# =============================================================================

@app.post(
    "/reload",
    response_model=ReloadResponse,
    tags=["Hotswap"],
    summary="Горячая замена модели",
    responses={
        500: {"model": ErrorResponse, "description": "Reload failed"},
    },
)
async def reload_model() -> ReloadResponse:
    """
    Принудительная перезагрузка модели из реестра моделей (Hotswap).
    
    Используется для обновления версии модели в памяти
    без перезапуска контейнера.
    
    Returns:
        ReloadResponse: статус выполнения и информация о новой версии модели.
    """
    try:
        success, previous_version, new_version = predictor.reload_model()

        if success:
            return ReloadResponse(
                success=True,
                previous_version=previous_version,
                new_version=new_version,
                message="Модель успешно перезагружена",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Не удалось перезагрузить модель",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка перезагрузки: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reload failed: {str(e)}",
        )


# =============================================================================
# DATABASE ENDPOINTS
# =============================================================================

@app.get(
    "/database/stats",
    response_model=DatabaseStatsResponse,
    tags=["Database"],
    summary="Статистика базы данных",
)
async def database_stats() -> DatabaseStatsResponse:
    """
    Получение статистики по транзакциям в базе данных.
    
    Returns:
        DatabaseStatsResponse: количество транзакций по категориям.
    """
    stats = database.count_transactions()
    return DatabaseStatsResponse(**stats)


@app.post(
    "/database/load",
    response_model=DataUploadResponse,
    tags=["Database"],
    summary="Загрузка данных из CSV в БД",
    responses={
        404: {"model": ErrorResponse, "description": "CSV file not found"},
        500: {"model": ErrorResponse, "description": "Load error"},
    },
)
async def load_data_to_database(replace: bool = False) -> DataUploadResponse:
    """
    Загрузка данных из CSV файла в базу данных.
    
    Args:
        replace: Если True, очищает БД перед загрузкой.
        
    Returns:
        DataUploadResponse: результат загрузки.
    """
    try:
        # Ищем CSV файл в директории данных
        csv_files = list(settings.data_path.glob("*.csv"))
        
        if not csv_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"CSV файлы не найдены в {settings.data_path}",
            )
        
        csv_path = csv_files[0]
        logger.info(f"Загрузка данных из: {csv_path}")
        
        count = database.load_from_csv(csv_path, replace=replace)
        
        return DataUploadResponse(
            success=True,
            records_loaded=count,
            message=f"Загружено {count} записей из {csv_path.name}",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Load failed: {str(e)}",
        )


# =============================================================================
# HEALTH & INFO ENDPOINTS
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Проверка состояния сервиса",
)
async def health_check() -> HealthResponse:
    """
    Проверка работоспособности сервиса (Health Check).
    
    Returns:
        HealthResponse: статус ('healthy' или 'degraded'), информация о загрузке модели и текущий timestamp.
    """
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        model_version=predictor.version,
        timestamp=datetime.now().isoformat(),
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Health"],
    summary="Информация о модели",
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def model_info() -> ModelInfoResponse:
    """
    Получение детальной информации о текущей загруженной модели.
    
    Returns:
        ModelInfoResponse: версия, метрики качества (f1, recall и др.), порог классификации и список шагов пайплайна.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена",
        )

    info = predictor.get_model_info()

    return ModelInfoResponse(
        version=info["version"],
        created_at=info["created_at"],
        threshold=info["threshold"],
        metrics=info["metrics"],
        pipeline_steps=info["pipeline_steps"],
    )


# =============================================================================
# EXCEPTION HANDLER
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Глобальный обработчик необработанных исключений.

    Перехватывает любые ошибки сервера (500), не обработанные в эндпоинтах, 
    и возвращает их клиенту в унифицированном JSON-формате.
    """
    logger.error(f"Необработанная ошибка: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "Произошла неожиданная ошибка",
            "detail": str(exc) if settings.debug else None,
        },
    )
