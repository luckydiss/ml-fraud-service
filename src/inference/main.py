import uvicorn

from src.common import get_logger, settings
from src.common.logging import configure_root_logger

logger = get_logger(__name__)


def main():
    configure_root_logger()

    logger.info("Запуск Fraud Detection Inference Service")
    logger.info(f"Окружение: {settings.app_env}")
    logger.info(f"Адрес: {settings.inference_host}:{settings.inference_port}")

    uvicorn.run(
        "src.inference.api:app",
        host=settings.inference_host,
        port=settings.inference_port,
        reload=settings.debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()
