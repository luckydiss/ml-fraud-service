import logging
import sys
from typing import Optional

from .config import settings

_configured_loggers: set = set()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Возвращает настроенный экземпляр логгера."""
    logger_name = name or settings.app_name

    if logger_name in _configured_loggers:
        return logging.getLogger(logger_name)

    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(settings.log_format)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    logger.propagate = False

    _configured_loggers.add(logger_name)
    return logger


def configure_root_logger() -> None:
    """Настраивает корневой логгер приложения."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(settings.log_format)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)