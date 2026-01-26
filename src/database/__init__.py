"""
Модуль работы с базой данных.

Предоставляет:
- TransactionDatabase: класс для работы с SQLite
- Функции для сохранения и загрузки транзакций
"""

from .db import TransactionDatabase, get_database

__all__ = ["TransactionDatabase", "get_database"]
