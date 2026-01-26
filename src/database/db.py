"""
SQLite база данных для хранения транзакций.

Обеспечивает:
- Хранение транзакций для обучения
- Загрузка данных из CSV в БД
- Выгрузка данных для обучения
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.common import get_logger, settings

logger = get_logger(__name__)


class TransactionDatabase:
    """SQLite база данных для хранения транзакций."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Инициализация базы данных.
        
        Args:
            db_path: Путь к файлу БД. По умолчанию: data/transactions.db
        """
        self.db_path = db_path or (settings.data_path / "transactions.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"База данных инициализирована: {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Контекстный менеджер для подключения к БД."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Создание схемы базы данных."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Таблица транзакций
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trans_date_trans_time TEXT NOT NULL,
                    amt REAL NOT NULL,
                    lat REAL NOT NULL,
                    long REAL NOT NULL,
                    city_pop INTEGER NOT NULL,
                    merch_lat REAL NOT NULL,
                    merch_long REAL NOT NULL,
                    merchant TEXT NOT NULL,
                    category TEXT NOT NULL,
                    gender TEXT NOT NULL,
                    job TEXT NOT NULL,
                    dob TEXT NOT NULL,
                    is_fraud INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Индексы для быстрого поиска
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trans_date 
                ON transactions(trans_date_trans_time)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_is_fraud 
                ON transactions(is_fraud)
            """)
            
            logger.debug("Схема БД инициализирована")
    
    def insert_transaction(
        self,
        trans_date_trans_time: str,
        amt: float,
        lat: float,
        long: float,
        city_pop: int,
        merch_lat: float,
        merch_long: float,
        merchant: str,
        category: str,
        gender: str,
        job: str,
        dob: str,
        is_fraud: Optional[int] = None,
    ) -> int:
        """
        Вставка одной транзакции в БД.
        
        Returns:
            int: ID вставленной записи.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO transactions (
                    trans_date_trans_time, amt, lat, long, city_pop,
                    merch_lat, merch_long, merchant, category, gender,
                    job, dob, is_fraud
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trans_date_trans_time, amt, lat, long, city_pop,
                merch_lat, merch_long, merchant, category, gender,
                job, dob, is_fraud
            ))
            return cursor.lastrowid
    
    def insert_many(self, transactions: List[dict]) -> int:
        """
        Пакетная вставка транзакций.
        
        Args:
            transactions: Список словарей с данными транзакций.
            
        Returns:
            int: Количество вставленных записей.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            columns = [
                "trans_date_trans_time", "amt", "lat", "long", "city_pop",
                "merch_lat", "merch_long", "merchant", "category", "gender",
                "job", "dob", "is_fraud"
            ]
            
            placeholders = ", ".join(["?"] * len(columns))
            sql = f"INSERT INTO transactions ({', '.join(columns)}) VALUES ({placeholders})"
            
            values = [
                tuple(t.get(col) for col in columns)
                for t in transactions
            ]
            
            cursor.executemany(sql, values)
            count = cursor.rowcount
            logger.info(f"Вставлено {count} транзакций")
            return count
    
    def load_from_csv(self, csv_path: Path, replace: bool = False) -> int:
        """
        Загрузка данных из CSV файла в БД.
        
        Args:
            csv_path: Путь к CSV файлу.
            replace: Если True, очищает таблицу перед загрузкой.
            
        Returns:
            int: Количество загруженных записей.
        """
        logger.info(f"Загрузка данных из CSV: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Переименуем колонки если нужно
        required_columns = [
            "trans_date_trans_time", "amt", "lat", "long", "city_pop",
            "merch_lat", "merch_long", "merchant", "category", "gender",
            "job", "dob", "is_fraud"
        ]
        
        # Проверяем наличие колонок
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Отсутствуют колонки в CSV: {missing}")
        
        if replace:
            self.clear_transactions()
        
        # Конвертируем в список словарей
        transactions = df[required_columns].to_dict("records")
        
        return self.insert_many(transactions)
    
    def get_training_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Получение данных для обучения.
        
        Args:
            limit: Максимальное количество записей. None = все.
            
        Returns:
            pd.DataFrame: DataFrame с транзакциями.
        """
        with self._get_connection() as conn:
            query = """
                SELECT 
                    trans_date_trans_time, amt, lat, long, city_pop,
                    merch_lat, merch_long, merchant, category, gender,
                    job, dob, is_fraud
                FROM transactions
                WHERE is_fraud IS NOT NULL
                ORDER BY trans_date_trans_time
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, conn)
            logger.info(f"Загружено {len(df)} транзакций для обучения")
            return df
    
    def count_transactions(self) -> dict:
        """
        Подсчёт количества транзакций.
        
        Returns:
            dict: Статистика по транзакциям.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM transactions")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = 1")
            fraud = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = 0")
            legit = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud IS NULL")
            unlabeled = cursor.fetchone()[0]
            
            return {
                "total": total,
                "fraud": fraud,
                "legitimate": legit,
                "unlabeled": unlabeled,
            }
    
    def clear_transactions(self):
        """Очистка всех транзакций из БД."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM transactions")
            logger.warning("Все транзакции удалены из БД")
    
    def get_recent_transactions(self, limit: int = 100) -> List[dict]:
        """Получение последних N транзакций."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM transactions
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]


# Глобальный экземпляр БД (singleton)
_database: Optional[TransactionDatabase] = None


def get_database() -> TransactionDatabase:
    """
    Получение глобального экземпляра базы данных.
    
    Returns:
        TransactionDatabase: Экземпляр БД.
    """
    global _database
    if _database is None:
        _database = TransactionDatabase()
    return _database
