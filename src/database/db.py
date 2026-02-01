"""
SQLAlchemy база данных для хранения транзакций.

Обеспечивает:
- Хранение транзакций для обучения
- Загрузка данных из CSV в БД
- Выгрузка данных для обучения
"""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    create_engine,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.common import get_logger, settings

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Базовый класс для всех моделей."""
    pass


class Transaction(Base):
    """Модель транзакции."""

    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trans_date_trans_time = Column(String, nullable=False)
    amt = Column(Float, nullable=False)
    lat = Column(Float, nullable=False)
    long = Column(Float, nullable=False)
    city_pop = Column(Integer, nullable=False)
    merch_lat = Column(Float, nullable=False)
    merch_long = Column(Float, nullable=False)
    merchant = Column(String, nullable=False)
    category = Column(String, nullable=False)
    gender = Column(String, nullable=False)
    job = Column(String, nullable=False)
    dob = Column(String, nullable=False)
    is_fraud = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_trans_date", "trans_date_trans_time"),
        Index("idx_is_fraud", "is_fraud"),
    )

    def to_dict(self) -> dict:
        """Конвертация в словарь."""
        return {
            "id": self.id,
            "trans_date_trans_time": self.trans_date_trans_time,
            "amt": self.amt,
            "lat": self.lat,
            "long": self.long,
            "city_pop": self.city_pop,
            "merch_lat": self.merch_lat,
            "merch_long": self.merch_long,
            "merchant": self.merchant,
            "category": self.category,
            "gender": self.gender,
            "job": self.job,
            "dob": self.dob,
            "is_fraud": self.is_fraud,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TransactionDatabase:
    """SQLAlchemy база данных для хранения транзакций."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Инициализация базы данных.

        Args:
            db_path: Путь к файлу БД. По умолчанию: data/transactions.db
        """
        self.db_path = db_path or (settings.data_path / "transactions.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

        self._init_db()
        logger.info(f"База данных инициализирована: {self.db_path}")

    @contextmanager
    def _get_session(self):
        """Контекстный менеджер для сессии БД."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _init_db(self):
        """Создание схемы базы данных."""
        Base.metadata.create_all(self.engine)
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
        with self._get_session() as session:
            transaction = Transaction(
                trans_date_trans_time=trans_date_trans_time,
                amt=amt,
                lat=lat,
                long=long,
                city_pop=city_pop,
                merch_lat=merch_lat,
                merch_long=merch_long,
                merchant=merchant,
                category=category,
                gender=gender,
                job=job,
                dob=dob,
                is_fraud=is_fraud,
            )
            session.add(transaction)
            session.flush()
            return transaction.id

    def insert_many(self, transactions: List[dict]) -> int:
        """
        Пакетная вставка транзакций.

        Args:
            transactions: Список словарей с данными транзакций.

        Returns:
            int: Количество вставленных записей.
        """
        with self._get_session() as session:
            objects = [
                Transaction(
                    trans_date_trans_time=t.get("trans_date_trans_time"),
                    amt=t.get("amt"),
                    lat=t.get("lat"),
                    long=t.get("long"),
                    city_pop=t.get("city_pop"),
                    merch_lat=t.get("merch_lat"),
                    merch_long=t.get("merch_long"),
                    merchant=t.get("merchant"),
                    category=t.get("category"),
                    gender=t.get("gender"),
                    job=t.get("job"),
                    dob=t.get("dob"),
                    is_fraud=t.get("is_fraud"),
                )
                for t in transactions
            ]
            session.bulk_save_objects(objects)
            count = len(objects)
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

        required_columns = [
            "trans_date_trans_time", "amt", "lat", "long", "city_pop",
            "merch_lat", "merch_long", "merchant", "category", "gender",
            "job", "dob", "is_fraud"
        ]

        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Отсутствуют колонки в CSV: {missing}")

        if replace:
            self.clear_transactions()

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
        with self._get_session() as session:
            query = (
                session.query(Transaction)
                .filter(Transaction.is_fraud.isnot(None))
                .order_by(Transaction.trans_date_trans_time)
            )

            if limit:
                query = query.limit(limit)

            results = query.all()

            data = []
            for t in results:
                data.append({
                    "trans_date_trans_time": t.trans_date_trans_time,
                    "amt": t.amt,
                    "lat": t.lat,
                    "long": t.long,
                    "city_pop": t.city_pop,
                    "merch_lat": t.merch_lat,
                    "merch_long": t.merch_long,
                    "merchant": t.merchant,
                    "category": t.category,
                    "gender": t.gender,
                    "job": t.job,
                    "dob": t.dob,
                    "is_fraud": t.is_fraud,
                })

            df = pd.DataFrame(data)
            logger.info(f"Загружено {len(df)} транзакций для обучения")
            return df

    def count_transactions(self) -> dict:
        """
        Подсчёт количества транзакций.

        Returns:
            dict: Статистика по транзакциям.
        """
        with self._get_session() as session:
            total = session.query(func.count(Transaction.id)).scalar()
            fraud = session.query(func.count(Transaction.id)).filter(
                Transaction.is_fraud == 1
            ).scalar()
            legit = session.query(func.count(Transaction.id)).filter(
                Transaction.is_fraud == 0
            ).scalar()
            unlabeled = session.query(func.count(Transaction.id)).filter(
                Transaction.is_fraud.is_(None)
            ).scalar()

            return {
                "total": total,
                "fraud": fraud,
                "legitimate": legit,
                "unlabeled": unlabeled,
            }

    def clear_transactions(self):
        """Очистка всех транзакций из БД."""
        with self._get_session() as session:
            session.query(Transaction).delete()
            logger.warning("Все транзакции удалены из БД")

    def get_recent_transactions(self, limit: int = 100) -> List[dict]:
        """Получение последних N транзакций."""
        with self._get_session() as session:
            results = (
                session.query(Transaction)
                .order_by(Transaction.created_at.desc())
                .limit(limit)
                .all()
            )
            return [t.to_dict() for t in results]


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
