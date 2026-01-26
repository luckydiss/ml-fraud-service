<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/luckydiss/ml_fraude_service">
    <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/gnuprivacyguard.svg" alt="Logo" width="150" height="150">
  </a>

  <h3 align="center">Credit Fraud Detection ML Service</h3>

  <p align="center">
    ML-сервис для детекции мошеннических транзакций по кредитным картам в реальном времени
    <br />
    <a href="#about-the-project"><strong>Исследовать документацию »</strong></a>
    <br />
    <br />
    <a href="#usage">Демо</a>
    ·
    <a href="https://github.com/luckydiss/ml_fraude_service/issues">Сообщить об ошибке</a>
    ·
    <a href="https://github.com/luckydiss/ml_fraude_service/issues">Предложить улучшение</a>
  </p>
</p>

<!-- BADGES -->
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.109+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/LightGBM-4.0+-yellow.svg" alt="LightGBM">
  <img src="https://img.shields.io/badge/SQLite-Database-orange.svg" alt="SQLite">
  <img src="https://img.shields.io/badge/Docker-Compose-blue.svg" alt="Docker">
  <img src="https://img.shields.io/badge/License-MIT-purple.svg" alt="License">
</p>

---

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><b>Содержание</b></summary>
  <ol>
    <li>
      <a href="#about-the-project">О проекте</a>
      <ul>
        <li><a href="#architecture">Архитектура</a></li>
        <li><a href="#built-with">Технологический стек</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Начало работы</a>
      <ul>
        <li><a href="#dependencies">Зависимости</a></li>
        <li><a href="#installation">Установка</a></li>
        <li><a href="#configuration">Конфигурация</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Использование</a>
      <ul>
        <li><a href="#running-services">Запуск сервисов</a></li>
        <li><a href="#api-reference">API Reference</a></li>
      </ul>
    </li>
    <li><a href="#project-structure">Структура проекта</a></li>
    <li><a href="#testing">Тестирование</a></li>
    <li><a href="#docker">Docker</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Вклад в проект</a></li>
    <li><a href="#license">Лицензия</a></li>
    <li><a href="#authors">Авторы</a></li>
    <li><a href="#acknowledgements">Благодарности</a></li>
  </ol>
</details>

---

<!-- ABOUT THE PROJECT -->
## About The Project

**Credit Fraud Detection ML Service** — это production-ready ML-система для детекции мошеннических транзакций по кредитным картам в реальном времени.

### Что делает сервис?

Сервис предоставляет полный ML-пайплайн через REST API:

| Компонент | Описание |
|-----------|----------|
| **Inference Endpoint** | Предсказание фрода для входящих транзакций |
| **Train Endpoint** | Обучение новых моделей через HTTP API |
| **Hotswap Endpoint** | Горячая замена модели без остановки сервиса |
| **Database** | SQLite хранилище для транзакций |
| **Preprocessing** | Единый пайплайн обработки данных |

---

<!-- ARCHITECTURE -->
### Architecture

<p align="center">
    <img src="architecture.png" alt="Logo" width="800" height="500">
  </a>
```

### Потоки данных

| Поток | Описание |
|-------|----------|
| **Prediction** | `Request → Inference endpoint → Preprocessing → Inference → Response` |
| **Training** | `Request → Train endpoint → Database → Preprocessing → Training → AI Model` |
| **Hotswap** | `Request → Hotswap endpoint → Load new model → Inference updated` |

---

<!-- BUILT WITH -->
### Built With

| Категория | Технологии |
|-----------|------------|
| **ML/DS** | scikit-learn, LightGBM, pandas, NumPy |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Database** | SQLite (встроенная) |
| **Package Manager** | uv (Astral) |
| **Containerization** | Docker, Docker Compose |
| **Configuration** | pydantic-settings, python-dotenv |
| **Testing** | pytest, pytest-asyncio, pytest-cov |
| **Linting** | Ruff |

---

<!-- GETTING STARTED -->
## Getting Started

### Dependencies

* **Python 3.10+**
* **uv** (рекомендуемый менеджер пакетов)
  ```sh
  # Windows (PowerShell)
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  
  # Linux/macOS
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* **Docker & Docker Compose** (опционально)

---

### Installation

1. **Клонируйте репозиторий**
   ```sh
   git clone https://github.com/luckydiss/ml_fraude_service.git
   cd ml_fraude_service
   ```

2. **Скопируйте файл окружения**
   ```sh
   cp .env.example .env
   ```

3. **Установите зависимости**
   ```sh
   uv sync
   ```

4. **Запустите сервис**
   ```sh
   uv run serve
   ```

Сервис доступен: http://localhost:8000

Swagger UI: http://localhost:8000/docs

---

### Configuration

Все настройки через переменные окружения (файл `.env`):

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `APP_ENV` | `development` | Окружение |
| `DEBUG` | `false` | Режим отладки |
| `MODEL_REGISTRY_PATH` | `./model_registry` | Путь к моделям |
| `DATA_PATH` | `./data` | Путь к данным |
| `INFERENCE_HOST` | `0.0.0.0` | Хост API |
| `INFERENCE_PORT` | `8000` | Порт API |
| `OPTIMAL_THRESHOLD` | `0.251` | Порог классификации |
| `MIN_PRECISION` | `0.80` | Quality Gate: precision |
| `MIN_RECALL` | `0.70` | Quality Gate: recall |
| `MIN_F1` | `0.75` | Quality Gate: F1 |

---

<!-- USAGE -->
## Usage

### Running Services

```sh
# Запуск API сервиса
uv run serve

# Обучение модели через CLI (альтернативный способ)
uv run train
```

---

### API Reference

#### Endpoints Overview

| Метод | Endpoint | Тип | Описание |
|-------|----------|-----|----------|
| `POST` | `/predict` | **Inference** | Предсказание фрода |
| `POST` | `/train` | **Training** | Обучение модели |
| `POST` | `/reload` | **Hotswap** | Горячая замена модели |
| `GET` | `/database/stats` | **Database** | Статистика БД |
| `POST` | `/database/load` | **Database** | Загрузка CSV в БД |
| `GET` | `/health` | **Health** | Состояние сервиса |
| `GET` | `/model/info` | **Health** | Информация о модели |

---

#### POST /predict — Inference Endpoint

Предсказание вероятности фрода для входящей транзакции.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "dob": "1985-06-20"
  }'
```

**Response:**
```json
{
  "is_fraud": false,
  "fraud_probability": 0.0234,
  "threshold_used": 0.251,
  "model_version": "v_20260126_120000"
}
```

---

#### POST /train — Train Endpoint

Запуск обучения новой модели через API.

**Request:**
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "use_database": true,
    "force_register": false,
    "skip_reload": false
  }'
```

**Параметры:**

| Параметр | Тип | Описание |
|----------|-----|----------|
| `use_database` | bool | Использовать данные из БД (true) или CSV (false) |
| `force_register` | bool | Зарегистрировать модель даже если не прошла quality gates |
| `skip_reload` | bool | Пропустить автоматическую перезагрузку модели |

**Response:**
```json
{
  "success": true,
  "version": "v_20260126_120000",
  "metrics": {
    "accuracy": 0.9987,
    "precision": 0.8234,
    "recall": 0.7512,
    "f1": 0.7857,
    "roc_auc": 0.9856
  },
  "validation_passed": true,
  "reload_triggered": true,
  "message": "Модель успешно обучена и зарегистрирована"
}
```

---

#### POST /reload — Hotswap Endpoint

Горячая замена модели без остановки сервиса.

**Request:**
```bash
curl -X POST "http://localhost:8000/reload"
```

**Response:**
```json
{
  "success": true,
  "previous_version": "v_20260125_191328",
  "new_version": "v_20260126_120000",
  "message": "Модель успешно перезагружена"
}
```

---

#### Database Endpoints

**GET /database/stats** — Статистика транзакций в БД:
```bash
curl http://localhost:8000/database/stats
```

```json
{
  "total": 1000000,
  "fraud": 5000,
  "legitimate": 995000,
  "unlabeled": 0
}
```

**POST /database/load** — Загрузка данных из CSV:
```bash
curl -X POST "http://localhost:8000/database/load?replace=false"
```

---

#### GET /health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v_20260126_120000",
  "timestamp": "2026-01-26T12:30:00.123456"
}
```

---

<!-- TESTING -->
## Testing

```sh
# Все тесты
uv run pytest

# С coverage
uv run pytest --cov=src --cov-report=html

# Конкретный модуль
uv run pytest tests/unit/test_api.py -v
```

---

<!-- DOCKER -->
## Docker

```sh
# Запуск сервиса
docker-compose up -d inference

# Обучение
docker-compose --profile training up training

# Логи
docker-compose logs -f inference
```

---


<!-- LICENSE -->
## License

Распространяется под лицензией MIT. См. файл `LICENSE`.

---

<!-- AUTHORS -->
## Authors

**luckydiss** — Разработчик

Ссылка на проект: [https://github.com/luckydiss/ml_fraude_service](https://github.com/luckydiss/ml_fraude_service)

---"# ml-fraud-service" 
