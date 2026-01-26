FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Установка системных зависимостей (нужны для LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем конфиги зависимостей
COPY pyproject.toml uv.lock ./

# Устанавливаем зависимости (без самого проекта для кэширования)
RUN uv sync --frozen --no-install-project --no-dev

# Копируем исходный код и README
COPY src/ ./src/
COPY README.md ./

# Пересинхронизируем, чтобы установить сам проект
RUN uv sync --frozen --no-dev

# Создаем директорию для реестра моделей
RUN mkdir -p /app/model_registry

# Выставляем PYTHONPATH, чтобы модули src были видны
ENV PYTHONPATH=/app

# Порт по умолчанию
EXPOSE 8000

# Запуск inference сервиса
CMD ["uv", "run", "serve"]
