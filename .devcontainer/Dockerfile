FROM mcr.microsoft.com/devcontainers/python:0-3.9

# Установка дополнительных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов проекта
COPY . /workspace
WORKDIR /workspace

# Установка Python зависимостей (если есть)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt