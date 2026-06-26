# Используем официальный образ Python
FROM python:3.11-slim

# Указываем рабочую директорию
WORKDIR /app

# Копируем все файлы проекта внутрь
COPY . /app

# Устанавливаем библиотеки
RUN pip install --no-cache-dir -r requirements.txt

# Команда для запуска сервера (Hugging Face ожидает порт 7860)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]