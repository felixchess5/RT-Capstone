FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OCR and document tooling used by backend processors
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-core.txt ./
RUN pip install --upgrade pip && pip install -r requirements-core.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "--app-dir", "src", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
