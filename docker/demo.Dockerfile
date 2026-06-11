FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-demo.txt ./
RUN pip install --upgrade pip && pip install -r requirements-demo.txt

COPY . .

EXPOSE 7860

CMD ["python", "launch_gradio.py"]
