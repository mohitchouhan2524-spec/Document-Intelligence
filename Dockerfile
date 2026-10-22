FROM python:3.13.0-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN grep -Ev '^(torch|torchvision)==' requirements.txt > requirements.docker.txt \
    && pip install -r requirements.docker.txt \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.0+cpu torchvision==0.27.0+cpu

COPY . .

RUN useradd --create-home app \
    && mkdir -p /app/.cache/huggingface /app/evaluation/reports \
    && chown -R app:app /app

USER app

EXPOSE 8000 8501

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]