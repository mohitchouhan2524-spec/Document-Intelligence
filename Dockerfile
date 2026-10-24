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

RUN sed -i 's/\r$//' start.sh

RUN python -c "from huggingface_hub import snapshot_download; from src.config import get_settings; c = get_settings(); p = ['*.json','*.safetensors','*.txt','*.model','*.jinja']; snapshot_download(c.embedding.local_model, allow_patterns=p); snapshot_download(c.retrieval.reranker_model, allow_patterns=p)" || true

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/.cache/huggingface /app/evaluation/reports \
    && chown -R app:app /app

USER app

EXPOSE 7860

CMD ["bash", "start.sh"]