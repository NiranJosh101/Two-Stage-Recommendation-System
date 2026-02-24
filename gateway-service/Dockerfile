# --- Stage 1: The Builder ---
FROM python:3.11-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# We need fastapi, uvicorn, redis-py, and httpx (for async HTTP calls)
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# --- Stage 2: The Final Runtime ---
FROM python:3.11-slim
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
WORKDIR /app

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

COPY ./src ./src
RUN chown -R appuser:appgroup /app
USER appuser

# Gateway usually sits on the standard port 80 or 8000
EXPOSE 8000

# Service URLs for internal Docker networking
ENV USER_EMBEDDING_URL="http://user-embedding:8000"
ENV RETRIEVAL_URL="http://retrieval:8001"
ENV RANKING_URL="http://ranking:8002"
ENV REDIS_HOST="redis-cache"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]