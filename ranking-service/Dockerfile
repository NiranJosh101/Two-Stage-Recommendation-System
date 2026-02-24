# --- Stage 1: The Builder ---
FROM python:3.11-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# --- Stage 2: The Final Runtime ---
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libpq5 && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appgroup && useradd -r -g appgroup appuser
WORKDIR /app

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

COPY ./src ./src
RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8002

# --- MLflow Registry Config ---
# These will be filled by your docker-compose or K8s secrets
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_MODEL_NAME="job-ranking-xgboost"
ENV MLFLOW_MODEL_STAGE="Production" 

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002"]