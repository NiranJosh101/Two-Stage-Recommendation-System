FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools and libraries for XGBoost and MLflow (psycopg2 for DB logging)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install heavy hitters (PyTorch/TF, XGBoost, MLflow) 
# Note: We use --no-cache-dir to keep the builder layer from exploding
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# --- Stage 2: The Final Runtime (The "Athlete") ---
FROM python:3.11-slim

# Install runtime-only libs (libgomp is required for XGBoost parallelization)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appgroup && useradd -r -g appgroup appuser
WORKDIR /app

# Copy and install the pre-compiled wheels from Stage 1
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy your training scripts
COPY ./src ./src

# Create local directories for the Feature Store (Read) and Model Artifacts (Write)
RUN mkdir -p /app/data/feature_store /app/models && \
    chown -R appuser:appgroup /app

USER appuser

# MLflow Environment Variables
ENV MLFLOW_TRACKING_URI="http://mlflow-server:5000"
ENV PYTHONUNBUFFERED=1

# Command to trigger your training pipeline
CMD ["python", "src/train_pipeline.py"]