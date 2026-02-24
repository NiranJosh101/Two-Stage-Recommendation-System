# --- Stage 1: The Builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

# build-essential for vector math libraries and DB clients
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies (Transformers, PyTorch/TF, VectorDB clients like qdrant-client or pymilvus)
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt


# --- Stage 2: The Final Runtime ---
FROM python:3.11-slim

# libgomp1 is often needed for the underlying math libraries in embeddings
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appgroup && useradd -r -g appgroup appuser
WORKDIR /app

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

COPY ./src ./src

# Create directories for the model and the data source
RUN mkdir -p /app/models /app/data/feature_store && \
    chown -R appuser:appgroup /app

USER appuser

# Environment variables for connecting to the Vector DB
ENV VECTOR_DB_URL="http://qdrant:6333"
ENV MODEL_PATH="/app/models/item_tower"

# This service likely runs a script to "refresh" embeddings
CMD ["python", "src/generate_embeddings.py"]