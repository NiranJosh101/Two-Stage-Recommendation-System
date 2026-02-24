FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools for deep learning libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install FastAPI, Uvicorn, and your DL Framework (PyTorch/TF)
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt


# --- Stage 2: The Final Runtime ---
FROM python:3.11-slim

# libgomp1 is vital for OpenMP (used by PyTorch/XGBoost for speed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appgroup && useradd -r -g appgroup appuser
WORKDIR /app

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy source code
COPY ./src ./src

# Create a directory for the User-Tower model weights
RUN mkdir -p /app/models/user_tower && \
    chown -R appuser:appgroup /app

USER appuser

# Expose the port FastAPI will run on
EXPOSE 8000

# Environment variables
ENV MODEL_PATH="/app/models/user_tower"
ENV PYTHONUNBUFFERED=1

# Start FastAPI using Uvicorn
# --host 0.0.0.0 is mandatory for Docker to route traffic properly
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]