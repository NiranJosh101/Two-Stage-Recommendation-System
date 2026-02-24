# --- Stage 1: The Builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

# We need fewer build tools here, but still need them for the Pinecone client
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install FastAPI and the Pinecone client
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt


# --- Stage 2: The Final Runtime ---
FROM python:3.11-slim

# No heavy ML libs like PyTorch needed here, keeping the image tiny!
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
WORKDIR /app

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

COPY ./src ./src

RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8001

# Connection variables for the "neighbors"
ENV PINECONE_API_KEY=""
ENV PINECONE_ENVIRONMENT=""
ENV RANKING_SERVICE_URL="http://ranking-service:8002"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]