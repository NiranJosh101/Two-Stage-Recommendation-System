# ===== Base image =====
FROM python:3.10-slim

# ===== Set working directory =====
WORKDIR /app

# ===== Install system dependencies (needed for pyarrow, torch, etc.) =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ===== Install PyTorch CPU-only =====
RUN pip install --no-cache-dir torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu


# ===== Copy requirements and install other Python packages =====
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# ===== Copy project code =====
COPY src_retriever/ src_retriever/
COPY two_tower_config/ two_tower_config/
COPY components/train_two_tower/two_tower_component.py two_tower_component.py

# ===== Set entrypoint =====
ENTRYPOINT ["python", "two_tower_component.py"]
