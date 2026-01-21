# ===== Base image =====
FROM python:3.10-slim

# ===== Set working directory =====
WORKDIR /app

# ===== Install system dependencies =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ===== STEP 1: Install Large/Stable Dependencies (HEAVY CACHE) =====
# Install Torch first using its specific index
RUN pip install \
    torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install other heavy tools from default PyPI (this avoids the "No matching distribution" error)
RUN pip install --no-cache-dir \
    mlflow \
    scikit-learn \
    kfp

# ===== STEP 2: Install remaining Python packages (LIGHTWEIGHT) =====
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===== STEP 3: Copy project code =====
# Ensure you run the build command from the root folder so 'src/' is visible
COPY src/ /app/src/

# ===== Set Environment Variables =====
ENV PYTHONPATH="/app"

# ===== Set Entrypoint =====
ENTRYPOINT ["python", "-m", "src.two_tower_training.components.train_two_tower.two_tower_component"]