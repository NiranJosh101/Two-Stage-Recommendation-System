# 1. Use an official PyTorch base image with CUDA support
# This ensures all drivers and C++ extensions are pre-configured
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 2. Set the working directory
WORKDIR /app

# 3. Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Install system dependencies (needed for certain python libs or parquet handling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy the source code and install the local 'two_tower' package
# This allows 'import two_tower' to work anywhere in the container
COPY src/ /app/src/
COPY setup.py /app/
RUN pip install -e /app/src/

# 7. Copy the component entrypoint script and default config
COPY components/train_two_tower/component.py /app/component.py
COPY configs/two_tower.yaml /app/configs/two_tower.yaml

# 8. Set the entrypoint to our component script
# This allows the pipeline to pass arguments like --train-path directly
ENTRYPOINT ["python", "component.py"]