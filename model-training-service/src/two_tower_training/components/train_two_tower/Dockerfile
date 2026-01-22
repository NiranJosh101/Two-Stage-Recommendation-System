FROM python:3.10-slim


WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install \
    torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu


RUN pip install --no-cache-dir \
    mlflow \
    scikit-learn \
    kfp


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY src/ /app/src/


ENV PYTHONPATH="/app"


ENTRYPOINT ["python", "-m", "src.two_tower_training.components.train_two_tower.two_tower_component"]