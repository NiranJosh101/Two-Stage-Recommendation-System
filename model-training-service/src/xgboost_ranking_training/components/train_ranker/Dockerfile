FROM python:3.9-slim

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY src/ /app/src/
COPY configs/ /app/configs/


ENV PYTHONPATH="/app/src:${PYTHONPATH}"

