FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and configs
COPY src/ /app/src/
COPY configs/ /app/configs/

# Add src to PYTHONPATH so we can import 'ranker'
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# We don't define an ENTRYPOINT here because Kubeflow 
# will pass the command via the component spec.