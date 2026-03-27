FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY refrag.py .
COPY app.py .
COPY train.py .
COPY data/ data/

# Environment
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

# Default entrypoint is the CLI; override for the UI
ENTRYPOINT ["python", "refrag.py"]
