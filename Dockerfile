FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p data chroma_db

# Expose API port
EXPOSE 8000

# Default command: run API server
CMD ["python", "-m", "app.main", "--api"]
