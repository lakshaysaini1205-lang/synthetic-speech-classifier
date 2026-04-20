# 1. Use a stable Python version
FROM python:3.11-slim

# 2. Install audio and build tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your code and model (145KB)
COPY . .

# 5. The Fix: Run in Shell Form so $PORT works!
CMD gunicorn --bind 0.0.0.0:$PORT app:app