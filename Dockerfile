# 1. Use Python 3.11 for maximum AI library compatibility
FROM python:3.11-slim

# 2. Install FFmpeg AND libsndfile (Required for librosa to work)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the directory inside the container
WORKDIR /app

# 4. Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your code
COPY . .

# 6. Start the server using the PORT Render provides
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]