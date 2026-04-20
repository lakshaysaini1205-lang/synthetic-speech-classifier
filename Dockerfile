# Use the slim version of Python 3.14
FROM python:3.14-slim

# Install FFmpeg (the decoder) and necessary system libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements first to cache them (makes builds faster)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files into the container
COPY . .

# Start the app using Gunicorn on the port Render provides
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]