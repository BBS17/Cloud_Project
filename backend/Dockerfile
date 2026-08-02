# Use lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files + enable stdout logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker layer caching)
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy application code and DB init script
COPY app ./app
COPY init_db.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# NOTE:
# misinformation_model is already inside app/,
# so it is included automatically in the COPY above.
# No need to copy it separately.

# Expose port (Cloud Run / containers expect 8080)
EXPOSE 8080

# Run DB init then start the API
ENTRYPOINT ["./entrypoint.sh"]