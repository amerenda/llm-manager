FROM python:3.12-slim

# Non-root user
RUN useradd -m -u 1000 moltbook

WORKDIR /app

# Install system deps for asyncpg (libpq) and general utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install backend deps
COPY moltbook-backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy moltbook-backend code
COPY moltbook-backend/ .

RUN mkdir -p /data/agents && chown -R moltbook:moltbook /data /app

USER moltbook

EXPOSE 8082
CMD ["python", "main.py"]
