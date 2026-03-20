FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install piper binary (arm64 / aarch64)
RUN wget -q -O /tmp/piper.tar.gz \
    https://github.com/rhasspy/piper/releases/latest/download/piper_linux_aarch64.tar.gz \
    && tar -xzf /tmp/piper.tar.gz -C /tmp \
    && mv /tmp/piper/piper /usr/local/bin/piper \
    && chmod +x /usr/local/bin/piper \
    && rm -rf /tmp/piper /tmp/piper.tar.gz

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY main.py .

# Voices will be provided via PVC
RUN mkdir -p /voices

EXPOSE 8091

CMD ["python", "main.py"]
