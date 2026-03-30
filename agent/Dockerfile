FROM python:3.12-slim

ARG AGENT_VERSION=unknown
ENV AGENT_VERSION=${AGENT_VERSION}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY entrypoint.sh .

EXPOSE 8090

CMD ["/bin/sh", "entrypoint.sh"]
