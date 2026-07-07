FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AGENT_CONFIG=/app/config.docker.toml

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY agent ./agent
COPY SOUL.md README.md config.example.toml config.docker.toml ./

RUN mkdir -p /app/data

EXPOSE 8787

CMD ["python", "-m", "agent.main"]
