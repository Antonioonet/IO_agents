FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV OLLAMA_HOST=127.0.0.1:11434
ENV OLLAMA_MODEL=llama3.2:3b
ENV SIM_CONFIG=agent_sim_v0/configs/local_ollama.yaml

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY agent_sim_v0/requirements.txt /tmp/agent_sim_v0_requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/agent_sim_v0_requirements.txt

COPY agent_sim_v0 ./agent_sim_v0
COPY docker/ollama-agent/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "agent_sim_v0.src.main"]
