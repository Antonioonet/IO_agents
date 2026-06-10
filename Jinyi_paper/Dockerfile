FROM nvcr.io/nvidia/pytorch:26.03-py3

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv .venv

RUN .venv/bin/pip install --upgrade pip setuptools wheel && \
    .venv/bin/pip install --no-build-isolation -r requirements.txt

COPY . .

CMD ["/bin/bash"]
