#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="infops"
HOST_PORT="8888"
CONTAINER_PORT="8888"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Current script directory: ${SCRIPT_DIR}"

rm -f "${SCRIPT_DIR}/token.txt"

docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}"

docker run --rm -it \
  --gpus=all \
  -p 127.0.0.1:${HOST_PORT}:${CONTAINER_PORT} \
  -v "${SCRIPT_DIR}:/app" \
  -w /app \
  "${IMAGE_NAME}" \
  bash -lc '
    set -euo pipefail

    mkdir -p /content

    python -m pip install --upgrade notebook jupyter_http_over_ws

    jupyter serverextension enable --py jupyter_http_over_ws --sys-prefix || true

    jupyter notebook \
      --ip=0.0.0.0 \
      --port=8888 \
      --no-browser \
      --allow-root \
      --notebook-dir=/app \
      --NotebookApp.allow_origin="https://colab.research.google.com" \
      --NotebookApp.allow_credentials=True \
      --NotebookApp.port_retries=0
  '