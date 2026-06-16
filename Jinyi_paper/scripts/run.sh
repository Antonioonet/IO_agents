#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-jinyi-paper-jupyter}"
HOST_PORT="${HOST_PORT:-8888}"
CONTAINER_PORT="8888"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/arm64}"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-ioagents}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or is not on PATH." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running. Start Docker Desktop, then try again." >&2
  exit 1
fi

echo "Building ${IMAGE_NAME} for ${DOCKER_PLATFORM}"
docker build \
  --platform "${DOCKER_PLATFORM}" \
  -t "${IMAGE_NAME}" \
  -f "${PROJECT_DIR}/docker/Dockerfile" \
  "${PROJECT_DIR}"

echo "Starting Jupyter Lab"
echo "Open: http://127.0.0.1:${HOST_PORT}/lab?token=${JUPYTER_TOKEN}"

docker run --rm -it \
  --platform "${DOCKER_PLATFORM}" \
  -p "127.0.0.1:${HOST_PORT}:${CONTAINER_PORT}" \
  -v "${PROJECT_DIR}:/app" \
  -w /app \
  "${IMAGE_NAME}" \
  jupyter lab \
    --ip=0.0.0.0 \
    --port="${CONTAINER_PORT}" \
    --no-browser \
    --allow-root \
    --notebook-dir=/app \
    --ServerApp.token="${JUPYTER_TOKEN}" \
    --ServerApp.allow_origin="https://colab.research.google.com" \
    --ServerApp.allow_credentials=True \
    --ServerApp.port_retries=0
