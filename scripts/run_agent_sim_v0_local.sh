#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-agent-sim-v0}"
MODEL_NAME="${OLLAMA_MODEL:-llama3.2:3b}"
USER_COUNT="${USER_COUNT:-10}"
MIN_TWEETS="${MIN_TWEETS:-10}"
MAX_TWEETS_PER_USER="${MAX_TWEETS_PER_USER:-10}"
SELECTION="${SELECTION:-top}"
USE_GPU="${USE_GPU:-auto}"

DATA_DIR="${REPO_DIR}/agent_sim_v0/data"
OUTPUT_DIR="${REPO_DIR}/agent_sim_v0/outputs"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or is not on PATH." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running. Start Docker Desktop, then try again." >&2
  exit 1
fi

if [[ ! -f "${DATA_DIR}/dummi_tweets_brazil.csv" ]]; then
  echo "Missing input file: ${DATA_DIR}/dummi_tweets_brazil.csv" >&2
  exit 1
fi

mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"

cd "${REPO_DIR}"

echo "Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" -f docker/ollama-agent/Dockerfile .

echo "Running Agent Sim V0 locally"
echo "Model: ${MODEL_NAME}"
echo "Users: ${USER_COUNT}; random tweets per user: ${MAX_TWEETS_PER_USER}; selection: ${SELECTION}"

DOCKER_RUN_ARGS=(--rm -it)
if [[ "${USE_GPU}" == "1" || "${USE_GPU}" == "true" ]]; then
  DOCKER_RUN_ARGS+=(--gpus all)
elif [[ "${USE_GPU}" == "auto" ]] && docker info 2>/dev/null | grep -qi "nvidia"; then
  DOCKER_RUN_ARGS+=(--gpus all)
fi

DOCKER_RUN_ARGS+=(
  -e "OLLAMA_MODEL=${MODEL_NAME}"
  -v "${DATA_DIR}:/app/agent_sim_v0/data"
  -v "${OUTPUT_DIR}:/app/agent_sim_v0/outputs"
)

docker run "${DOCKER_RUN_ARGS[@]}" "${IMAGE_NAME}" bash -lc '
    set -euo pipefail

    echo "Running simulation with personas generated directly from tweet samples"
    python -m agent_sim_v0.src.main \
      --config agent_sim_v0/configs/tweets_ollama.yaml \
      --model "${OLLAMA_MODEL}" \
      --user-count "'"${USER_COUNT}"'" \
      --min-tweets "'"${MIN_TWEETS}"'" \
      --max-tweets-per-user "'"${MAX_TWEETS_PER_USER}"'" \
      --selection "'"${SELECTION}"'"
  '

echo "Done. Outputs are in: ${OUTPUT_DIR}"
