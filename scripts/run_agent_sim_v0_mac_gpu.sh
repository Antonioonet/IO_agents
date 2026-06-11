#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME="${OLLAMA_MODEL:-llama3.2:3b}"
USER_COUNT="${USER_COUNT:-10}"
MIN_TWEETS="${MIN_TWEETS:-10}"
MAX_TWEETS_PER_USER="${MAX_TWEETS_PER_USER:-10}"
SELECTION="${SELECTION:-top}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv-agent-sim-v0}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"

DATA_DIR="${REPO_DIR}/agent_sim_v0/data"
OUTPUT_DIR="${REPO_DIR}/agent_sim_v0/outputs"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is intended for macOS. Use scripts/run_agent_sim_v0_local.sh for Docker/Linux." >&2
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Install it from https://ollama.com/download, then try again." >&2
  exit 1
fi

if [[ ! -f "${DATA_DIR}/dummi_tweets_brazil.csv" ]]; then
  echo "Missing input file: ${DATA_DIR}/dummi_tweets_brazil.csv" >&2
  exit 1
fi

mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"

cd "${REPO_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating Python environment: ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

echo "Installing Python dependencies"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r agent_sim_v0/requirements.txt

if ! curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
  echo "Starting native Ollama on macOS. This is the path that can use Apple Silicon GPU/Metal."
  ollama serve > "${OUTPUT_DIR}/ollama_mac.log" 2>&1 &
  OLLAMA_PID="$!"
  cleanup() {
    kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT

  until curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; do
    sleep 1
  done
else
  echo "Using already-running native Ollama server."
fi

echo "Ensuring model is available: ${MODEL_NAME}"
ollama pull "${MODEL_NAME}"

echo "Running simulation with personas generated directly from tweet samples"
"${VENV_DIR}/bin/python" -m agent_sim_v0.src.main \
  --config agent_sim_v0/configs/tweets_ollama.yaml \
  --model "${MODEL_NAME}" \
  --base-url "${OLLAMA_BASE_URL}" \
  --user-count "${USER_COUNT}" \
  --min-tweets "${MIN_TWEETS}" \
  --max-tweets-per-user "${MAX_TWEETS_PER_USER}" \
  --selection "${SELECTION}"

echo "Done. Outputs are in: ${OUTPUT_DIR}"
