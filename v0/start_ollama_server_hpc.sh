#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-qwen3.6:35b-a3b-mtp-q4_K_M}"
OLLAMA_BASE="${OLLAMA_BASE:-/scratch1/$USER/ollama-gpu-oasis}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-$OLLAMA_BASE/models}"
OLLAMA_SIF="${OLLAMA_SIF:-$OLLAMA_BASE/ollama_latest.sif}"
OLLAMA_SERVER_LOG="${OLLAMA_SERVER_LOG:-$OLLAMA_BASE/ollama-server-interactive.log}"
OLLAMA_CONNECTION_ENV="${OLLAMA_CONNECTION_ENV:-$OLLAMA_BASE/ollama_connection.env}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch1/$USER/apptainer-cache}"

OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-16}"
OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
OLLAMA_MAX_QUEUE="${OLLAMA_MAX_QUEUE:-512}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
OLLAMA_SCHED_SPREAD="${OLLAMA_SCHED_SPREAD:-true}"
OLLAMA_TEST_TIMEOUT="${OLLAMA_TEST_TIMEOUT:-600}"
OLLAMA_TEST_PREDICT="${OLLAMA_TEST_PREDICT:-8}"

mkdir -p "$OLLAMA_BASE" "$OLLAMA_MODELS_DIR" "$APPTAINER_CACHEDIR"

if command -v module >/dev/null 2>&1; then
    module load apptainer || true
fi

if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is not available. Load the apptainer module before running this script."
    exit 1
fi

if [ ! -f "$OLLAMA_SIF" ]; then
    echo "Pulling Ollama Apptainer image to $OLLAMA_SIF"
    apptainer pull "$OLLAMA_SIF" docker://ollama/ollama:latest
fi

export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_BASE_URL="http://$OLLAMA_HOST"
export OLLAMA_OPENAI_URL="$OLLAMA_BASE_URL/v1"

export APPTAINERENV_OLLAMA_HOST="$OLLAMA_HOST"
export APPTAINERENV_OLLAMA_MODELS="/models"
export APPTAINERENV_OLLAMA_NUM_PARALLEL="$OLLAMA_NUM_PARALLEL"
export APPTAINERENV_OLLAMA_MAX_LOADED_MODELS="$OLLAMA_MAX_LOADED_MODELS"
export APPTAINERENV_OLLAMA_MAX_QUEUE="$OLLAMA_MAX_QUEUE"
export APPTAINERENV_OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE"
export APPTAINERENV_OLLAMA_SCHED_SPREAD="$OLLAMA_SCHED_SPREAD"
export APPTAINER_CACHEDIR

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
else
    export APPTAINERENV_CUDA_VISIBLE_DEVICES=0
fi

APPTAINER_OLLAMA=(
    apptainer exec
    --nv
    --bind "$OLLAMA_MODELS_DIR:/models"
    "$OLLAMA_SIF"
)

cleanup() {
    echo "Stopping Ollama..."
    if [ -n "${OLLAMA_PID:-}" ]; then
        kill "$OLLAMA_PID" 2>/dev/null || true
        wait "$OLLAMA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

check_ollama_ready() {
    python - "$OLLAMA_BASE_URL/api/tags" <<'PY'
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

try:
    with urlopen(sys.argv[1], timeout=5) as response:
        sys.exit(0 if 200 <= response.status < 500 else 1)
except (HTTPError, URLError, TimeoutError, OSError):
    sys.exit(1)
PY
}

echo "Starting Ollama server on $OLLAMA_HOST"
echo "Server log: $OLLAMA_SERVER_LOG"

"${APPTAINER_OLLAMA[@]}" ollama serve > "$OLLAMA_SERVER_LOG" 2>&1 &
OLLAMA_PID=$!

echo "Waiting for Ollama to become ready..."
OLLAMA_READY=0
for i in {1..180}; do
    if check_ollama_ready; then
        OLLAMA_READY=1
        break
    fi

    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "Ollama server died."
        tail -n 120 "$OLLAMA_SERVER_LOG" || true
        exit 1
    fi

    if [ "$((i % 15))" -eq 0 ]; then
        echo "Still waiting after $((i * 2)) seconds..."
        tail -n 20 "$OLLAMA_SERVER_LOG" || true
    fi
    sleep 2
done

if [ "$OLLAMA_READY" -ne 1 ]; then
    echo "Ollama server did not become ready in time."
    tail -n 120 "$OLLAMA_SERVER_LOG" || true
    exit 1
fi

echo "Ollama is ready."
echo "Pulling model if missing: $MODEL"
"${APPTAINER_OLLAMA[@]}" ollama pull "$MODEL"

cat > "$OLLAMA_CONNECTION_ENV" <<EOF
export OLLAMA_HOST="$OLLAMA_HOST"
export OLLAMA_BASE_URL="$OLLAMA_BASE_URL"
export OLLAMA_OPENAI_URL="$OLLAMA_OPENAI_URL"
export OLLAMA_MODEL="$MODEL"
export MODEL="$MODEL"
EOF

echo "Wrote connection file: $OLLAMA_CONNECTION_ENV"
echo "Use it from the same node with:"
echo "  source $OLLAMA_CONNECTION_ENV"
echo
echo "Testing Ollama API..."
python - "$OLLAMA_BASE_URL/api/generate" "$MODEL" "$OLLAMA_TEST_TIMEOUT" "$OLLAMA_TEST_PREDICT" <<'PY'
import json
import sys
from urllib.request import Request, urlopen

url, model, timeout, num_predict = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
payload = json.dumps(
    {
        "model": model,
        "prompt": "Reply with exactly: OLLAMA_OK",
        "stream": False,
        "options": {"num_predict": num_predict},
    }
).encode("utf-8")
request = Request(url, data=payload, headers={"Content-Type": "application/json"})
with urlopen(request, timeout=timeout) as response:
    data = json.load(response)
print(data.get("response", data))
PY

echo
echo "Ollama server is running. Keep this terminal/session open."
echo "Press Ctrl-C to stop it."
wait "$OLLAMA_PID"
source /scratch1/$USER/ollama-gpu-oasis/ollama_connection.env