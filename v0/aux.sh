#!/bin/bash

#SBATCH --account=<your_project_id>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=ollama-gpu-oasis
#SBATCH --output=ollama-gpu-oasis-%j.out
#SBATCH --error=ollama-gpu-oasis-%j.err

set -euo pipefail

cd "$(dirname "$0")"

module purge
module load conda
module load apptainer

source "$(conda info --base)/etc/profile.d/conda.sh"

CONDA_ENV="${CONDA_ENV:-oasis}"
conda activate "$CONDA_ENV"

echo "Python environment:"
echo "  CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}"
echo "  python=$(command -v python)"
python - <<'PY'
import importlib.util
import site
import sys

print(f"  sys.executable={sys.executable}")
print(f"  sys.version={sys.version.split()[0]}")
print(f"  user_site={site.getusersitepackages()}")
print(f"  oasis_spec={importlib.util.find_spec('oasis')}")
PY

python - <<'PY'
import oasis

print(f"Imported oasis from: {getattr(oasis, '__file__', '<namespace package>')}")
PY

MODEL="${MODEL:-smollm2:135m}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-exp_$(date +%Y%m%d_%H%M%S)}"

export OLLAMA_BASE="/scratch1/$USER/ollama-gpu-oasis"
export OLLAMA_MODELS_DIR="$OLLAMA_BASE/models"
export OLLAMA_SIF="$OLLAMA_BASE/ollama_latest.sif"
export OLLAMA_SERVER_LOG="$OLLAMA_BASE/ollama-server-${SLURM_JOB_ID}.log"
export APPTAINER_CACHEDIR="/scratch1/$USER/apptainer-cache"

mkdir -p "$OLLAMA_BASE" "$OLLAMA_MODELS_DIR" "$APPTAINER_CACHEDIR"

if [ ! -f "$OLLAMA_SIF" ]; then
    apptainer pull "$OLLAMA_SIF" docker://ollama/ollama:latest
fi

# Local only: visible only inside this compute node.
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export OLLAMA_OPENAI_URL="$OLLAMA_BASE_URL/v1"

# Pass env vars into Apptainer.
export APPTAINERENV_OLLAMA_HOST="$OLLAMA_HOST"
export APPTAINERENV_OLLAMA_MODELS="/models"

# CPU threading hint for Python and Ollama.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

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
    fi
}
trap cleanup EXIT

echo "Starting Ollama GPU server..."

"${APPTAINER_OLLAMA[@]}" \
    ollama serve > "$OLLAMA_SERVER_LOG" 2>&1 &

OLLAMA_PID=$!

echo "Waiting for Ollama server..."

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

OLLAMA_READY=0
for i in {1..180}; do
    if check_ollama_ready; then
        echo "Ollama is ready."
        OLLAMA_READY=1
        break
    fi

    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "Ollama server died."
        echo "Check log: $OLLAMA_SERVER_LOG"
        tail -n 80 "$OLLAMA_SERVER_LOG" || true
        exit 1
    fi

    if [ "$((i % 15))" -eq 0 ]; then
        echo "Still waiting for Ollama after $((i * 2)) seconds..."
        tail -n 20 "$OLLAMA_SERVER_LOG" || true
    fi

    sleep 2
done

if [ "$OLLAMA_READY" -ne 1 ]; then
    echo "Ollama server did not become ready in time."
    echo "Check log: $OLLAMA_SERVER_LOG"
    tail -n 120 "$OLLAMA_SERVER_LOG" || true
    exit 1
fi

echo "Using model: $MODEL"
echo "Experiment name: $EXPERIMENT_NAME"
echo "GPU visibility from job:"
nvidia-smi || true

echo "Pulling model: $MODEL"

"${APPTAINER_OLLAMA[@]}" ollama pull "$MODEL"

echo "Testing Ollama API..."

python - "$OLLAMA_BASE_URL/api/generate" "$MODEL" <<'PY'
import json
import sys
from urllib.request import Request, urlopen

url, model = sys.argv[1], sys.argv[2]
payload = json.dumps(
    {"model": model, "prompt": "Reply with exactly: OLLAMA_OK", "stream": False}
).encode("utf-8")
request = Request(url, data=payload, headers={"Content-Type": "application/json"})
with urlopen(request, timeout=180) as response:
    print(json.dumps(json.load(response), indent=2))
PY

echo "Running Python simulation test..."

export OLLAMA_MODEL="$MODEL"

python generate_russia_personas.py \
    --model "$MODEL" \
    --ollama-url "$OLLAMA_BASE_URL" \
    --experiment-name "$EXPERIMENT_NAME"

python v0.py \
    --model "$MODEL" \
    --ollama-url "$OLLAMA_OPENAI_URL" \
    --experiment-name "$EXPERIMENT_NAME"

echo "GPU OASIS run finished successfully."
