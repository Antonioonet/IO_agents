#!/bin/bash

#SBATCH --account=ll_774_951
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=ollama-gpu-oasis
#SBATCH --output=ollama-gpu-oasis-%j.out
#SBATCH --error=ollama-gpu-oasis-%j.err

set -euo pipefail

SCRIPT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "$SCRIPT_DIR/generate_russia_personas.py" ] && [ -d "$SCRIPT_DIR/IO_agents/v0" ]; then
    SCRIPT_DIR="$SCRIPT_DIR/IO_agents/v0"
fi
RUN_DIR="${RUN_DIR:-/scratch1/$USER/oasis-runs/${SLURM_JOB_ID:-manual}}"

if [ ! -f "$SCRIPT_DIR/generate_russia_personas.py" ] || [ ! -f "$SCRIPT_DIR/v0.py" ]; then
    echo "Could not find simulation scripts in: $SCRIPT_DIR"
    echo "Submit from IO_agents/v0, or set PROJECT_DIR=/path/to/IO_agents/v0 when calling sbatch."
    exit 1
fi

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "Script directory: $SCRIPT_DIR"
echo "Run directory: $RUN_DIR"

module purge
module load conda
module load apptainer

source "$(conda info --base)/etc/profile.d/conda.sh"

CONDA_ENV="${CONDA_ENV:-oasis}"
conda activate "$CONDA_ENV"
echo "$CONDA_ENV"
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

MODEL="${MODEL:-qwen3.6:35b-a3b}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-exp_$(date +%Y%m%d_%H%M%S)}"
OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-2}"
OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
OLLAMA_MAX_QUEUE="${OLLAMA_MAX_QUEUE:-512}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
OLLAMA_SCHED_SPREAD="${OLLAMA_SCHED_SPREAD:-true}"
PERSONA_WORKERS="${PERSONA_WORKERS:-$OLLAMA_NUM_PARALLEL}"
PERSONA_TIMEOUT="${PERSONA_TIMEOUT:-300}"
OLLAMA_TEST_TIMEOUT="${OLLAMA_TEST_TIMEOUT:-600}"

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
export APPTAINERENV_OLLAMA_NUM_PARALLEL="$OLLAMA_NUM_PARALLEL"
export APPTAINERENV_OLLAMA_MAX_LOADED_MODELS="$OLLAMA_MAX_LOADED_MODELS"
export APPTAINERENV_OLLAMA_MAX_QUEUE="$OLLAMA_MAX_QUEUE"
export APPTAINERENV_OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE"
export APPTAINERENV_OLLAMA_SCHED_SPREAD="$OLLAMA_SCHED_SPREAD"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

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
echo "Ollama concurrency:"
echo "  OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL"
echo "  OLLAMA_MAX_LOADED_MODELS=$OLLAMA_MAX_LOADED_MODELS"
echo "  OLLAMA_MAX_QUEUE=$OLLAMA_MAX_QUEUE"
echo "  OLLAMA_KEEP_ALIVE=$OLLAMA_KEEP_ALIVE"
echo "  OLLAMA_SCHED_SPREAD=$OLLAMA_SCHED_SPREAD"
echo "Python request concurrency:"
echo "  PERSONA_WORKERS=$PERSONA_WORKERS"
echo "GPU visibility from job:"
nvidia-smi || true
echo "GPU visibility from Ollama container:"
"${APPTAINER_OLLAMA[@]}" nvidia-smi || true

echo "Pulling model: $MODEL"

"${APPTAINER_OLLAMA[@]}" ollama pull "$MODEL"

echo "Testing Ollama API..."
echo "  OLLAMA_TEST_TIMEOUT=$OLLAMA_TEST_TIMEOUT"

python - "$OLLAMA_BASE_URL/api/generate" "$MODEL" "$OLLAMA_TEST_TIMEOUT" <<'PY'
import json
import sys
from urllib.request import Request, urlopen

url, model, timeout = sys.argv[1], sys.argv[2], int(sys.argv[3])
payload = json.dumps(
    {
        "model": model,
        "prompt": "Reply with exactly: OLLAMA_OK",
        "stream": False,
        "options": {"num_predict": 8},
    }
).encode("utf-8")
request = Request(url, data=payload, headers={"Content-Type": "application/json"})
with urlopen(request, timeout=timeout) as response:
    print(json.dumps(json.load(response), indent=2))
PY

echo "Running Python simulation test..."

export OLLAMA_MODEL="$MODEL"

python "$SCRIPT_DIR/generate_russia_personas.py" \
    --model "$MODEL" \
    --ollama-url "$OLLAMA_BASE_URL" \
    --experiment-name "$EXPERIMENT_NAME" \
    --workers "$PERSONA_WORKERS" \
    --timeout "$PERSONA_TIMEOUT" 

python "$SCRIPT_DIR/v0.py" \
    --model "$MODEL" \
    --ollama-url "$OLLAMA_OPENAI_URL" \
    --experiment-name "$EXPERIMENT_NAME" \
    --llm-steps 10 
     

echo "GPU OASIS run finished successfully."
