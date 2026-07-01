#!/bin/bash

#SBATCH --account=<your_project_id>
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=ollama-cpu-test
#SBATCH --output=ollama-cpu-test-%j.out
#SBATCH --error=ollama-cpu-test-%j.err

set -euo pipefail

module purge
module load apptainer

export OLLAMA_BASE="/scratch1/$USER/ollama-cpu-test"
export OLLAMA_MODELS_DIR="$OLLAMA_BASE/models"
export OLLAMA_SIF="$OLLAMA_BASE/ollama_latest.sif"
export APPTAINER_CACHEDIR="/scratch1/$USER/apptainer-cache"

mkdir -p "$OLLAMA_BASE" "$OLLAMA_MODELS_DIR" "$APPTAINER_CACHEDIR"

if [ ! -f "$OLLAMA_SIF" ]; then
    apptainer pull "$OLLAMA_SIF" docker://ollama/ollama:latest
fi

# Local only: visible only inside this compute node.
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"

# Pass env vars into Apptainer.
export APPTAINERENV_OLLAMA_HOST="$OLLAMA_HOST"
export APPTAINERENV_OLLAMA_MODELS="/models"

# Optional CPU threading hint.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cleanup() {
    echo "Stopping Ollama..."
    if [ -n "${OLLAMA_PID:-}" ]; then
        kill "$OLLAMA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Starting Ollama CPU server..."

apptainer exec \
    --bind "$OLLAMA_MODELS_DIR:/models" \
    "$OLLAMA_SIF" \
    ollama serve > "$OLLAMA_BASE/ollama-server-${SLURM_JOB_ID}.log" 2>&1 &

OLLAMA_PID=$!

echo "Waiting for Ollama server..."

for i in {1..60}; do
    if curl -s "$OLLAMA_BASE_URL/api/tags" > /dev/null; then
        echo "Ollama is ready."
        break
    fi

    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "Ollama server died."
        echo "Check log: $OLLAMA_BASE/ollama-server-${SLURM_JOB_ID}.log"
        exit 1
    fi

    sleep 2
done

MODEL="smollm2:135m"

echo "Pulling small CPU-friendly model: $MODEL"

echo "CPU info:"
lscpu | egrep 'Model name|Flags|Architecture'

echo "Ollama version:"
apptainer exec \
    --bind "$OLLAMA_MODELS_DIR:/models" \
    "$OLLAMA_SIF" \
    ollama --version
    
apptainer exec \
    --bind "$OLLAMA_MODELS_DIR:/models" \
    "$OLLAMA_SIF" \
    ollama pull "$MODEL"

echo "Testing Ollama API with streaming..."

curl -N -S "$OLLAMA_BASE_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\", \"prompt\":\"Reply with exactly: OLLAMA_OK\"}"

echo
echo "Checking server process..."

if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "Ollama died during generation."
    echo "Server log:"
    cat "$OLLAMA_BASE/ollama-server-${SLURM_JOB_ID}.log"
    exit 1
fi
echo "Activating Conda..."
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate oasis
echo "Conda activated."

echo "Running Python simulation test..."

python "generate_russia --model smollm2:135m v0.py  --model smollm2:135m"

echo "CPU test finished successfully."