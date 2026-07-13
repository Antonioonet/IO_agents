#!/bin/bash

#SBATCH --account=emiliofe_74
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=persona-generation
#SBATCH --output=ollama-gpu-oasis-%j.out
#SBATCH --error=ollama-gpu-oasis-%j.err

set -e

# Load the HPC software environment.
module load conda
module load apptainer
source ~/.bashrc
cinit
conda activate oasis

# The script works from any directory.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Change these values here, or override them before running the script.
MODEL="${MODEL:-qwen3.6:35b-a3b-mtp-q4_K_M}"
NORMAL_FILE="${NORMAL_FILE:-$SCRIPT_DIR/data/russia/russia_201901_1_tweets_control.pkl}"
IO_FILE="${IO_FILE:-$SCRIPT_DIR/data/russia/russia_201901_1_tweets_io.pkl}"
NORMAL_LIMIT="${NORMAL_LIMIT:-50}"
IO_LIMIT="${IO_LIMIT:-10}"
MIN_TWEETS="${MIN_TWEETS:-10}"
TWEETS_PER_USER="${TWEETS_PER_USER:-20}"
ACTION_SEED="${ACTION_SEED:-0}"
PRIOR_SAMPLES="${PRIOR_SAMPLES:-10}"
PRIOR_FEED_SIZE="${PRIOR_FEED_SIZE:-10}"
PRIOR_SEED="${PRIOR_SEED:-0}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
OUTPUT_PATH="${OUTPUT_PATH:-$SCRIPT_DIR/data/generated_personas.csv}"

# Ollama files are kept in scratch so they do not fill the home directory.
OLLAMA_DIR="${OLLAMA_DIR:-/scratch1/$USER/ollama}"
OLLAMA_MODELS_DIR="$OLLAMA_DIR/models"
OLLAMA_IMAGE="$OLLAMA_DIR/ollama_latest.sif"
OLLAMA_URL="http://127.0.0.1:11434"

mkdir -p "$OLLAMA_MODELS_DIR"

# Download the Ollama container the first time only.
if [ ! -f "$OLLAMA_IMAGE" ]; then
    apptainer pull "$OLLAMA_IMAGE" docker://ollama/ollama:latest
fi

# Make the host model directory available inside the container.
export APPTAINERENV_OLLAMA_HOST="0.0.0.0:11434"
export APPTAINERENV_OLLAMA_MODELS="/models"
export APPTAINERENV_OLLAMA_LOAD_TIMEOUT="30m"

apptainer exec --nv \
    --bind "$OLLAMA_MODELS_DIR:/models" \
    "$OLLAMA_IMAGE" \
    ollama serve &
OLLAMA_PID=$!

# Always stop the Ollama server when this script exits.
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT

echo "Waiting for Ollama to start..."
until python -c \
    "from urllib.request import urlopen; urlopen('$OLLAMA_URL/api/tags', timeout=2)" \
    >/dev/null 2>&1; do
    sleep 2
done

echo "Pulling model: $MODEL"
export APPTAINERENV_OLLAMA_HOST="127.0.0.1:11434"
apptainer exec --nv \
    --bind "$OLLAMA_MODELS_DIR:/models" \
    "$OLLAMA_IMAGE" \
    ollama pull "$MODEL"

echo "Generating personas..."
python "$SCRIPT_DIR/persona_generation.py" \
    --normal-file "$NORMAL_FILE" \
    --io-file "$IO_FILE" \
    --normal-limit "$NORMAL_LIMIT" \
    --io-limit "$IO_LIMIT" \
    --min-tweets "$MIN_TWEETS" \
    --tweets-per-user "$TWEETS_PER_USER" \
    --action-seed "$ACTION_SEED" \
    --prior-samples "$PRIOR_SAMPLES" \
    --prior-feed-size "$PRIOR_FEED_SIZE" \
    --prior-seed "$PRIOR_SEED" \
    --model "$MODEL" \
    --ollama-url "$OLLAMA_URL" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --output-path "$OUTPUT_PATH"

echo "Personas written to: $OUTPUT_PATH"
