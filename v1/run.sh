#!/bin/bash

module load conda
module load apptainer

source ~/.bashrc 
cinit
conda activate oasis

cd /project2/emiliofe_74/antonio/IO_agents

export MODEL="qwen3.6:35b-a3b-mtp-q4_K_M"
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"

export OLLAMA_BASE="/scratch1/$USER/ollama"
export OLLAMA_MODELS_DIR="$OLLAMA_BASE/models"
export OLLAMA_SIF="$OLLAMA_BASE/ollama_latest.sif"

mkdir -p "$OLLAMA_MODELS_DIR"

apptainer pull "$OLLAMA_SIF" docker://ollama/ollama:latest

export APPTAINERENV_OLLAMA_HOST="$OLLAMA_HOST"
export APPTAINERENV_OLLAMA_MODELS="/models"

apptainer exec --nv \
  --bind "$OLLAMA_MODELS_DIR:/models" \
  "$OLLAMA_SIF" \
  ollama serve &

sleep 10

apptainer exec --nv \
  --bind "$OLLAMA_MODELS_DIR:/models" \
  "$OLLAMA_SIF" \
  ollama pull "$MODEL"

python v1/simulation.py \
  --model "$MODEL" \
  --ollama-url "$OLLAMA_BASE_URL" \
  --llm-steps 1