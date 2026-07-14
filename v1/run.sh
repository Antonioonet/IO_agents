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

export NORMAL_FILE="${NORMAL_FILE:-v1/data/real_twitter_data/normal.pkl}"
export IO_FILE="${IO_FILE:-v1/data/real_twitter_data/io.pkl}"
export NORMAL_LIMIT="${NORMAL_LIMIT:-1}"
export IO_LIMIT="${IO_LIMIT:-1}"
export TWEETS_PER_USER="${TWEETS_PER_USER:-10}"
export LLM_STEPS="${LLM_STEPS:-1}"
export DO_NOTHING_PROB="${DO_NOTHING_PROB:-0.25}"

mkdir -p "$OLLAMA_MODELS_DIR"

apptainer pull "$OLLAMA_SIF" docker://ollama/ollama:latest

export APPTAINERENV_OLLAMA_LLM_LIBRARY=cuda_v12
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
  --action-mode "prompt_probabilities" \
  --model "$MODEL" \
  --ollama-url "$OLLAMA_BASE_URL" \
  --llm-steps "$LLM_STEPS"

