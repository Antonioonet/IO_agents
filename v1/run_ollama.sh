export OLLAMA_BASE="/scratch/$USER/ollama" 
export OLLAMA_MODELS_DIR="$OLLAMA_BASE/models"
export OLLAMA_SIF="$OLLAMA_BASE/ollama_latest.sif"
export APPTAINER_CACHEDIR="$/scratch1/$USER/apptainer-cache"
export MODEL="qwen-35b-a3b-mtp-q4_K_M"


mkdir -p "$OLLAMA_BASE" "$OLLAMA_MODELS_DIR" "$APPTAINER_CACHEDIR"


module load apptainer
if [ ! -f "$OLLAMA_SIF" ]; then
    apptainer pull "$OLLAMA_SIF" docker://ollama/ollama:latest
fi

export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_BASE_URL="http://$OLLAMA_HOST"

export APPTAINERENV_OLLAMA_HOST="$OLLAMA_HOST"
export APPTAINERENV_OLLAMA_MODELS="/models"

APPTAINER_OLLAMA=(
    apptainer exec
    --nv
    --bind "$OLLAMA_MODELS_DIR:/models"
    "$OLLAMA_SIF"
)

"${APPTAINER_OLLAMA[@]}" \
    ollama serve > "$OLLAMA_SERVER_LOG" 2>&1 & OLLAMA_PID=$!

sleep 10 

"${APPTAINER_OLLAMA[@]}" ollama pull "$MODEL"

