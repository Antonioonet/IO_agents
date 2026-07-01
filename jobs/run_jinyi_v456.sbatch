#!/bin/bash -l
#SBATCH --account=emiliofe_74
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2%1
#SBATCH --job-name=jinyi_v456
#SBATCH --output=logs/jinyi_v456_%A_%a.out
#SBATCH --error=logs/jinyi_v456_%A_%a.err

set -euo pipefail

cd /home1/ad_892/projects/IO_agents

VERSIONS=(v4_hashtag v5_hashtag v6_hashtag)
VERSION="${VERSIONS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="${VERSION}"
OUT_DIR="outputs/${VERSION}"
VENV_DIR="${HOME}/.venvs/io_agents_jinyi_${VERSION}"
OLLAMA_INSTALL_DIR="${HOME}/.local"
OLLAMA_MODELS_DIR="/scratch1/ad_892/ollama_models"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.3:latest}"
OLLAMA_LOG="${PWD}/logs/ollama_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

mkdir -p "Jinyi_paper/${OUT_DIR}" logs "${OLLAMA_MODELS_DIR}"
rm -f "Jinyi_paper/${OUT_DIR}/memory.csv" "Jinyi_paper/${OUT_DIR}/follow_log.csv" "Jinyi_paper/${OUT_DIR}/discussion.csv"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array job: ${SLURM_ARRAY_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Running: ${VERSION}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Ollama model: ${OLLAMA_MODEL}"

module purge
module load gcc/13.3.0
module load python/3.11.9
module load zstd/1.5.6

export PATH="${OLLAMA_INSTALL_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${OLLAMA_INSTALL_DIR}/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_MODELS="${OLLAMA_MODELS_DIR}"
export OLLAMA_KEEP_ALIVE="24h"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Installing Ollama under ${OLLAMA_INSTALL_DIR}"
  curl -L https://ollama.com/download/ollama-linux-amd64.tar.zst -o "${TMPDIR:-/tmp}/ollama-linux-amd64.tar.zst"
  mkdir -p "${OLLAMA_INSTALL_DIR}"
  tar -C "${OLLAMA_INSTALL_DIR}" -xf "${TMPDIR:-/tmp}/ollama-linux-amd64.tar.zst"
fi

ollama --version || true

echo "Starting Ollama server"
ollama serve > "${OLLAMA_LOG}" 2>&1 &
OLLAMA_PID=$!
cleanup() {
  if kill -0 "${OLLAMA_PID}" >/dev/null 2>&1; then
    kill "${OLLAMA_PID}" || true
  fi
}
trap cleanup EXIT

for attempt in $(seq 1 60); do
  if curl -sS --max-time 2 "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    echo "Ollama is ready"
    break
  fi
  if ! kill -0 "${OLLAMA_PID}" >/dev/null 2>&1; then
    echo "Ollama server exited early. Log follows:" >&2
    sed -n "1,200p" "${OLLAMA_LOG}" >&2 || true
    exit 1
  fi
  sleep 2
  if [[ "${attempt}" == "60" ]]; then
    echo "Timed out waiting for Ollama. Log follows:" >&2
    sed -n "1,200p" "${OLLAMA_LOG}" >&2 || true
    exit 1
  fi
done

echo "Ensuring model is available: ${OLLAMA_MODEL}"
ollama pull "${OLLAMA_MODEL}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install ag2 openai groq diskcache numpy pandas pyyaml

cd Jinyi_paper
python "${RUN_DIR}/main_resume.py" \
  --memory_output_path "${OUT_DIR}/memory.csv" \
  --follow_output_path "${OUT_DIR}/follow_log.csv" \
  --discussion_output_path "${OUT_DIR}/discussion.csv"

echo "Finished: $(date)"
