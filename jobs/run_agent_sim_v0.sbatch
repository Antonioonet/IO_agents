#!/usr/bin/env bash
#SBATCH --job-name=agent-sim-v0
#SBATCH --output=logs/agent-sim-v0-%j.out
#SBATCH --error=logs/agent-sim-v0-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
IMAGE_NAME="agent-sim-v0"
DATA_DIR="${REPO_DIR}/agent_sim_v0/data"
OUTPUT_DIR="${REPO_DIR}/agent_sim_v0/outputs"
MODEL_NAME="${OLLAMA_MODEL:-llama3.2:3b}"

mkdir -p "${REPO_DIR}/logs" "${DATA_DIR}" "${OUTPUT_DIR}"

cd "${REPO_DIR}"

echo "Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" -f docker/ollama-agent/Dockerfile .

echo "Running full Agent Sim V0 pipeline with model: ${MODEL_NAME}"
docker run --rm \
  --gpus all \
  -e OLLAMA_MODEL="${MODEL_NAME}" \
  -v "${DATA_DIR}:/app/agent_sim_v0/data" \
  -v "${OUTPUT_DIR}:/app/agent_sim_v0/outputs" \
  "${IMAGE_NAME}" \
  bash -lc '
    set -euo pipefail

    echo "Running simulation with personas generated directly from tweet samples"
    python -m agent_sim_v0.src.main \
      --config agent_sim_v0/configs/tweets_ollama.yaml \
      --model "${OLLAMA_MODEL}" \
      --user-count 10 \
      --min-tweets 10 \
      --max-tweets-per-user 10 \
      --selection top
  '
