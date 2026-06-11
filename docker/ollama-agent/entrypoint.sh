#!/usr/bin/env bash
set -euo pipefail

ollama serve &
OLLAMA_PID="$!"

cleanup() {
  kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Waiting for Ollama at ${OLLAMA_HOST}..."
until curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null; do
  sleep 1
done

echo "Ensuring model is available: ${OLLAMA_MODEL}"
ollama pull "${OLLAMA_MODEL}"

if [[ "$#" -eq 0 ]]; then
  set -- python -m agent_sim_v0.src.main --config "${SIM_CONFIG}"
elif [[ "$*" == "python -m agent_sim_v0.src.main" ]]; then
  set -- "$@" --config "${SIM_CONFIG}"
fi

exec "$@"
