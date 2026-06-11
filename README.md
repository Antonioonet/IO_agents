# IO Agents

This repository contains agent-based information operations experiments and helper notebooks.

## Python Environments

For an HPC node with NVIDIA GPUs and Conda/Mamba:

```bash
cd /path/to/IO_agents
mamba env create -f environment.hpc.yml
conda activate io-agents
python -m ipykernel install --user --name io-agents --display-name "Python (io-agents)"
```

For a CPU-only machine, including a local laptop:

```bash
cd /path/to/IO_agents
mamba env create -f environment.cpu.yml
conda activate io-agents-cpu
python -m ipykernel install --user --name io-agents-cpu --display-name "Python (io-agents-cpu)"
```

If `mamba` is not available, replace `mamba` with `conda`.

## Small LLM Options

The current agent configs can call Ollama through an OpenAI-compatible endpoint at:

```text
http://127.0.0.1:11434/v1
```

On a machine where Ollama is available:

```bash
ollama pull llama3.2:3b
ollama serve
```

Then update the model name in the relevant `config.py` file if needed.

For direct Hugging Face/PyTorch inference inside Python, the environments include `torch`, `transformers`, `accelerate`, and related packages. On the HPC, request a GPU node through the scheduler before running model-heavy scripts.

## Agent Sim V0

`agent_sim_v0` is a first small local-LLM simulation that talks to Ollama through its OpenAI-compatible API.

Build and run the Docker image locally:

```bash
docker build -t agent-sim-v0 -f docker/ollama-agent/Dockerfile .
docker run --rm -it -v "$PWD/agent_sim_v0/outputs:/app/agent_sim_v0/outputs" agent-sim-v0
```

With NVIDIA GPUs:

```bash
docker run --rm -it --gpus all -v "$PWD/agent_sim_v0/outputs:/app/agent_sim_v0/outputs" agent-sim-v0
```

On an HPC, adapt `jobs/run_agent_sim_v0.sbatch` to your partition, account, and container runtime.

The active simulation config is `agent_sim_v0/configs/tweets_ollama.yaml`. It selects users directly from `dummi_tweets_brazil.csv`, samples tweets, asks the LLM once per user to create the final agent persona, then starts the discussion simulation.

Run the full Docker pipeline locally:

```bash
scripts/run_agent_sim_v0_local.sh
```

Optional overrides:

```bash
OLLAMA_MODEL=qwen2.5:1.5b USER_COUNT=10 MAX_TWEETS_PER_USER=10 scripts/run_agent_sim_v0_local.sh
```

On a MacBook, run without Docker GPU flags:

```bash
USE_GPU=false scripts/run_agent_sim_v0_local.sh
```

To use the MacBook Apple Silicon GPU, run Ollama natively on macOS instead of inside Docker:

```bash
scripts/run_agent_sim_v0_mac_gpu.sh
```

Docker Desktop on macOS does not expose Apple Metal GPUs to Linux containers the same way NVIDIA GPUs are exposed on Linux. Native Ollama is the recommended Mac GPU path.
