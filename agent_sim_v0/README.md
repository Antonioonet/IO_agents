# Agent Sim V0

This is a minimal Ollama-backed agent simulation. It creates a few agents, lets them respond to a shared topic for several rounds, and saves the transcript to `outputs/`.

## Local Python Run

Install the small Python dependency set:

```bash
python -m pip install -r agent_sim_v0/requirements.txt
```

Start Ollama:

```bash
ollama pull llama3.2:3b
ollama serve
```

Run the simulation:

```bash
python -m agent_sim_v0.src.main --config agent_sim_v0/configs/local_ollama.yaml
```

Run with personas generated directly from tweet samples:

```bash
python -m agent_sim_v0.src.main --config agent_sim_v0/configs/tweets_ollama.yaml
```

Validate the config without calling the model:

```bash
python -m agent_sim_v0.src.main --config agent_sim_v0/configs/local_ollama.yaml --dry-run
```

The tweet workflow reads `agent_sim_v0/data/dummi_tweets_brazil.csv`, selects users with enough tweets, randomly samples tweets from each selected user, sends those samples to the LLM with `tweet_source.prompt_template`, saves generated personas to `outputs/personas_*.jsonl`, then uses those personas as the agents' roles during the discussion.

## Preview Tweet Selection

Preview which users will be selected without calling the LLM:

```bash
python -m agent_sim_v0.src.main --config agent_sim_v0/configs/tweets_ollama.yaml --dry-run
```

Run the merged workflow:

```bash
python -m agent_sim_v0.src.main --config agent_sim_v0/configs/tweets_ollama.yaml
```

## Docker Run

From the repository root:

```bash
docker build -t agent-sim-v0 -f docker/ollama-agent/Dockerfile .
docker run --rm -it -v "$PWD/agent_sim_v0/outputs:/app/agent_sim_v0/outputs" agent-sim-v0
```

Or run the full local Docker pipeline:

```bash
scripts/run_agent_sim_v0_local.sh
```

On a MacBook, use native Ollama to let Ollama use Apple Silicon GPU/Metal:

```bash
scripts/run_agent_sim_v0_mac_gpu.sh
```

Set a different Ollama model:

```bash
docker run --rm -it -e OLLAMA_MODEL=qwen2.5:1.5b agent-sim-v0
```
