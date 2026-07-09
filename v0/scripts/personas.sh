cd .. 

python generate_russia_personas.py \
  --model "$MODEL" \
  --ollama-url "$OLLAMA_BASE_URL" \
  --io-input data/Russia/GRU_202012_tweets_io.pkl \
  --normal-input data/Russia/russia_201901_1_tweets_control.pkl \
  --experiments-dir experiments/tests \
  --experiment-name test_1 \
  --io-agent-count 4 \
  --normal-agent-count 4 \
  --io-hard-action-filter-threshold 10