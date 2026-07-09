cd ..

python generate_action_probabilities.py \
  --mode action \
  --input data/Russia/GRU_202012_tweets_io.pkl \
  --output user_prob/io_action_probabilities.csv \
  --threshold 10 \
  --samples 4 \
  --seed 0