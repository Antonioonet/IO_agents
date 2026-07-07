import pandas as pd


threshold = 10


for file in [
    "data/Russia/russia_201901_1_tweets_io.pkl",
    "data/Russia/russia_201901_1_tweets_control.pkl",
]:
    df = pd.read_pickle(file)
    print(df.info())
    counts = df.groupby(["user_id", "tweet_type"]).size().unstack(fill_value=0)

    print("\n", file)
    users = (
        (counts.get("tweet", 0) > threshold)
        & (counts.get("reply", 0) > threshold)
    )
    print(
        f"users with more than {threshold} tweets AND replies:",
        users.sum(),
    )
