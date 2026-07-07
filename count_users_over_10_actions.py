import pandas as pd


threshold = 10


def add_tweet_type(df: pd.DataFrame) -> pd.DataFrame:
    if "tweet_type" in df.columns:
        return df

    tweet_type = pd.Series("tweet", index=df.index)

    if "quoted_tweet_tweetid" in df.columns:
        tweet_type = tweet_type.mask(df["quoted_tweet_tweetid"].notna(), "quote")
    if "in_reply_to_tweetid" in df.columns:
        tweet_type = tweet_type.mask(df["in_reply_to_tweetid"].notna(), "reply")
    elif "in_reply_to_userid" in df.columns:
        tweet_type = tweet_type.mask(df["in_reply_to_userid"].notna(), "reply")
    if "is_retweet" in df.columns:
        tweet_type = tweet_type.mask(df["is_retweet"].fillna(False), "retweet")
    elif "retweet_tweetid" in df.columns:
        tweet_type = tweet_type.mask(df["retweet_tweetid"].notna(), "retweet")

    df = df.copy()
    df["tweet_type"] = tweet_type
    return df


def get_user_column(df: pd.DataFrame) -> str:
    for column in ("user_id", "userid"):
        if column in df.columns:
            return column
    raise ValueError("DataFrame is missing a user id column: expected user_id or userid")


for file in [
    "data/Russia/russia_201901_1_tweets_io.pkl",
    "data/Russia/russia_201901_1_tweets_control.pkl",
]:
    df = pd.read_pickle(file)
    print(df.info())
    df = add_tweet_type(df)
    user_column = get_user_column(df)
    counts = df.groupby([user_column, "tweet_type"]).size().unstack(fill_value=0)

    print("\n", file)
    users = (
        (counts.get("tweet", 0) > threshold)
        & (counts.get("reply", 0) > threshold)
    )
    print(
        f"users with more than {threshold} tweets AND replies:",
        users.sum(),
    )
