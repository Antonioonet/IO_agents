from pathlib import Path

import pandas as pd


threshold = -1
CONTROL_MIN_POSTS = 5
DATA_DIR = Path(__file__).resolve().parent / "data" / "Russia"


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


def read_merged_pickles(data_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {data_dir}")

    print(f"Reading {len(files)} file(s) matching {pattern!r}:")
    for file in files:
        print(f"  {file.name}")

    return pd.concat((pd.read_pickle(file) for file in files), ignore_index=True)


def run_procedure(label: str, df: pd.DataFrame) -> None:
    print(f"\n{label}")
    print(df.info())
    df = add_tweet_type(df)
    user_column = get_user_column(df)
    counts = df.groupby([user_column, "tweet_type"]).size().unstack(fill_value=0)

    users = (
        (counts.get("tweet", 0) > threshold)
        & (counts.get("reply", 0) > threshold)
        & (counts.get("retweet", 0) > threshold)
    )
    print(
        f"users with more than {threshold} tweets AND replies AND retweets:",
        users.sum(),
    )


def calculate_action_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = add_tweet_type(df)
    user_column = get_user_column(df)
    counts = df.groupby([user_column, "tweet_type"]).size().unstack(fill_value=0)
    return counts.reindex(columns=["tweet", "reply", "retweet"], fill_value=0)


def summarize_control_action_proportions(df: pd.DataFrame) -> None:
    counts = calculate_action_counts(df)
    totals = counts.sum(axis=1)
    eligible_counts = counts[totals > CONTROL_MIN_POSTS]
    totals = eligible_counts.sum(axis=1)
    proportions = eligible_counts.div(totals, axis=0)

    print(
        f"\nControl users with more than {CONTROL_MIN_POSTS} posts "
        "(tweet + reply + retweet): "
        f"{len(eligible_counts)}"
    )
    print("Average action proportions:")
    print(proportions.mean().to_string())
    print("Standard deviation of action proportions:")
    print(proportions.std().to_string())


def main() -> None:
    io_df = read_merged_pickles(DATA_DIR, "*io.pkl")
    control_df = read_merged_pickles(DATA_DIR, "*control.pkl")

    run_procedure("merged IO dataframe", io_df)
    run_procedure("merged control dataframe", control_df)
    summarize_control_action_proportions(control_df)


if __name__ == "__main__":
    main()
