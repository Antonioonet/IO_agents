import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ACTION_COLUMNS = ["post", "reply", "retweet"]
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "Russia"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-user action probability CSVs from tweet datasets. "
            "Default mode uses each eligible user's empirical action proportions; "
            "normal mode samples per-user probabilities from dataset-level normal "
            "distributions."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("action", "normal"),
        default="action",
        help="Generation mode. action uses empirical user proportions; normal samples from dataset-level normal distributions.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        help="Input pickle file(s). If omitted, files are read from --data-dir using --pattern.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory used when --input is omitted.",
    )
    parser.add_argument(
        "--pattern",
        default="*.pkl",
        help="Glob pattern used in --data-dir when --input is omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="In action mode, keep users with more than this many posts, replies, and retweets each.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for normal mode.",
    )
    return parser.parse_args()


def has_value(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    missing_text = text.isin(["", "nan", "none", "null", "na", "n/a", "<na>"])
    return series.notna() & ~missing_text


def is_true(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    text = series.astype("string").str.strip().str.lower()
    return text.isin(["true", "1", "t", "yes", "y"])


def add_tweet_type(df: pd.DataFrame) -> pd.DataFrame:
    if "tweet_type" in df.columns:
        df = df.copy()
        df["tweet_type"] = df["tweet_type"].replace({"tweet": "post"})
        return df

    tweet_type = pd.Series("post", index=df.index)

    if "in_reply_to_tweetid" in df.columns:
        tweet_type = tweet_type.mask(has_value(df["in_reply_to_tweetid"]), "reply")
    elif "in_reply_to_userid" in df.columns:
        tweet_type = tweet_type.mask(has_value(df["in_reply_to_userid"]), "reply")

    if "is_retweet" in df.columns:
        retweet_mask = is_true(df["is_retweet"])
        if "retweet_tweetid" in df.columns:
            retweet_mask = retweet_mask | has_value(df["retweet_tweetid"])
        tweet_type = tweet_type.mask(retweet_mask, "retweet")
    elif "retweet_tweetid" in df.columns:
        tweet_type = tweet_type.mask(has_value(df["retweet_tweetid"]), "retweet")

    df = df.copy()
    df["tweet_type"] = tweet_type
    return df


def get_user_column(df: pd.DataFrame) -> str:
    for column in ("user_id", "userid"):
        if column in df.columns:
            return column
    raise ValueError("DataFrame is missing a user id column: expected user_id or userid")


def read_inputs(args: argparse.Namespace) -> pd.DataFrame:
    files = args.input
    if files is None:
        files = sorted(args.data_dir.glob(args.pattern))
        if not files:
            raise FileNotFoundError(
                f"No files matching {args.pattern!r} found in {args.data_dir}"
            )

    print(f"Reading {len(files)} input file(s):")
    for file in files:
        print(f"  {file}")

    return pd.concat((pd.read_pickle(file) for file in files), ignore_index=True)


def calculate_action_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = add_tweet_type(df)
    user_column = get_user_column(df)
    counts = df.groupby([user_column, "tweet_type"]).size().unstack(fill_value=0)
    return counts.reindex(columns=ACTION_COLUMNS, fill_value=0)


def counts_to_probability_rows(counts: pd.DataFrame) -> pd.DataFrame:
    totals = counts.sum(axis=1)
    probabilities = counts.div(totals, axis=0)
    probabilities = probabilities.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    output = probabilities.reset_index().rename(columns={counts.index.name: "user_id"})
    output.insert(1, "p_action", 0.0)
    return output[["user_id", "p_action", *ACTION_COLUMNS]]


def generate_empirical_probabilities(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    counts = calculate_action_counts(df)
    eligible = (
        (counts["post"] > threshold)
        & (counts["reply"] > threshold)
        & (counts["retweet"] > threshold)
    )
    eligible_counts = counts[eligible]
    return counts_to_probability_rows(eligible_counts)


def generate_normal_probabilities(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = calculate_action_counts(df)
    counts = counts[counts.sum(axis=1) > 0]
    user_ids = counts.index.to_list()
    proportions = counts.div(counts.sum(axis=1), axis=0)

    means = proportions.mean()
    stds = proportions.std().fillna(0.0)
    rng = np.random.default_rng(seed)
    samples = rng.normal(
        loc=means.to_numpy(),
        scale=stds.to_numpy(),
        size=(len(user_ids), len(ACTION_COLUMNS)),
    )
    samples = np.clip(samples, 0.0, None)
    row_totals = samples.sum(axis=1)
    zero_rows = row_totals <= 0
    if zero_rows.any():
        samples[zero_rows] = means.to_numpy()
        row_totals = samples.sum(axis=1)
    samples = samples / row_totals[:, None]

    output = pd.DataFrame(samples, columns=ACTION_COLUMNS)
    output.insert(0, "user_id", user_ids)
    output.insert(1, "p_action", 0.0)
    return output[["user_id", "p_action", *ACTION_COLUMNS]]


def main() -> None:
    args = parse_args()
    df = read_inputs(args)

    if args.mode == "action":
        output = generate_empirical_probabilities(df, args.threshold)
    else:
        output = generate_normal_probabilities(df, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} user probability rows to {args.output}")


if __name__ == "__main__":
    main()
