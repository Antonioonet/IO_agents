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
            "normal mode samples per-user probabilities from a dataset-level "
            "Dirichlet distribution."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("action", "normal"),
        default="action",
        help="Generation mode. action uses empirical user proportions; normal samples from a dataset-level Dirichlet distribution.",
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
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of probability rows to generate. Defaults to one row per eligible/user row.",
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


def validate_samples_count(samples_count: int | None) -> None:
    if samples_count is not None and samples_count <= 0:
        raise ValueError("--samples must be greater than 0.")


def generate_empirical_probabilities(
    df: pd.DataFrame,
    threshold: int,
    seed: int,
    samples_count: int | None,
) -> pd.DataFrame:
    validate_samples_count(samples_count)
    counts = calculate_action_counts(df)
    eligible = (
        (counts["post"] > threshold)
        & (counts["reply"] > threshold)
        & (counts["retweet"] > threshold)
    )
    eligible_counts = counts[eligible]
    output = counts_to_probability_rows(eligible_counts)

    if samples_count is None:
        return output
    if samples_count > len(output):
        raise ValueError(
            f"--samples={samples_count} is larger than the {len(output)} eligible users."
        )
    return output.sample(n=samples_count, random_state=seed).sort_values("user_id")


def generate_normal_probabilities(
    df: pd.DataFrame,
    seed: int,
    samples_count: int | None,
) -> pd.DataFrame:
    validate_samples_count(samples_count)
    counts = calculate_action_counts(df)
    counts = counts[counts.sum(axis=1) > 0]
    samples_count = samples_count or len(counts)
    user_ids = list(range(1, samples_count + 1))
    proportions = counts.div(counts.sum(axis=1), axis=0)

    alpha = estimate_dirichlet_alpha(proportions)
    rng = np.random.default_rng(seed)
    samples = rng.dirichlet(alpha, size=samples_count)

    output = pd.DataFrame(samples, columns=ACTION_COLUMNS)
    output.insert(0, "user_id", user_ids)
    output.insert(1, "p_action", 0.0)
    return output[["user_id", "p_action", *ACTION_COLUMNS]]


def estimate_dirichlet_alpha(proportions: pd.DataFrame) -> np.ndarray:
    means = proportions.mean().to_numpy(dtype=float)
    means = np.clip(means, 1e-6, None)
    means = means / means.sum()

    variances = proportions.var(ddof=1).fillna(0.0).to_numpy(dtype=float)
    concentration_estimates = []
    for mean, variance in zip(means, variances):
        if variance <= 0:
            continue
        estimate = mean * (1.0 - mean) / variance - 1.0
        if np.isfinite(estimate) and estimate > 0:
            concentration_estimates.append(estimate)

    if concentration_estimates:
        concentration = float(np.median(concentration_estimates))
    else:
        concentration = 100.0

    concentration = float(np.clip(concentration, 20.0, 1000.0))
    return np.clip(means * concentration, 0.05, None)


def main() -> None:
    args = parse_args()
    df = read_inputs(args)

    if args.mode == "action":
        output = generate_empirical_probabilities(
            df,
            args.threshold,
            args.seed,
            args.samples,
        )
    else:
        output = generate_normal_probabilities(df, args.seed, args.samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} user probability rows to {args.output}")


if __name__ == "__main__":
    main()