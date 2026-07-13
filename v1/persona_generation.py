import argparse
import json
import random
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "real_twitter_data"
OUTPUT_PATH = BASE_DIR / "data" / "generated_personas.csv"
ACTION_COLUMNS = ["post", "reply", "retweet"]


PERSONA_PROMPT = """
You are creating a social-media persona profile for an OASIS Twitter/X agent simulation.

Read the sampled tweets and public profile metadata, then describe how this simulated user writes posts and replies. Infer stable behavior from the examples: recurring topics, opinions, tone, emotional style, interaction habits, likely replies to other users, and the kinds of posts they would naturally create. Do not invent private biographical facts, hidden affiliations, real names, exact locations, or offline activity. Preserve culturally relevant signals, but write the final persona in English.

Return only the persona description as plain text. Do not use JSON, markdown, bullet points, headings, code fences, or section labels. Write one long paragraph. Start exactly in the natural style "{username} is a ..." and make the paragraph specific enough to guide how this simulated user writes posts and replies.

Generated username: {username}
Reported location: {location}
Profile description: {profile_description}
Follower count: {follower_count}
Following count: {following_count}

Sampled tweets:
{tweets}
""".strip()


USERNAME_PROMPT = """
Create one fake Twitter/X username for a simulated social-media user based on the style, topics, and tone of these sampled tweets.

The username must be fictional. It must not copy, resemble, parody, or be a small variation of the original username or display name. Do not reuse words, numbers, initials, handles, or distinctive fragments from the original username or display name. It should sound natural for Twitter/X and fit the user's posting style, but it must not identify the real user.

Return only the username. Do not include @. Do not use spaces. Do not use JSON, markdown, bullet points, explanations, headings, or code fences.

Original display name to avoid: {display_name}
Original username to avoid: {screen_name}

Sampled tweets:
{tweets}
""".strip()


def call_ollama(
    prompt,
    model="qwen3.6:35b-a3b-mtp-q4_K_M",
    ollama_url="http://127.0.0.1:11434",
    request_timeout=1800,
):
    request = Request(
        ollama_url.rstrip("/") + "/api/generate",
        data=json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=request_timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["response"].strip()


def clean_username(username):
    return (
        username.strip()
        .splitlines()[0]
        .replace("@", "")
        .replace('"', "")
        .replace("'", "")
        .replace(" ", "_")
    )


def format_tweets(tweets):
    return "\n".join(f"- {tweet}" for tweet in tweets)


def has_value(series):
    text = series.astype("string").str.strip().str.lower()
    missing_text = text.isin(["", "nan", "none", "null", "na", "n/a", "<na>"])
    return series.notna() & ~missing_text


def is_true(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    text = series.astype("string").str.strip().str.lower()
    return text.isin(["true", "1", "t", "yes", "y"])


def exclude_quotes(df):
    quote_columns = [
        "is_quote",
        "is_quote_status",
        "quoted_status_id",
        "quoted_tweetid",
        "quoted_tweet_id",
        "quote_tweetid",
        "quote_tweet_id",
    ]
    quote_mask = pd.Series(False, index=df.index)
    for column in quote_columns:
        if column not in df.columns:
            continue
        if column.startswith("is_"):
            quote_mask = quote_mask | is_true(df[column])
        else:
            quote_mask = quote_mask | has_value(df[column])
    return df[~quote_mask].copy()


def add_tweet_type(df):
    if "tweet_type" in df.columns:
        df = df.copy()
        df["tweet_type"] = df["tweet_type"].replace({"tweet": "post"})
        return df[df["tweet_type"].isin(ACTION_COLUMNS)].copy()

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
    return df[df["tweet_type"].isin(ACTION_COLUMNS)].copy()


def prepare_action_rows(df):
    return add_tweet_type(exclude_quotes(df))


def calculate_action_counts(df):
    df = prepare_action_rows(df)
    counts = df.groupby(["userid", "tweet_type"]).size().unstack(fill_value=0)
    return counts.reindex(columns=ACTION_COLUMNS, fill_value=0)


def counts_to_probabilities(counts):
    total = float(sum(counts.get(action, 0) for action in ACTION_COLUMNS))
    if total <= 0:
        return {"p_action": 0.0, "post": 1 / 3, "reply": 1 / 3, "retweet": 1 / 3}
    return {
        "p_action": 0.0,
        **{
            action: float(counts.get(action, 0) / total)
            for action in ACTION_COLUMNS
        },
    }


def estimate_dirichlet_alpha(action_counts):
    action_counts = action_counts[action_counts.sum(axis=1) > 0]
    if action_counts.empty:
        return np.ones(len(ACTION_COLUMNS), dtype=float)

    proportions = action_counts.div(action_counts.sum(axis=1), axis=0)
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

    concentration = (
        float(np.median(concentration_estimates))
        if concentration_estimates
        else 100.0
    )
    concentration = float(np.clip(concentration, 20.0, 1000.0))
    return np.clip(means * concentration, 0.05, None)


def sample_dirichlet_probabilities(rng, alpha):
    sample = rng.dirichlet(alpha)
    return {
        "p_action": 0.0,
        **{
            action: float(sample[index])
            for index, action in enumerate(ACTION_COLUMNS)
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate OASIS personas and action probabilities from normal "
            "and IO Twitter pickle datasets."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=(
            "Directory containing normal.pkl and io.pkl. Explicit "
            "--normal-file/--io-file values override these defaults."
        ),
    )
    parser.add_argument(
        "--normal-file",
        type=Path,
        default=None,
        help="Path to the normal-user pickle dataset.",
    )
    parser.add_argument(
        "--io-file",
        type=Path,
        default=None,
        help="Path to the IO-user pickle dataset.",
    )
    parser.add_argument(
        "--normal-limit",
        type=int,
        default=None,
        help="Maximum number of normal-user personas to generate.",
    )
    parser.add_argument(
        "--io-limit",
        type=int,
        default=None,
        help="Maximum number of IO-user personas to generate.",
    )
    parser.add_argument(
        "--min-tweets",
        type=int,
        default=10,
        help="Minimum number of usable actions required for a normal user.",
    )
    parser.add_argument(
        "--tweets-per-user",
        type=int,
        default=20,
        help="Maximum number of tweets sampled for each LLM prompt.",
    )
    parser.add_argument(
        "--action-seed",
        type=int,
        default=0,
        help="Random seed used to sample normal-user action probabilities.",
    )
    parser.add_argument(
        "--model",
        default="qwen3.6:35b-a3b-mtp-q4_K_M",
        help="Ollama model used to generate usernames and personas.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Base URL of the Ollama server.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=1800,
        help="Maximum seconds to wait for each Ollama response.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="Destination CSV path.",
    )
    args = parser.parse_args(argv)

    for option in ("normal_limit", "io_limit"):
        value = getattr(args, option)
        if value is not None and value < 0:
            parser.error(f"--{option.replace('_', '-')} must be 0 or greater")
    if args.min_tweets < 0:
        parser.error("--min-tweets must be 0 or greater")
    if args.tweets_per_user <= 0:
        parser.error("--tweets-per-user must be greater than 0")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be greater than 0")

    return args


def build_username_prompt(user_row, tweets):
    return USERNAME_PROMPT.format(
        display_name=user_row.get("user_display_name", ""),
        screen_name=user_row.get("user_screen_name", ""),
        tweets=format_tweets(tweets),
    )


def build_persona_prompt(user_row, tweets, username):
    return PERSONA_PROMPT.format(
        username=username,
        location=user_row.get("user_reported_location", ""),
        profile_description=user_row.get("user_profile_description", ""),
        follower_count=user_row.get("follower_count", ""),
        following_count=user_row.get("following_count", ""),
        tweets=format_tweets(tweets),
    )


def filter_io_users(df, min_original_tweets=5, min_retweets=5, min_comments=5):
    df = prepare_action_rows(df)
    counts = calculate_action_counts(df)
    selected_userids = counts[
        (counts["post"] >= min_original_tweets)
        & (counts["retweet"] >= min_retweets)
        & (counts["reply"] >= min_comments)
    ].index
    return df[df["userid"].isin(selected_userids)]


def filter_normal_users(df, min_actions=10):
    df = prepare_action_rows(df)
    counts = calculate_action_counts(df)
    selected_userids = counts[counts.sum(axis=1) >= min_actions].index
    return df[df["userid"].isin(selected_userids)]


def generate_personas(
    data_dir=DATA_DIR,
    io_file=None,
    normal_file=None,
    normal_limit=None,
    io_limit=None,
    min_tweets=10,
    tweets_per_user=20,
    action_seed=0,
    model="qwen3.6:35b-a3b-mtp-q4_K_M",
    ollama_url="http://127.0.0.1:11434",
    request_timeout=1800,
    output_path=OUTPUT_PATH,
):
    data_dir = Path(data_dir)
    normal_file = Path(normal_file) if normal_file else data_dir / "normal.pkl"
    io_file = Path(io_file) if io_file else data_dir / "io.pkl"

    df_normal = pd.read_pickle(normal_file)
    df_io = pd.read_pickle(io_file)
    df_normal = filter_normal_users(df_normal, min_actions=min_tweets)
    normal_action_counts = calculate_action_counts(df_normal)
    normal_action_alpha = estimate_dirichlet_alpha(normal_action_counts)
    action_rng = np.random.default_rng(action_seed)

    personas = []
    ct = 0

    for df, is_io in [(df_normal, False), (df_io, True)]:
        limit = io_limit if is_io else normal_limit

        if is_io:
            df = filter_io_users(df)
        else:
            df = filter_normal_users(df, min_actions=min_tweets)

        group_count = 0
        for _, user_tweets in df.groupby("userid"):
            if limit is not None and group_count >= limit:
                break

            clean_tweets = user_tweets["tweet_text"].dropna().astype(str).tolist()
            sampled_tweets = random.sample(clean_tweets, min(tweets_per_user, len(clean_tweets)))
            user_row = user_tweets.iloc[0]

            username_prompt = build_username_prompt(user_row, sampled_tweets)
            username = clean_username(
                call_ollama(
                    username_prompt,
                    model=model,
                    ollama_url=ollama_url,
                    request_timeout=request_timeout,
                )
            )

            persona_prompt = build_persona_prompt(user_row, sampled_tweets, username)
            description = call_ollama(
                persona_prompt,
                model=model,
                ollama_url=ollama_url,
                request_timeout=request_timeout,
            )
            if is_io:
                action_probabilities = counts_to_probabilities(
                    user_tweets["tweet_type"].value_counts()
                )
            else:
                action_probabilities = sample_dirichlet_probabilities(
                    action_rng,
                    normal_action_alpha,
                )

            personas.append(
                {
                    "user_id": ct,
                    "real_user_id": user_row.get("userid", ""),
                    "name": username,
                    "username": username,
                    "user_char": description,
                    "description": description,
                    "I.O": is_io,
                    **action_probabilities,
                }
            )
            print(f"Generated persona {len(personas)}: {username}")
            ct += 1
            group_count += 1

    personas_df = pd.DataFrame(personas)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    personas_df.to_csv(output_path, index=False)
    return personas_df


def main(argv=None):
    args = parse_args(argv)
    return generate_personas(
        data_dir=args.data_dir,
        normal_file=args.normal_file,
        io_file=args.io_file,
        normal_limit=args.normal_limit,
        io_limit=args.io_limit,
        min_tweets=args.min_tweets,
        tweets_per_user=args.tweets_per_user,
        action_seed=args.action_seed,
        model=args.model,
        ollama_url=args.ollama_url,
        request_timeout=args.request_timeout,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
