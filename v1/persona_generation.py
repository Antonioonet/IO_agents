import json
import random
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "real_twitter_data"
OUTPUT_PATH = BASE_DIR / "data" / "generated_personas.csv"


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


def call_ollama(prompt, model="qwen3.6:35b-a3b-mtp-q4_K_M", ollama_url="http://127.0.0.1:11434"):
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
    with urlopen(request, timeout=180) as response:
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
    is_retweet = df["is_retweet"].fillna(False) | df["retweet_tweetid"].notna()
    is_comment = df["in_reply_to_tweetid"].notna()
    is_original_tweet = ~is_retweet & ~is_comment

    counts = pd.DataFrame(
        {
            "tweets": is_original_tweet.groupby(df["userid"]).sum(),
            "retweets": is_retweet.groupby(df["userid"]).sum(),
            "comments": is_comment.groupby(df["userid"]).sum(),
        }
    )
    selected_userids = counts[
        (counts["tweets"] >= min_original_tweets)
        & (counts["retweets"] >= min_retweets)
        & (counts["comments"] >= min_comments)
    ].index
    return df[df["userid"].isin(selected_userids)]


def generate_personas(
    data_dir=DATA_DIR,
    io_file=None,
    normal_file=None,
    normal_limit=None,
    io_limit=None,
    min_tweets=10,
    tweets_per_user=20,
    model="qwen3.6:35b-a3b-mtp-q4_K_M",
    ollama_url="http://127.0.0.1:11434",
    output_path=OUTPUT_PATH,
):
    data_dir = Path(data_dir)
    normal_file = Path(normal_file) if normal_file else data_dir / "normal.pkl"
    io_file = Path(io_file) if io_file else data_dir / "io.pkl"

    df_normal = pd.read_pickle(normal_file)
    df_io = pd.read_pickle(io_file)

    personas = []
    ct = 0

    for df, is_io in [(df_normal, False), (df_io, True)]:
        limit = io_limit if is_io else normal_limit

        if is_io:
            df = filter_io_users(df)
        else:
            tweet_counts = df.groupby("userid").size()
            selected_userids = tweet_counts[tweet_counts > min_tweets].index
            df = df[df["userid"].isin(selected_userids)]

        group_count = 0
        for _, user_tweets in df.groupby("userid"):
            if limit is not None and group_count >= limit:
                break

            clean_tweets = user_tweets["tweet_text"].dropna().astype(str).tolist()
            sampled_tweets = random.sample(clean_tweets, min(tweets_per_user, len(clean_tweets)))
            user_row = user_tweets.iloc[0]

            username_prompt = build_username_prompt(user_row, sampled_tweets)
            username = clean_username(call_ollama(username_prompt, model=model, ollama_url=ollama_url))

            persona_prompt = build_persona_prompt(user_row, sampled_tweets, username)
            description = call_ollama(persona_prompt, model=model, ollama_url=ollama_url)

            personas.append(
                {
                    "user_id": ct,
                    "real_user_id": user_row.get("userid", ""),
                    "name": username,
                    "username": username,
                    "user_char": description,
                    "description": description,
                    "I.O": is_io,
                }
            )
            print(f"Generated persona {len(personas)}: {username}")
            ct += 1
            group_count += 1

    personas_df = pd.DataFrame(personas)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    personas_df.to_csv(output_path, index=False)
    return personas_df


if __name__ == "__main__":
    generate_personas()
