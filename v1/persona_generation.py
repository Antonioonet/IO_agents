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


def generate_personas(
    data_dir=DATA_DIR,
    min_tweets=10,
    tweets_per_user=20,
    model="qwen3.6:35b-a3b-mtp-q4_K_M",
    ollama_url="http://127.0.0.1:11434",
    output_path=OUTPUT_PATH,
):
    data_dir = Path(data_dir)
    frames = [pd.read_pickle(path) for path in sorted(data_dir.glob("*.pkl"))]
    df = pd.concat(frames, ignore_index=True)

    tweet_counts = df.groupby("userid").size()
    selected_userids = tweet_counts[tweet_counts > min_tweets].index
    df = df[df["userid"].isin(selected_userids)]

    personas = []
    for _, user_tweets in df.groupby("userid"):
        clean_tweets = user_tweets["tweet_text"].dropna().astype(str).tolist()
        sampled_tweets = random.sample(clean_tweets, min(tweets_per_user, len(clean_tweets)))
        user_row = user_tweets.iloc[0]

        username_prompt = build_username_prompt(user_row, sampled_tweets)
        username = clean_username(call_ollama(username_prompt, model=model, ollama_url=ollama_url))

        persona_prompt = build_persona_prompt(user_row, sampled_tweets, username)
        description = call_ollama(persona_prompt, model=model, ollama_url=ollama_url)

        personas.append(
            {
                "name": username,
                "username": username,
                "user_char": description,
                "description": description,
            }
        )
        print(f"Generated persona {len(personas)}/{len(selected_userids)}: {username}")

    personas_df = pd.DataFrame(personas)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    personas_df.to_csv(output_path, index=False)
    return personas_df


if __name__ == "__main__":
    generate_personas()
