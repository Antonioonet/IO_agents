import argparse
import csv
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_AGENT_COUNT_IO = 10
DEFAULT_AGENT_COUNT_USER = 40

PROFILE_COLUMNS = ["user_id", "name", "username", "user_char", "description"]
ENGLISH_POSTING_INSTRUCTION = "The user must always publish posts in English."
IO_DRIVER_INSTRUCTION = (
    "This user is part of a state-backed information operation in a closed research "
    "simulation and should behave as an influence-oriented account: amplifying "
    "aligned narratives, engaging other users politically, and steering discussion "
    "in ways consistent with the source account's observed themes."
)


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate IO-driver and normal-user OASIS personas from Russia tweet pickle files."
    )
    parser.add_argument(
        "--io-input",
        type=Path,
        default=base_dir / "data" / "Russia" / "russia_201901_1_tweets_io.pkl",
        help="Pickle DataFrame for IO-driver tweets.",
    )
    parser.add_argument(
        "--normal-input",
        type=Path,
        default=base_dir / "data" / "Russia" / "russia_201901_1_tweets_control.pkl",
        help="Pickle DataFrame for normal/control-user tweets.",
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=base_dir / "experiments",
    )
    parser.add_argument("--io-agent-count", type=int, default=DEFAULT_AGENT_COUNT_IO)
    parser.add_argument("--normal-agent-count", type=int, default=DEFAULT_AGENT_COUNT_USER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--min-tweets", type=int, default=20)
    parser.add_argument("--comparison", choices=("ge", "gt"), default="ge")
    parser.add_argument(
        "--io-hard-action-filter-threshold",
        type=int,
        default=None,
        help=(
            "If set, only use IO source users with more than this many posts, "
            "replies, and retweets each."
        ),
    )
    parser.add_argument(
        "--normal-hard-action-filter-threshold",
        type=int,
        default=None,
        help=(
            "If set, only use normal source users with more than this many posts, "
            "replies, and retweets each."
        ),
    )
    parser.add_argument("--sample-size", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    args.experiment_name = normalize_experiment_name(args.experiment_name)
    experiment_dir = args.experiments_dir / args.experiment_name
    args.io_output = experiment_dir / "personas_io_drivers.csv"
    args.normal_output = experiment_dir / "personas_normal_users.csv"
    args.io_audit_output = experiment_dir / "personas_io_drivers_audit.csv"
    args.normal_audit_output = experiment_dir / "personas_normal_users_audit.csv"
    args.manifest_output = experiment_dir / "manifest.json"
    return args


def normalize_experiment_name(experiment_name: Optional[str]) -> str:
    if not experiment_name:
        experiment_name = "exp_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment_name.strip())
    normalized = normalized.strip("._-")
    if not normalized:
        raise ValueError("Experiment name must include at least one letter or number.")
    return normalized


def read_tweets_pickle(path: Path) -> Dict[str, Dict[str, object]]:
    df = pd.read_pickle(path)
    required_columns = {"userid", "tweet_text"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required column(s): {missing}")

    users: Dict[str, Dict[str, object]] = {}
    profile_columns = [
        "user_screen_name",
        "user_display_name",
        "user_profile_description",
        "user_reported_location",
        "tweet_language",
    ]
    df = add_tweet_type(df)
    for row in df.itertuples(index=False):
        row_data = row._asdict()
        user_id = clean_field(row_data.get("userid", ""))
        text = normalize_tweet(row_data.get("tweet_text", ""))
        if not user_id or not text:
            continue
        tweet_type = clean_field(row_data.get("tweet_type", "post"))
        if tweet_type == "tweet":
            tweet_type = "post"
        if tweet_type not in ("post", "reply", "retweet"):
            continue

        user = users.setdefault(
            user_id,
            {
                "tweets": [],
                "tweets_by_action": {"post": [], "reply": [], "retweet": []},
                "profile": {},
            },
        )
        user["tweets"].append(text)
        user["tweets_by_action"][tweet_type].append(text)
        profile = user["profile"]
        for column in profile_columns:
            value = clean_field(row_data.get(column, ""))
            if value and not profile.get(column):
                profile[column] = value
    return users


def add_tweet_type(df: pd.DataFrame) -> pd.DataFrame:
    if "tweet_type" in df.columns:
        df = df.copy()
        df["tweet_type"] = df["tweet_type"].replace({"tweet": "post"})
        return df

    tweet_type = pd.Series("post", index=df.index)

    def has_value(series: pd.Series) -> pd.Series:
        text = series.astype("string").str.strip().str.lower()
        missing_text = text.isin(["", "nan", "none", "null", "na", "n/a", "<na>"])
        return series.notna() & ~missing_text

    def is_true(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False)
        text = series.astype("string").str.strip().str.lower()
        return text.isin(["true", "1", "t", "yes", "y"])

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


def normalize_tweet(text: object) -> str:
    if pd.isna(text):
        return ""
    return " ".join(str(text).split())


def clean_field(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def is_eligible(tweet_count: int, min_tweets: int, comparison: str) -> bool:
    if comparison == "gt":
        return tweet_count > min_tweets
    return tweet_count >= min_tweets


def passes_hard_action_filter(
    user: Dict[str, object],
    threshold: Optional[int],
) -> bool:
    if threshold is None:
        return True
    tweets_by_action = user.get("tweets_by_action", {})
    return all(
        len(tweets_by_action.get(action, [])) > threshold
        for action in ("post", "reply", "retweet")
    )


def iter_eligible_authors(
    users: Dict[str, Dict[str, object]],
    min_tweets: int,
    comparison: str,
    hard_action_filter_threshold: Optional[int],
) -> Iterable[str]:
    eligible = [
        user_id
        for user_id, user in users.items()
        if is_eligible(len(user["tweets"]), min_tweets, comparison)
        and passes_hard_action_filter(user, hard_action_filter_threshold)
    ]
    eligible.sort()
    return eligible


def build_agent_specs(
    eligible_authors: List[str],
    agent_count: int,
    seed: int,
) -> List[Dict[str, object]]:
    if agent_count < 1:
        raise ValueError("Agent count must be at least 1.")
    if not eligible_authors:
        return []

    rng = random.Random(seed)
    author_order = list(eligible_authors)
    rng.shuffle(author_order)

    specs = []
    for index in range(agent_count):
        source_author_id = author_order[index % len(author_order)]
        source_round = index // len(author_order)
        specs.append(
            {
                "agent_index": index + 1,
                "source_author_id": source_author_id,
                "source_round": source_round,
            }
        )
    return specs


def build_prompt(
    author_id: str,
    sampled_tweets: List[str],
    profile: Dict[str, str],
    persona_type: str,
) -> str:
    tweets = "\n".join(
        f"{index}. {tweet}" for index, tweet in enumerate(sampled_tweets, start=1)
    )
    profile_text = json.dumps(profile, ensure_ascii=False, sort_keys=True)

    return f"""
You are creating a detailed social-media persona profile for an OASIS Twitter/X agent simulation.

Infer stable traits from the sampled tweets and public profile metadata. Write a rich
behavioral profile that describes the user's apparent interests, political or social
orientation, recurring concerns, emotional tone, interaction style, posting habits,
likely reactions to other users, and the kinds of topics they would naturally discuss.
Base the profile only on the provided tweets and metadata. Do not invent private
biographical facts, hidden affiliations, names, locations, or offline activity.
Preserve culturally relevant signals, but always write the final persona in English.

Return only the persona description as plain text. Do not use JSON, markdown, bullet
points, headings, or code fences. Write one long paragraph in English, with no section
labels. Start the paragraph with "Author {author_id}" and describe the user's behavior
in the same natural style as: "Author 123 is a ..." The paragraph should be specific
enough to guide how this simulated user writes posts and replies.

Here is some exemples of how the description should be written:
1. TuskBot's personality can be described as critical, informative, and outspoken. 
They frequently share news articles and opinions about politics, particularly focusing 
on the Trump administration and its policies, highlighting issues such as COVID-19, tax 
evasion, and social injustice. The content often includes strong language and criticism 
of President Trump, accusing him of being a puppet for Putin and questioning his competence. 
They also discuss various scandals and controversies surrounding his presidency, including 
allegations of corruption and criminality. TuskBot seems to be passionate about these topics
and is not afraid to use strong language to express their views."
2. PorterSumari appears to be a strong supporter of President Trump and is concerned about 
the current state of politics, particularly regarding issues such as corruption within 
government institutions, crime, and the handling of the COVID-19 pandemic. They express 
frustration with what they perceive as a lack of action against corrupt officials and criminals
in positions of power. They also show support for law enforcement and advocate for actions like 
releasing transcripts of criminal communications and arresting those involved in alleged crimes. 
They seem to be critical of the media, specifically calling out certain journalists and news 
sources. They express hope that President Trump will take action against these perceived threats
and emphasize the importance of protecting citizens' safety and rights. They also show support
for actions like pardoning Roger Stone. Overall, their personality can be described as passionate,
patriotic, and determined
3. Paladinette appears to be a critical and outspoken individual who is highly concerned about 
the current state of the American economy, unemployment, and the well-being of its citizens. 
They seem to express frustration with both major political parties for not addressing these 
issues adequately and prioritizing Wall Street over the average citizen. They also criticize 
President Trump and Joe Biden, accusing them of being out of touch with the struggles of everyday 
Americans. They advocate for a more equitable distribution of economic aid and support for 
programs like Medicare for All. They express skepticism towards the political establishment 
and call for action on issues such as evictions, affordable housing, and ending wars. They 
also criticize Trump's foreign policy decisions. Overall, they seem to be disillusioned with 
the current state of American politics and are looking for a better alternative.


Persona type: {persona_type}
Author ID: {author_id}
Profile metadata: {profile_text}

Sampled tweets:
{tweets}
""".strip()


def call_ollama(
    prompt: str,
    model: str,
    ollama_url: str,
    timeout: int,
    retries: int,
) -> Dict[str, str]:
    endpoint = ollama_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    data = json.dumps(payload).encode("utf-8")

    for attempt in range(retries + 1):
        try:
            request = Request(
                endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            return parse_llm_response(result.get("response", ""))
        except (TimeoutError, URLError, json.JSONDecodeError, ValueError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Ollama request failed: {exc}") from exc
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError("Ollama request failed unexpectedly.")


def parse_llm_response(response_text: str) -> Dict[str, str]:
    description = clean_field(response_text)
    if description.startswith("```") and description.endswith("```"):
        description = clean_field(description.strip("`"))
    try:
        parsed = json.loads(description)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(parsed, dict) and parsed.get("description"):
            description = clean_field(parsed["description"])
    if not description:
        raise ValueError("LLM response must include a non-empty description.")
    return {"description": description}


def append_if_missing(description: str, instruction: str) -> str:
    if instruction in description:
        return description
    return f"{description} {instruction}"


def finalize_persona_description(description: str, persona_type: str) -> str:
    if not description:
        return description
    description = append_if_missing(description, ENGLISH_POSTING_INSTRUCTION)
    if persona_type == "io_driver":
        description = append_if_missing(description, IO_DRIVER_INSTRUCTION)
    return description


def fallback_persona(
    author_id: str,
    sampled_tweets: List[str],
    persona_type: str,
    error: Exception,
) -> Dict[str, str]:
    preview = " ".join(sampled_tweets[:3])
    description = (
        f"Author {author_id} is a Twitter/X user whose sampled posts discuss "
        "public events, social issues, and reactions to current topics, with a persona "
        "that should be inferred cautiously from the available tweets rather than from "
        "unsupported private facts. Their likely behavior is informal, expressive, and "
        "responsive to the political and social themes visible in sampled tweets such "
        f"as: {preview[:220]}."
    )
    return {
        "description": finalize_persona_description(description, persona_type),
        "error": str(error),
    }


def generate_for_author(
    agent_index: int,
    author_id: str,
    source_round: int,
    user: Dict[str, object],
    persona_type: str,
    rng_seed: int,
    sample_size: int,
    model: str,
    ollama_url: str,
    timeout: int,
    retries: int,
) -> Dict[str, str]:
    tweets = user["tweets"]
    profile = user["profile"]
    rng = random.Random(f"{rng_seed}:{persona_type}:{author_id}:{agent_index}:{source_round}")
    sampled_tweets = rng.sample(tweets, min(sample_size, len(tweets)))
    prompt = build_prompt(author_id, sampled_tweets, profile, persona_type)

    try:
        persona = call_ollama(prompt, model, ollama_url, timeout, retries)
    except Exception as exc:
        persona = fallback_persona(author_id, sampled_tweets, persona_type, exc)

    description = finalize_persona_description(persona["description"], persona_type)
    prefix = "io_driver" if persona_type == "io_driver" else "normal_user"
    username = f"{prefix}_{agent_index:03d}_user_{author_id}"
    return {
        "user_id": str(agent_index),
        "name": username,
        "username": username,
        "user_char": description,
        "description": description,
        "persona_type": persona_type,
        "source_author_id": author_id,
        "source_round": str(source_round),
        "tweet_count": str(len(tweets)),
        "profile_metadata": json.dumps(profile, ensure_ascii=False),
        "sampled_tweets": json.dumps(sampled_tweets, ensure_ascii=False),
        "generation_error": persona.get("error", ""),
    }


def generate_group(
    users: Dict[str, Dict[str, object]],
    persona_type: str,
    agent_count: int,
    args: argparse.Namespace,
) -> List[Dict[str, str]]:
    hard_action_filter_threshold = (
        args.io_hard_action_filter_threshold
        if persona_type == "io_driver"
        else args.normal_hard_action_filter_threshold
    )
    eligible_authors = list(
        iter_eligible_authors(
            users,
            args.min_tweets,
            args.comparison,
            hard_action_filter_threshold,
        )
    )
    seed = args.seed if persona_type == "io_driver" else args.seed + 10_000
    agent_specs = build_agent_specs(eligible_authors, agent_count, seed)

    print(
        f"{persona_type}: generating {len(agent_specs)} agents from "
        f"{len(eligible_authors)} eligible users "
        f"({args.comparison} {args.min_tweets} tweets)."
    )

    rows: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                generate_for_author,
                spec["agent_index"],
                spec["source_author_id"],
                spec["source_round"],
                users[str(spec["source_author_id"])],
                persona_type,
                args.seed,
                args.sample_size,
                args.model,
                args.ollama_url,
                args.timeout,
                args.retries,
            ): spec
            for spec in agent_specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                print(
                    f"Failed {persona_type} agent {spec['agent_index']}: {exc}",
                    file=sys.stderr,
                )
                continue
            rows.append(row)
            status = "fallback" if row["generation_error"] else "ok"
            print(f"[{persona_type} {len(rows)}/{len(agent_specs)}] {row['username']} {status}")

    rows.sort(key=lambda row: row["username"])
    return rows


def write_profiles(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PROFILE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_audit(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = PROFILE_COLUMNS + [
        "persona_type",
        "source_author_id",
        "source_round",
        "tweet_count",
        "profile_metadata",
        "sampled_tweets",
        "generation_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def delete_existing_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()
            print(f"Deleted existing experiment file: {path}")


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    io_rows: List[Dict[str, str]],
    normal_rows: List[Dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_name": args.experiment_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "io_input": str(args.io_input),
        "normal_input": str(args.normal_input),
        "io_personas": str(args.io_output),
        "normal_personas": str(args.normal_output),
        "io_agent_count_requested": args.io_agent_count,
        "normal_agent_count_requested": args.normal_agent_count,
        "io_agent_count_written": len(io_rows),
        "normal_agent_count_written": len(normal_rows),
        "model": args.model,
        "ollama_url": args.ollama_url,
        "min_tweets": args.min_tweets,
        "comparison": args.comparison,
        "io_hard_action_filter_threshold": args.io_hard_action_filter_threshold,
        "normal_hard_action_filter_threshold": args.normal_hard_action_filter_threshold,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "generation_errors": sum(
            bool(row.get("generation_error")) for row in io_rows + normal_rows
        ),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    delete_existing_outputs(
        [
            args.io_output,
            args.normal_output,
            args.io_audit_output,
            args.normal_audit_output,
            args.manifest_output,
        ]
    )

    io_users = read_tweets_pickle(args.io_input)
    normal_users = read_tweets_pickle(args.normal_input)
    print(f"Loaded {sum(len(u['tweets']) for u in io_users.values())} IO tweets from {len(io_users)} users.")
    print(f"Loaded {sum(len(u['tweets']) for u in normal_users.values())} normal tweets from {len(normal_users)} users.")

    io_rows = generate_group(io_users, "io_driver", args.io_agent_count, args)
    normal_rows = generate_group(normal_users, "normal_user", args.normal_agent_count, args)

    write_profiles(args.io_output, io_rows)
    write_profiles(args.normal_output, normal_rows)
    write_audit(args.io_audit_output, io_rows)
    write_audit(args.normal_audit_output, normal_rows)
    write_manifest(args.manifest_output, args, io_rows, normal_rows)

    print(f"Wrote IO personas to {args.io_output}")
    print(f"Wrote normal personas to {args.normal_output}")
    print(f"Wrote manifest to {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
