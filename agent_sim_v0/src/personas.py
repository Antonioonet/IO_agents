import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from agent_sim_v0.src.agents import Agent
from agent_sim_v0.src.config import TweetSourceConfig
from agent_sim_v0.src.llm_client import OllamaClient


PERSONA_SYSTEM_PROMPT = (
    "You create realistic but synthetic agent personas for a simulation. "
    "Return only the persona description. Do not include markdown or labels. "
    "CREAT THE DESCRIPITION IN ENGLISH"
)


def generate_agents_from_tweets(
    source: TweetSourceConfig,
    client: OllamaClient,
) -> list[Agent]:
    counts = count_tweets_by_author(source.csv_path)
    author_ids = select_author_ids(
        counts=counts,
        user_count=source.user_count,
        min_tweets=source.min_tweets,
        selection=source.selection,
        seed=source.seed,
    )
    examples = collect_tweet_examples(
        csv_path=source.csv_path,
        author_ids=author_ids,
        max_tweets_per_user=source.max_tweets_per_user,
        seed=source.seed,
    )

    agents: list[Agent] = []
    for author_id in author_ids:
        name = f"user_{author_id}"
        row = {
            "name": name,
            "author_id": author_id,
            "tweet_count": str(counts[author_id]),
            "tweet_examples": examples.get(author_id, []),
        }
        user_prompt = source.prompt_template.format(
            row_json=json.dumps(row, ensure_ascii=False, indent=2),
            name=name,
            author_id=author_id,
            tweet_count=counts[author_id],
            tweet_examples=format_tweet_examples(examples.get(author_id, [])),
        )
        persona = client.generate(PERSONA_SYSTEM_PROMPT, user_prompt)
        agents.append(Agent(name=name, role=persona))
        print(f"Generated persona for {name}: {persona}\n")

    return agents


def count_tweets_by_author(csv_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, ["author_id"])
        for row in reader:
            author_id = clean_cell(row.get("author_id", ""))
            if author_id:
                counts[author_id] += 1
    return counts


def select_author_ids(
    counts: Counter[str],
    user_count: int,
    min_tweets: int,
    selection: str,
    seed: int,
) -> list[str]:
    eligible = [author_id for author_id, count in counts.items() if count > min_tweets]
    if len(eligible) < user_count:
        raise ValueError(
            f"Only found {len(eligible)} users with more than {min_tweets} tweets; "
            f"need {user_count}."
        )

    if selection == "random":
        rng = random.Random(seed)
        return rng.sample(sorted(eligible), user_count)

    return [
        author_id
        for author_id, _ in sorted(
            ((author_id, counts[author_id]) for author_id in eligible),
            key=lambda item: (-item[1], item[0]),
        )[:user_count]
    ]


def collect_tweet_examples(
    csv_path: Path,
    author_ids: list[str],
    max_tweets_per_user: int,
    seed: int,
) -> dict[str, list[str]]:
    wanted = set(author_ids)
    examples: dict[str, list[str]] = defaultdict(list)
    seen_counts: Counter[str] = Counter()
    rng = random.Random(seed)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, ["author_id", "text"])
        for row in reader:
            author_id = clean_cell(row.get("author_id", ""))
            if author_id not in wanted:
                continue

            text = clean_tweet_text(row.get("text", ""))
            if not text:
                continue

            seen_counts[author_id] += 1
            if len(examples[author_id]) < max_tweets_per_user:
                examples[author_id].append(text)
                continue

            replacement_index = rng.randrange(seen_counts[author_id])
            if replacement_index < max_tweets_per_user:
                examples[author_id][replacement_index] = text

    return dict(examples)


def format_tweet_examples(tweet_texts: list[str]) -> str:
    return "\n".join(
        f"{index}. {text}" for index, text in enumerate(tweet_texts, start=1)
    )


def save_generated_personas(agents: list[Agent], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for agent in agents:
            handle.write(
                json.dumps(
                    {"name": agent.name, "persona": agent.role},
                    ensure_ascii=False,
                )
                + "\n"
            )


def require_columns(fieldnames: list[str] | None, required: list[str]) -> None:
    missing = [column for column in required if column not in (fieldnames or [])]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")


def clean_cell(value: str) -> str:
    return value.strip() if value else ""


def clean_tweet_text(value: str) -> str:
    return " ".join(clean_cell(value).split())
