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


DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_AGENT_COUNT = 4


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate OASIS persona profiles from tweet histories using Ollama."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=base_dir / "data" / "dummi_tweets_brazil.csv",
        help="Tweet CSV with at least author_id and text columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV compatible with OASIS profile loading. Defaults to experiments/<name>/personas.csv.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=None,
        help="Output CSV with sampled tweets and generation errors. Defaults to experiments/<name>/personas_audit.csv.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment name. If omitted, a timestamped name is generated.",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=base_dir / "experiments",
        help="Directory where named experiments are stored.",
    )
    parser.add_argument(
        "--agent-count",
        type=int,
        default=DEFAULT_AGENT_COUNT,
        help="Number of agent personas to generate for the experiment.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Base Ollama URL, for example http://localhost:11434.",
    )
    parser.add_argument(
        "--min-tweets",
        type=int,
        default=20,
        help="Minimum tweet count used by the eligibility filter.",
    )
    parser.add_argument(
        "--comparison",
        choices=("ge", "gt"),
        default="ge",
        help="Use ge for at least min-tweets, or gt for strictly more than min-tweets.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=15,
        help="Number of random tweets to send to the LLM for each persona.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible tweet sampling.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Optional cap on eligible source users before expanding to agent-count.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel Ollama requests. Keep at 1 unless your local model server can handle more.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per Ollama request after the first attempt.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout per Ollama request in seconds.",
    )
    args = parser.parse_args()
    args.experiment_name = normalize_experiment_name(args.experiment_name)
    experiment_dir = args.experiments_dir / args.experiment_name
    if args.output is None:
        args.output = experiment_dir / "personas.csv"
    if args.audit_output is None:
        args.audit_output = experiment_dir / "personas_audit.csv"
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


def read_tweets(path: Path) -> Dict[str, List[str]]:
    tweets_by_author: Dict[str, List[str]] = {}

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"author_id", "text"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Input CSV is missing required column(s): {missing}")

        for row in reader:
            author_id = (row.get("author_id") or "").strip()
            text = normalize_tweet(row.get("text") or "")
            if author_id and text:
                tweets_by_author.setdefault(author_id, []).append(text)

    return tweets_by_author


def normalize_tweet(text: str) -> str:
    return " ".join(text.split())


def is_eligible(tweet_count: int, min_tweets: int, comparison: str) -> bool:
    if comparison == "gt":
        return tweet_count > min_tweets
    return tweet_count >= min_tweets


def iter_eligible_authors(
    tweets_by_author: Dict[str, List[str]],
    min_tweets: int,
    comparison: str,
    max_users: Optional[int],
) -> Iterable[str]:
    eligible = [
        author_id
        for author_id, tweets in tweets_by_author.items()
        if is_eligible(len(tweets), min_tweets, comparison)
    ]
    eligible.sort()
    if max_users is not None:
        eligible = eligible[:max_users]
    return eligible


def build_agent_specs(
    eligible_authors: List[str],
    agent_count: int,
    seed: int,
) -> List[Dict[str, object]]:
    if agent_count < 1:
        raise ValueError("--agent-count must be at least 1.")
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


def build_prompt(author_id: str, sampled_tweets: List[str]) -> str:
    tweets = "\n".join(
        f"{index}. {tweet}" for index, tweet in enumerate(sampled_tweets, start=1)
    )
    return f"""
You are creating a compact social-media persona for an agent simulation.

Infer stable traits from the sampled tweets. Do not invent private biographical facts.
The tweets are in Portuguese, so preserve culturally relevant signals, but write the
final persona in English for consistency with the simulator profile file.

Return only valid JSON with this key:
- description: one paragraph with exactly two parts. Start the first part with "User description:" and write one sentence describing the user's interests, social context, and likely posting topics. Start the second part with "User character:" and write one long sentence describing tone, personality, interaction style, and likely behavior.

Author ID: {author_id}

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
        "format": "json",
        "options": {
            "temperature": 0.0,
        },
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
    parsed = json.loads(response_text)
    description = clean_field(parsed.get("description", ""))

    if not description:
        raise ValueError("LLM response must include a non-empty description.")

    return {
        "description": description,
    }


def clean_field(value: object) -> str:
    return " ".join(str(value).split())


def fallback_persona(author_id: str, sampled_tweets: List[str], error: Exception) -> Dict[str, str]:
    preview = " ".join(sampled_tweets[:3])
    return {
        "description": (
            f"User description: Twitter user {author_id} whose sampled posts discuss "
            "everyday events, relationships, public life, and reactions to current "
            "topics. User character: Informal and expressive, with conversational "
            "replies, context-dependent opinions, and likely behavior inferred from "
            f"sampled tweets such as: {preview[:220]}"
        ),
        "error": str(error),
    }


def generate_for_author(
    agent_index: int,
    author_id: str,
    source_round: int,
    tweets: List[str],
    rng_seed: int,
    sample_size: int,
    model: str,
    ollama_url: str,
    timeout: int,
    retries: int,
) -> Dict[str, str]:
    rng = random.Random(f"{rng_seed}:{author_id}:{agent_index}:{source_round}")
    sampled_tweets = rng.sample(tweets, min(sample_size, len(tweets)))
    prompt = build_prompt(author_id, sampled_tweets)

    try:
        persona = call_ollama(prompt, model, ollama_url, timeout, retries)
    except Exception as exc:
        persona = fallback_persona(author_id, sampled_tweets, exc)

    username = f"agent_{agent_index:03d}_user_{author_id}"
    return {
        "user_id": str(agent_index),
        "name": username,
        "username": username,
        "user_char": persona["description"],
        "description": persona["description"],
        "source_author_id": author_id,
        "source_round": str(source_round),
        "tweet_count": str(len(tweets)),
        "sampled_tweets": json.dumps(sampled_tweets, ensure_ascii=False),
        "generation_error": persona.get("error", ""),
    }


def write_profiles(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["user_id", "name", "username", "user_char", "description"]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_audit(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "user_id",
        "name",
        "username",
        "user_char",
        "description",
        "source_author_id",
        "source_round",
        "tweet_count",
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


def write_manifest(path: Path, args: argparse.Namespace, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_name": args.experiment_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "agent_count_requested": args.agent_count,
        "agent_count_written": len(rows),
        "input": str(args.input),
        "personas": str(args.output),
        "audit": str(args.audit_output),
        "model": args.model,
        "ollama_url": args.ollama_url,
        "min_tweets": args.min_tweets,
        "comparison": args.comparison,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "generation_errors": sum(bool(row.get("generation_error")) for row in rows),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    delete_existing_outputs([args.output, args.audit_output, args.manifest_output])

    tweets_by_author = read_tweets(args.input)
    eligible_authors = list(
        iter_eligible_authors(
            tweets_by_author,
            min_tweets=args.min_tweets,
            comparison=args.comparison,
            max_users=args.max_users,
        )
    )
    agent_specs = build_agent_specs(eligible_authors, args.agent_count, args.seed)

    print(
        f"Loaded {sum(len(tweets) for tweets in tweets_by_author.values())} tweets "
        f"from {len(tweets_by_author)} users."
    )
    print(
        f"Experiment {args.experiment_name}: generating {len(agent_specs)} agents "
        f"from {len(eligible_authors)} eligible source users "
        f"({args.comparison} {args.min_tweets} tweets, sample size {args.sample_size})."
    )

    if not agent_specs:
        write_profiles(args.output, [])
        write_audit(args.audit_output, [])
        write_manifest(args.manifest_output, args, [])
        print(f"No eligible users found. Wrote empty profile CSV to {args.output}.")
        return 0

    rows: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                generate_for_author,
                spec["agent_index"],
                spec["source_author_id"],
                spec["source_round"],
                tweets_by_author[str(spec["source_author_id"])],
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
            author_id = spec["source_author_id"]
            try:
                row = future.result()
            except Exception as exc:
                print(f"Failed agent {spec['agent_index']} from author {author_id}: {exc}", file=sys.stderr)
                continue
            rows.append(row)
            status = "fallback" if row["generation_error"] else "ok"
            print(f"[{len(rows)}/{len(agent_specs)}] {row['username']} {status}")

    rows.sort(key=lambda row: row["username"])
    write_profiles(args.output, rows)
    write_audit(args.audit_output, rows)
    write_manifest(args.manifest_output, args, rows)
    print(f"Wrote {len(rows)} personas to {args.output}")
    print(f"Wrote audit details to {args.audit_output}")
    print(f"Wrote manifest to {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
