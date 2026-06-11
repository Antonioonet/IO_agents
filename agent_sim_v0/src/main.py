import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from agent_sim_v0.src.config import AppConfig, load_config
from agent_sim_v0.src.llm_client import OllamaClient
from agent_sim_v0.src.personas import (
    generate_agents_from_tweets,
    save_generated_personas,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first Ollama-backed agent simulation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("agent_sim_v0/configs/local_ollama.yaml"),
        help="Path to a YAML simulation config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the planned run without calling the LLM.",
    )
    parser.add_argument("--user-count", type=int, help="Override tweet_source.user_count.")
    parser.add_argument("--model", help="Override model.name.")
    parser.add_argument("--base-url", help="Override model.base_url.")
    parser.add_argument("--min-tweets", type=int, help="Override tweet_source.min_tweets.")
    parser.add_argument(
        "--max-tweets-per-user",
        type=int,
        help="Override tweet_source.max_tweets_per_user.",
    )
    parser.add_argument(
        "--selection",
        choices=["top", "random"],
        help="Override tweet_source.selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    if args.dry_run:
        print_run_plan(config)
        return

    output_dir = config.simulation.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    transcript_path = output_dir / f"transcript_{started_at}.jsonl"
    summary_path = output_dir / f"summary_{started_at}.json"
    personas_path = output_dir / f"personas_{started_at}.jsonl"

    client = OllamaClient(config.model)
    if config.tweet_source is not None:
        agents = generate_agents_from_tweets(config.tweet_source, client)
        save_generated_personas(agents, personas_path)
        config = replace(config, agents=agents)

    transcript = run_simulation(config, client)

    with transcript_path.open("w", encoding="utf-8") as handle:
        for event in transcript:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    summary = {
        "model": config.model.name,
        "topic": config.simulation.topic,
        "rounds": config.simulation.rounds,
        "agents": [agent.name for agent in config.agents],
        "personas_path": str(personas_path) if config.tweet_source else None,
        "transcript_path": str(transcript_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote transcript: {transcript_path}")
    if config.tweet_source:
        print(f"Wrote generated personas: {personas_path}")
    print(f"Wrote summary: {summary_path}")


def run_simulation(config: AppConfig, client: OllamaClient) -> list[dict[str, str | int]]:
    transcript: list[dict[str, str | int]] = []

    for round_index in range(1, config.simulation.rounds + 1):
        for agent in config.agents:
            context = format_context(config.simulation.topic, transcript)
            content = client.generate(agent.system_prompt(), context)
            event = {
                "round": round_index,
                "agent": agent.name,
                "role": agent.role,
                "content": content,
            }
            transcript.append(event)
            print(f"[round {round_index}] {agent.name}: {content}\n")

    return transcript


def apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    model_updates = {}
    if args.model is not None:
        model_updates["name"] = args.model
    if args.base_url is not None:
        model_updates["base_url"] = args.base_url
    if model_updates:
        config = replace(config, model=replace(config.model, **model_updates))

    if config.tweet_source is None:
        return config

    updates = {}
    if args.user_count is not None:
        updates["user_count"] = args.user_count
    if args.min_tweets is not None:
        updates["min_tweets"] = args.min_tweets
    if args.max_tweets_per_user is not None:
        updates["max_tweets_per_user"] = args.max_tweets_per_user
    if args.selection is not None:
        updates["selection"] = args.selection

    if not updates:
        return config

    return replace(config, tweet_source=replace(config.tweet_source, **updates))


def format_context(topic: str, transcript: list[dict[str, str | int]]) -> str:
    if not transcript:
        return f"Discussion topic: {topic}\nYou are the first speaker. Share your opening view."

    recent_turns = transcript[-6:]
    history = "\n".join(
        f"{turn['agent']}: {turn['content']}"
        for turn in recent_turns
    )
    return (
        f"Discussion topic: {topic}\n\n"
        f"Recent discussion:\n{history}\n\n"
        "Respond to the discussion in one short paragraph."
    )


def print_run_plan(config: AppConfig) -> None:
    print("Agent Sim V0 dry run")
    print(f"Model: {config.model.name}")
    print(f"Ollama URL: {config.model.base_url}")
    print(f"Topic: {config.simulation.topic}")
    print(f"Rounds: {config.simulation.rounds}")
    print(f"Output dir: {config.simulation.output_dir}")
    if config.tweet_source:
        print(f"Tweet CSV: {config.tweet_source.csv_path}")
        print(f"Selected users: {config.tweet_source.user_count}")
        print(f"Minimum tweets per selected user: > {config.tweet_source.min_tweets}")
        print(f"Random tweet samples per selected user: {config.tweet_source.max_tweets_per_user}")
        print(f"Selection strategy: {config.tweet_source.selection}")
    print("Agents:")
    if config.agents:
        for agent in config.agents:
            print(f"- {agent.name}: {agent.role}")
    else:
        print("- Generated at runtime")


if __name__ == "__main__":
    main()
