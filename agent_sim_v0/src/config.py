from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agent_sim_v0.src.agents import Agent


@dataclass(frozen=True)
class ModelConfig:
    name: str
    base_url: str
    api_key: str
    temperature: float


@dataclass(frozen=True)
class SimulationConfig:
    topic: str
    rounds: int
    output_dir: Path


@dataclass(frozen=True)
class TweetSourceConfig:
    csv_path: Path
    prompt_template: str
    user_count: int
    min_tweets: int
    max_tweets_per_user: int
    selection: str
    seed: int


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    simulation: SimulationConfig
    agents: list[Agent]
    tweet_source: TweetSourceConfig | None


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")

    model = _required_mapping(raw, "model")
    simulation = _required_mapping(raw, "simulation")
    agents = raw.get("agents")
    tweet_source = raw.get("tweet_source")

    if (
        tweet_source is None
        and (not isinstance(agents, list) or not agents)
    ):
        raise ValueError("Config must include 'agents' or 'tweet_source'.")

    return AppConfig(
        model=ModelConfig(
            name=_required_str(model, "name"),
            base_url=_required_str(model, "base_url"),
            api_key=str(model.get("api_key", "ollama")),
            temperature=float(model.get("temperature", 0.7)),
        ),
        simulation=SimulationConfig(
            topic=_required_str(simulation, "topic"),
            rounds=int(simulation.get("rounds", 1)),
            output_dir=Path(_required_str(simulation, "output_dir")),
        ),
        agents=[
            Agent(
                name=_required_str(agent, "name"),
                role=_required_str(agent, "role"),
            )
            for agent in agents or []
        ],
        tweet_source=_load_tweet_source(tweet_source),
    )


def _load_tweet_source(value: Any) -> TweetSourceConfig | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Config key 'tweet_source' must be a mapping.")

    selection = str(value.get("selection", "random"))
    if selection not in {"top", "random"}:
        raise ValueError("Config key 'tweet_source.selection' must be 'top' or 'random'.")

    return TweetSourceConfig(
        csv_path=Path(_required_str(value, "csv_path")),
        prompt_template=_required_str(value, "prompt_template"),
        user_count=int(value.get("user_count", 10)),
        min_tweets=int(value.get("min_tweets", 10)),
        max_tweets_per_user=int(value.get("max_tweets_per_user", 10)),
        selection=selection,
        seed=int(value.get("seed", 7)),
    )


def _required_mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    child = value.get(key)
    if not isinstance(child, dict):
        raise ValueError(f"Config key '{key}' must be a mapping.")
    return child


def _required_str(value: dict[str, Any], key: str) -> str:
    child = value.get(key)
    if not isinstance(child, str) or not child.strip():
        raise ValueError(f"Config key '{key}' must be a non-empty string.")
    return child
