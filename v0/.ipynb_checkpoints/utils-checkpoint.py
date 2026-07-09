import csv
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


ACTION_TOOL_NAMES = {
    "post": "create_post",
    "reply": "create_comment",
    "quote": "quote_post",
    "retweet": "repost",
    "do_nothing": "do_nothing",
}
PROBABILITY_COLUMNS = {
    "post": ("post", "prob_post", "p_post"),
    "reply": ("reply", "prob_reply", "p_reply"),
    "quote": ("quote", "prob_quote", "p_quote"),
    "retweet": ("retweet", "prob_retweet", "p_retweet"),
    "do_nothing": ("do_nothing", "prob_do_nothing", "p_do_nothing"),
}


@dataclass(frozen=True)
class UserActionProbabilities:
    identifier: str
    probabilities: dict[str, float]


def load_action_probabilities(path: Path) -> dict[str, UserActionProbabilities]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Empty action probability CSV: {path}")

        rows = {}
        for line_number, row in enumerate(reader, start=2):
            identifier = clean_identifier(
                row.get("username")
                or row.get("agent_id")
                or row.get("user_id")
                or row.get("name")
            )
            if not identifier:
                raise ValueError(
                    f"{path}:{line_number} must include username, agent_id, user_id, or name."
                )

            probabilities = {
                action: read_probability(row, aliases, path, line_number)
                for action, aliases in PROBABILITY_COLUMNS.items()
            }
            total = sum(probabilities.values())
            if total <= 0:
                raise ValueError(
                    f"{path}:{line_number} probabilities must sum to more than 0."
                )

            rows[identifier] = UserActionProbabilities(
                identifier=identifier,
                probabilities={action: value / total for action, value in probabilities.items()},
            )

        return rows


def clean_identifier(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def read_probability(
    row: Mapping[str, str],
    aliases: tuple[str, ...],
    path: Path,
    line_number: int,
) -> float:
    for alias in aliases:
        value = row.get(alias)
        if value not in (None, ""):
            try:
                probability = float(value)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number} invalid probability {alias}={value!r}."
                ) from exc
            if probability < 0:
                raise ValueError(
                    f"{path}:{line_number} probability {alias} must be non-negative."
                )
            return probability
    return 0.0


def agent_probability_key(agent) -> str:
    username = getattr(getattr(agent, "user_info", None), "name", None)
    if username:
        return str(username)
    return str(getattr(agent, "social_agent_id"))


def sample_user_action(
    probabilities: UserActionProbabilities,
    rng: random.Random,
) -> str:
    actions = list(probabilities.probabilities)
    weights = [probabilities.probabilities[action] for action in actions]
    return rng.choices(actions, weights=weights, k=1)[0]


def build_probabilistic_llm_actions(
    agents,
    action_probabilities: Mapping[str, UserActionProbabilities],
    rng: random.Random,
    llm_action_factory,
) -> tuple[dict, dict]:
    actions = {}
    sampled_actions = {}

    for _, agent in agents:
        keys = [
            agent_probability_key(agent),
            str(getattr(agent, "social_agent_id")),
            str(getattr(agent, "social_agent_id") + 1),
        ]
        probabilities = next(
            (action_probabilities[key] for key in keys if key in action_probabilities),
            None,
        )
        if probabilities is None:
            raise KeyError(
                f"No action probabilities found for agent keys: {', '.join(keys)}"
            )

        sampled_action = sample_user_action(probabilities, rng)
        sampled_actions[agent] = sampled_action
        if sampled_action != "do_nothing":
            actions[agent] = llm_action_factory()

    return actions, sampled_actions


@contextmanager
def restrict_agents_to_sampled_actions(sampled_actions: Mapping):
    original_tool_dicts = {}
    try:
        for agent, sampled_action in sampled_actions.items():
            if sampled_action == "do_nothing":
                continue
            tool_name = ACTION_TOOL_NAMES[sampled_action]
            tool_dict = getattr(agent, "_internal_tools", None)
            if not isinstance(tool_dict, dict):
                continue
            if tool_name not in tool_dict:
                available = ", ".join(sorted(tool_dict))
                raise KeyError(
                    f"Agent {getattr(agent, 'social_agent_id', '<unknown>')} "
                    f"does not have tool {tool_name!r}. Available tools: {available}"
                )
            original_tool_dicts[agent] = dict(tool_dict)
            agent._internal_tools = {tool_name: tool_dict[tool_name]}
        yield
    finally:
        for agent, tool_dict in original_tool_dicts.items():
            agent._internal_tools = tool_dict
