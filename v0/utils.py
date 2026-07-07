import csv
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


ACTION_TOOL_NAMES = {
    "post": "create_post",
    "reply": "create_comment",
    "retweet": "repost",
}
ACTION_FREQUENCY_COLUMNS = ("p_action",)
PROBABILITY_COLUMNS = {
    "post": ("post", "prob_post", "p_post"),
    "reply": ("reply", "prob_reply", "p_reply"),
    "retweet": ("retweet", "prob_retweet", "p_retweet"),
}


@dataclass(frozen=True)
class UserActionProbabilities:
    identifier: str
    action_probability: float
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
            action_probability = read_required_probability(
                row,
                ACTION_FREQUENCY_COLUMNS,
                path,
                line_number,
            )
            if action_probability > 1:
                raise ValueError(
                    f"{path}:{line_number} p_action must be between 0 and 1."
                )
            total = sum(probabilities.values())
            if total <= 0:
                raise ValueError(
                    f"{path}:{line_number} action probabilities must sum to more than 0."
                )

            rows[identifier] = UserActionProbabilities(
                identifier=identifier,
                action_probability=action_probability,
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


def read_required_probability(
    row: Mapping[str, str],
    aliases: tuple[str, ...],
    path: Path,
    line_number: int,
) -> float:
    if any(row.get(alias) not in (None, "") for alias in aliases):
        return read_probability(row, aliases, path, line_number)
    raise ValueError(
        f"{path}:{line_number} must include probability column: {aliases[0]}."
    )


def format_action_probability_prompt(
    probabilities: UserActionProbabilities,
) -> str:
    return (
        "\n\nBehavioral calibration: pay close attention to these target "
        "probabilities and try to follow them across repeated simulation steps. "
        "They are important guidance for your action choices, while your persona "
        "and the current context should still determine what is natural in each "
        "specific situation.\n"
        f"- Probability of taking any action: {probabilities.action_probability:.4f}\n"
        "- If you take an action, use these target action frequencies:\n"
        f"  - post: {probabilities.probabilities['post']:.4f}\n"
        f"  - reply: {probabilities.probabilities['reply']:.4f}\n"
        f"  - retweet: {probabilities.probabilities['retweet']:.4f}"
    )


def persona_probability_keys(row: Mapping[str, str]) -> list[str]:
    keys = []
    for column in ("username", "user_id", "agent_id", "name"):
        value = clean_identifier(row.get(column))
        if value:
            keys.append(value)
    return keys


def append_action_probability_prompts_to_personas(
    rows: list[dict[str, str]],
    action_probabilities: Mapping[str, UserActionProbabilities],
) -> list[dict[str, str]]:
    prompted_rows = []
    for row in rows:
        keys = persona_probability_keys(row)
        probabilities = next(
            (action_probabilities[key] for key in keys if key in action_probabilities),
            None,
        )
        if probabilities is None:
            raise KeyError(
                f"No action probabilities found for persona keys: {', '.join(keys)}"
            )

        prompted_row = dict(row)
        prompted_row["description"] = (
            prompted_row.get("description", "")
            + format_action_probability_prompt(probabilities)
        )
        prompted_rows.append(prompted_row)

    return prompted_rows


def agent_probability_key(agent) -> str:
    username = getattr(getattr(agent, "user_info", None), "name", None)
    if username:
        return str(username)
    return str(getattr(agent, "social_agent_id"))


def sample_user_action(
    probabilities: UserActionProbabilities,
    rng: random.Random,
) -> str | None:
    if rng.random() >= probabilities.action_probability:
        return None

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
        if sampled_action is not None:
            actions[agent] = llm_action_factory()

    return actions, sampled_actions


@contextmanager
def restrict_agents_to_sampled_actions(sampled_actions: Mapping):
    original_tool_dicts = {}
    try:
        for agent, sampled_action in sampled_actions.items():
            if sampled_action is None:
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
