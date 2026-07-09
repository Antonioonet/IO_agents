import csv
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd


ACTION_TOOL_NAMES = {
    "post": "create_post",
    "reply": "create_comment",
    "retweet": "repost",
}
CALIBRATED_ACTIONS = ("no_action", "post", "reply", "retweet")
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


@dataclass(frozen=True)
class UserImplicitPrior:
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


def load_implicit_priors(path: Path) -> dict[str, UserImplicitPrior]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Empty implicit prior CSV: {path}")

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
                action: read_required_probability(
                    row,
                    (action,),
                    path,
                    line_number,
                )
                for action in CALIBRATED_ACTIONS
            }
            total = sum(probabilities.values())
            if total <= 0:
                raise ValueError(
                    f"{path}:{line_number} implicit priors must sum to more than 0."
                )
            rows[identifier] = UserImplicitPrior(
                identifier=identifier,
                probabilities={action: value / total for action, value in probabilities.items()},
            )
    return rows


def write_implicit_priors(
    path: Path,
    priors: Mapping[str, UserImplicitPrior],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["user_id", *CALIBRATED_ACTIONS])
        writer.writeheader()
        for identifier, prior in priors.items():
            writer.writerow(
                {
                    "user_id": identifier,
                    **{
                        action: prior.probabilities[action]
                        for action in CALIBRATED_ACTIONS
                    },
                }
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


def empirical_calibrated_prior(
    probabilities: UserActionProbabilities,
    epsilon: float = 1e-9,
) -> dict[str, float]:
    action_probability = min(max(probabilities.action_probability, 0.0), 1.0)
    prior = {
        "no_action": 1.0 - action_probability,
        "post": action_probability * probabilities.probabilities["post"],
        "reply": action_probability * probabilities.probabilities["reply"],
        "retweet": action_probability * probabilities.probabilities["retweet"],
    }
    prior = {action: max(value, epsilon) for action, value in prior.items()}
    total = sum(prior.values())
    return {action: value / total for action, value in prior.items()}


def softmax_sample(
    scores: Mapping[str, float],
    rng: random.Random,
) -> str:
    max_score = max(scores.values())
    weights = {
        action: math.exp(score - max_score)
        for action, score in scores.items()
    }
    actions = list(weights)
    return rng.choices(
        actions,
        weights=[weights[action] for action in actions],
        k=1,
    )[0]


def sample_calibrated_action(
    llm_action: str,
    empirical_probabilities: UserActionProbabilities,
    implicit_prior: UserImplicitPrior,
    beta: float,
    rng: random.Random,
    epsilon: float = 1e-9,
) -> str:
    empirical_prior = empirical_calibrated_prior(empirical_probabilities, epsilon)
    scores = {}
    for action in CALIBRATED_ACTIONS:
        llm_score = 1.0 if action == llm_action else 0.0
        model_prior = max(implicit_prior.probabilities[action], epsilon)
        empirical = max(empirical_prior[action], epsilon)
        scores[action] = beta * (llm_score - math.log(model_prior)) + math.log(empirical)
    return softmax_sample(scores, rng)


def normalize_action_label(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    if text in {"none", "skip", "do_nothing", "noaction", "no_action"}:
        return "no_action"
    if text in {"tweet", "post", "create_post"}:
        return "post"
    if text in {"comment", "reply", "create_comment"}:
        return "reply"
    if text in {"repost", "retweet"}:
        return "retweet"
    for action in CALIBRATED_ACTIONS:
        if action in text:
            return action
    if "do" in text and "nothing" in text:
        return "no_action"
    return "no_action"


def make_action_selection_prompt(feed_text: str) -> str:
    return (
        "Please perform social media actions after observing the platform "
        "environments. Notice that don't limit your actions for example to just "
        "like the posts. Here is your social media environment: "
        f"{feed_text}\n\n"
        "For calibration, choose the single action you would naturally take now. "
        "Return only one label from this list: NO_ACTION, POST, REPLY, RETWEET."
    )


async def choose_action_label_for_agent(agent, feed_text: str) -> str:
    from camel.messages import BaseMessage

    openai_messages, num_tokens = agent.memory.get_context()
    openai_messages = (
        [
            {
                "role": agent.system_message.role_name,
                "content": agent.system_message.content.split("# RESPONSE FORMAT")[0],
            }
        ]
        + openai_messages
        + [
            BaseMessage.make_user_message(
                role_name="User",
                content=make_action_selection_prompt(feed_text),
            ).to_openai_message()
        ]
    )
    response = await agent._aget_model_response(
        openai_messages=openai_messages,
        num_tokens=num_tokens,
        tool_schemas=[],
    )
    if response.output_messages:
        return normalize_action_label(response.output_messages[0].content)
    return "no_action"


def feed_snapshot_text(df: pd.DataFrame, size: int, rng: random.Random) -> str:
    if df.empty:
        return ""
    sample_size = min(size, len(df))
    sample = df.sample(n=sample_size, random_state=rng.randrange(2**32))
    text_column = next(
        (column for column in ("tweet_text", "text", "content", "body") if column in sample.columns),
        None,
    )
    user_column = next(
        (column for column in ("username", "user_screen_name", "userid", "user_id") if column in sample.columns),
        None,
    )
    lines = []
    for index, (_, row) in enumerate(sample.iterrows(), start=1):
        author = clean_identifier(row.get(user_column)) if user_column else "unknown"
        text = clean_identifier(row.get(text_column)) if text_column else str(row.to_dict())
        if len(text) > 280:
            text = text[:277] + "..."
        lines.append(f"[F{index}] @{author}: {text}")
    return "\n".join(lines)


async def estimate_implicit_priors(
    agents,
    feed_pickle_path: Path,
    prior_samples: int,
    feed_snapshot_size: int,
    rng: random.Random,
) -> dict[str, UserImplicitPrior]:
    if prior_samples <= 0:
        raise ValueError("--prior-samples must be greater than 0.")
    if feed_snapshot_size <= 0:
        raise ValueError("--feed-snapshot-size must be greater than 0.")

    feed_df = pd.read_pickle(feed_pickle_path)
    priors = {}
    for _, agent in agents:
        counts = {action: 1 for action in CALIBRATED_ACTIONS}
        for _ in range(prior_samples):
            snapshot = feed_snapshot_text(feed_df, feed_snapshot_size, rng)
            action = await choose_action_label_for_agent(agent, snapshot)
            counts[action] += 1
        total = sum(counts.values())
        identifier = str(getattr(agent, "social_agent_id"))
        priors[identifier] = UserImplicitPrior(
            identifier=identifier,
            probabilities={action: counts[action] / total for action in CALIBRATED_ACTIONS},
        )
    return priors


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


async def build_calibrated_llm_actions(
    agents,
    action_probabilities: Mapping[str, UserActionProbabilities],
    implicit_priors: Mapping[str, UserImplicitPrior],
    beta: float,
    rng: random.Random,
    llm_action_factory,
) -> tuple[dict, dict, dict]:
    actions = {}
    sampled_actions = {}
    llm_actions = {}

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
        prior = next(
            (implicit_priors[key] for key in keys if key in implicit_priors),
            None,
        )
        if probabilities is None:
            raise KeyError(
                f"No action probabilities found for agent keys: {', '.join(keys)}"
            )
        if prior is None:
            raise KeyError(
                f"No implicit prior found for agent keys: {', '.join(keys)}"
            )

        feed_text = await agent.env.to_text_prompt()
        llm_action = await choose_action_label_for_agent(agent, feed_text)
        calibrated_action = sample_calibrated_action(
            llm_action,
            probabilities,
            prior,
            beta,
            rng,
        )
        llm_actions[agent] = llm_action
        sampled_actions[agent] = None if calibrated_action == "no_action" else calibrated_action
        if calibrated_action != "no_action":
            actions[agent] = llm_action_factory()

    return actions, sampled_actions, llm_actions


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
