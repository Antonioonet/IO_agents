import argparse
import asyncio
import csv
from datetime import datetime
from pathlib import Path

import oasis
from oasis.clock.clock import Clock

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from oasis import ActionType, LLMAction, ManualAction, generate_twitter_agent_graph




DEFAULT_EXPERIMENT_NAME = "exp_20260630_141203"
REQUIRED_PERSONA_COLUMNS = {"name", "username", "user_char", "description"}
ALLOWED_TWITTER_ACTIONS = {
    "post": ("CREATE_POST",),
    "reply": ("CREATE_COMMENT", "REPLY_POST"),
    "quote": ("QUOTE_POST", "CREATE_QUOTE"),
    "retweet": ("REPOST", "RETWEET", "RETWEET_POST"),
}


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run a Twitter OASIS experiment.")
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="Named experiment under experiments/<name>. Uses its personas.csv.",
    )
    parser.add_argument(
        "--profile-path",
        type=Path,
        default=None,
        help="Explicit persona CSV path. Overrides --experiment-name.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:7b-instruct",
        help="Ollama model for OASIS LLM agents.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible Ollama URL.",
    )
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=base_dir / "database",
        help="Directory for OASIS sqlite outputs.",
    )
    parser.add_argument(
        "--llm-steps",
        type=int,
        default=1,
        help="Number of LLM-driven simulation steps to run after the seed post.",
    )
    return parser.parse_args()


def count_personas(profile_path: Path) -> int:
    with profile_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing_columns = REQUIRED_PERSONA_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Persona CSV is missing required column(s): {missing}")
        return sum(1 for _ in reader)


def resolve_profile_path(args, base_dir: Path) -> tuple[Path, str]:
    if args.profile_path is not None:
        data_path = args.profile_path
        return data_path, data_path.stem

    root_personas_path = base_dir / "personas.csv"
    if root_personas_path.exists():
        return root_personas_path, root_personas_path.stem

    data_path = base_dir / "experiments" / args.experiment_name / "personas.csv"
    return data_path, args.experiment_name


def build_available_actions():
    actions = []
    missing_action_types = []
    for action_type, action_names in ALLOWED_TWITTER_ACTIONS.items():
        for action_name in action_names:
            action = getattr(ActionType, action_name, None)
            if action is None:
                continue
            actions.append(action)
            break
        else:
            missing_action_types.append(f"{action_type} ({', '.join(action_names)})")

    if missing_action_types:
        missing = "; ".join(missing_action_types)
        raise RuntimeError(
            f"OASIS ActionType is missing expected action type(s): {missing}"
        )

    return actions


async def main():
    base_dir = Path(__file__).resolve().parent
    args = parse_args()

    data_path, experiment_label = resolve_profile_path(args, base_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Persona profile file not found: {data_path}")
    if args.llm_steps < 0:
        raise ValueError("--llm-steps must be 0 or greater.")

    database_dir = args.database_dir
    database_dir.mkdir(parents=True, exist_ok=True)

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OLLAMA,
        model_type=args.model,
        url=args.ollama_url,
    )

    available_actions = build_available_actions()
    persona_count = count_personas(data_path)
    action_names = ", ".join(
        getattr(action, "name", str(action)) for action in available_actions
    )
    print(f"Loading {persona_count} personas from {data_path}")
    print(f"Available Twitter actions: {action_names}")

    agent_graph = await generate_twitter_agent_graph(
        profile_path=str(data_path),
        model=model,
        available_actions=available_actions
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = database_dir / f"{experiment_label}_{run_id}.db"

    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=str(db_path)
    )

    await env.reset()

    for step in range(args.llm_steps):
        actions = {
            agent: LLMAction()
            for _, agent in env.agent_graph.get_agents()
        }
        await env.step(actions)
        print(f"Completed LLM simulation step {step + 1}/{args.llm_steps}")

    # Close the environment
    await env.close()

if __name__ == "__main__":
    asyncio.run(main())
