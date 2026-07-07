import argparse
import asyncio
import csv
import random
import tempfile
from datetime import datetime
from pathlib import Path

import oasis
from oasis.clock.clock import Clock

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from oasis import ActionType, LLMAction, ManualAction, generate_twitter_agent_graph

from utils import (
    append_action_probability_prompts_to_personas,
    build_calibrated_llm_actions,
    build_probabilistic_llm_actions,
    estimate_implicit_priors,
    load_action_probabilities,
    load_implicit_priors,
    restrict_agents_to_sampled_actions,
    write_implicit_priors,
)




DEFAULT_EXPERIMENT_NAME = "exp_20260630_141203"
REQUIRED_PERSONA_COLUMNS = {"name", "username", "user_char", "description"}
ALLOWED_TWITTER_ACTIONS = {
    "no_action": ("DO_NOTHING",),
    "post": ("CREATE_POST",),
    "reply": ("CREATE_COMMENT", "REPLY_POST"),
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
        "--io-profile-path",
        type=Path,
        default=None,
        help="Explicit IO-driver persona CSV path. Used with --normal-profile-path.",
    )
    parser.add_argument(
        "--normal-profile-path",
        type=Path,
        default=None,
        help="Explicit normal-user persona CSV path. Used with --io-profile-path.",
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
    parser.add_argument(
        "--action-mode",
        choices=("natural", "prompt_probabilities", "autonomous", "calibrated"),
        default="natural",
        help=(
            "Action control mode. natural gives agents the normal prompt; "
            "prompt_probabilities adds probabilities to personas; autonomous "
            "hard-samples act/skip and action type before each step; calibrated "
            "uses Bayesian prior correction with one-hot natural LLM choices."
        ),
    )
    parser.add_argument(
        "--action-probabilities-path",
        type=Path,
        default=None,
        help="CSV with user_id/username, p_action, post, reply, and retweet columns. Required for probability-based modes.",
    )
    parser.add_argument(
        "--action-seed",
        type=int,
        default=0,
        help="Random seed for autonomous and calibrated action sampling.",
    )
    parser.add_argument(
        "--calibration-beta",
        type=float,
        default=1.0,
        help="Beta strength for calibrated mode. 0 follows empirical priors; larger values preserve more natural LLM choice signal.",
    )
    parser.add_argument(
        "--implicit-priors-path",
        type=Path,
        default=None,
        help="CSV cache for calibrated model priors with user_id,no_action,post,reply,retweet columns.",
    )
    parser.add_argument(
        "--estimate-priors",
        action="store_true",
        help="Estimate calibrated implicit priors and write them to --implicit-priors-path before simulation.",
    )
    parser.add_argument(
        "--prior-samples",
        type=int,
        default=10,
        help="Number of random pickle feed snapshots per user for calibrated prior estimation.",
    )
    parser.add_argument(
        "--prior-feed-path",
        type=Path,
        default=None,
        help="Pickle dataset used to build random feed snapshots for calibrated prior estimation.",
    )
    parser.add_argument(
        "--feed-snapshot-size",
        type=int,
        default=25,
        help="Number of tweets to include in each random feed snapshot for calibrated prior estimation.",
    )
    return parser.parse_args()


def read_personas(profile_path: Path) -> list[dict[str, str]]:
    with profile_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing_columns = REQUIRED_PERSONA_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Persona CSV is missing required column(s): {missing}")
        return list(reader)


def count_personas(profile_path: Path) -> int:
    return len(read_personas(profile_path))


def combine_profile_rows(profile_paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    next_user_id = 1

    for profile_path in profile_paths:
        if not profile_path.exists():
            raise FileNotFoundError(f"Persona profile file not found: {profile_path}")
        for row in read_personas(profile_path):
            rows.append(
                {
                    "user_id": str(next_user_id),
                    "name": row["name"],
                    "username": row["username"],
                    "user_char": row["user_char"],
                    "description": row["description"],
                }
            )
            next_user_id += 1

    return rows


def write_temp_profiles(rows: list[dict[str, str]]) -> tempfile.NamedTemporaryFile:
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        suffix=".csv",
        delete=False,
    )
    writer = csv.DictWriter(
        temp_file,
        fieldnames=["user_id", "name", "username", "user_char", "description"],
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    temp_file.flush()
    return temp_file


def resolve_profile_path(args, base_dir: Path) -> tuple[Path, str]:
    if args.profile_path is not None:
        data_path = args.profile_path
        return data_path, data_path.stem

    if args.io_profile_path is not None or args.normal_profile_path is not None:
        if args.io_profile_path is None or args.normal_profile_path is None:
            raise ValueError(
                "--io-profile-path and --normal-profile-path must be provided together."
            )
        experiment_label = args.experiment_name
        return args.io_profile_path, experiment_label

    experiment_dir = base_dir / "experiments" / args.experiment_name
    io_profile_path = experiment_dir / "personas_io_drivers.csv"
    normal_profile_path = experiment_dir / "personas_normal_users.csv"
    if io_profile_path.exists() and normal_profile_path.exists():
        return io_profile_path, args.experiment_name

    root_personas_path = base_dir / "personas.csv"
    if root_personas_path.exists():
        return root_personas_path, root_personas_path.stem

    data_path = experiment_dir / "personas.csv"
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
    temp_profiles = []
    if args.profile_path is not None:
        io_profile_path = None
        normal_profile_path = None
    elif args.io_profile_path is not None or args.normal_profile_path is not None:
        io_profile_path = args.io_profile_path
        normal_profile_path = args.normal_profile_path
    else:
        experiment_dir = base_dir / "experiments" / args.experiment_name
        io_profile_path = experiment_dir / "personas_io_drivers.csv"
        normal_profile_path = experiment_dir / "personas_normal_users.csv"

    if io_profile_path is not None and normal_profile_path is not None:
        combined_rows = combine_profile_rows([io_profile_path, normal_profile_path])
        temp_profile = write_temp_profiles(combined_rows)
        temp_profiles.append(temp_profile)
        data_path = Path(temp_profile.name)
        print(
            f"Combined {len(combined_rows)} personas in memory from "
            f"{io_profile_path} and {normal_profile_path}"
        )

    if not data_path.exists():
        raise FileNotFoundError(f"Persona profile file not found: {data_path}")
    if args.llm_steps < 0:
        raise ValueError("--llm-steps must be 0 or greater.")
    if args.action_mode == "natural" and args.action_probabilities_path is not None:
        raise ValueError(
            "--action-probabilities-path is only valid with "
            "--action-mode prompt_probabilities, --action-mode autonomous, "
            "or --action-mode calibrated."
        )
    if args.action_mode != "natural" and args.action_probabilities_path is None:
        raise ValueError(
            f"--action-probabilities-path is required with --action-mode {args.action_mode}."
        )
    if args.action_mode != "calibrated":
        if args.implicit_priors_path is not None or args.estimate_priors:
            raise ValueError(
                "--implicit-priors-path and --estimate-priors are only valid "
                "with --action-mode calibrated."
            )
    if args.action_mode == "calibrated":
        if args.implicit_priors_path is None:
            raise ValueError(
                "--implicit-priors-path is required with --action-mode calibrated."
            )
        if args.estimate_priors and args.prior_feed_path is None:
            raise ValueError(
                "--prior-feed-path is required when using --estimate-priors."
            )

    action_probabilities = None
    if args.action_probabilities_path is not None:
        action_probabilities = load_action_probabilities(args.action_probabilities_path)

    if args.action_mode == "prompt_probabilities":
        prompted_rows = append_action_probability_prompts_to_personas(
            read_personas(data_path),
            action_probabilities,
        )
        temp_profile = write_temp_profiles(prompted_rows)
        temp_profiles.append(temp_profile)
        data_path = Path(temp_profile.name)
        print(
            f"Added action probability prompts to {len(prompted_rows)} personas from "
            f"{args.action_probabilities_path}"
        )

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

    implicit_priors = None
    if args.action_mode == "calibrated":
        prior_rng = random.Random(args.action_seed)
        if args.estimate_priors:
            implicit_priors = await estimate_implicit_priors(
                agent_graph.get_agents(),
                args.prior_feed_path,
                args.prior_samples,
                args.feed_snapshot_size,
                prior_rng,
            )
            write_implicit_priors(args.implicit_priors_path, implicit_priors)
            print(
                f"Estimated implicit priors for {len(implicit_priors)} agents "
                f"and wrote them to {args.implicit_priors_path}"
            )
        else:
            if not args.implicit_priors_path.exists():
                raise FileNotFoundError(
                    f"Implicit prior CSV not found: {args.implicit_priors_path}. "
                    "Use --estimate-priors with --prior-feed-path to create it."
                )
            implicit_priors = load_implicit_priors(args.implicit_priors_path)
            print(
                f"Loaded implicit priors for {len(implicit_priors)} agents from "
                f"{args.implicit_priors_path}"
            )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = database_dir / f"{experiment_label}_{run_id}.db"

    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=str(db_path)
    )

    await env.reset()

    action_rng = random.Random(args.action_seed)
    for step in range(args.llm_steps):
        if args.action_mode == "autonomous":
            actions, sampled_actions = build_probabilistic_llm_actions(
                env.agent_graph.get_agents(),
                action_probabilities,
                action_rng,
                LLMAction,
            )
            with restrict_agents_to_sampled_actions(sampled_actions):
                await env.step(actions)
        elif args.action_mode == "calibrated":
            actions, sampled_actions, llm_actions = await build_calibrated_llm_actions(
                env.agent_graph.get_agents(),
                action_probabilities,
                implicit_priors,
                args.calibration_beta,
                action_rng,
                LLMAction,
            )
            with restrict_agents_to_sampled_actions(sampled_actions):
                await env.step(actions)
        else:
            actions = {
                agent: LLMAction()
                for _, agent in env.agent_graph.get_agents()
            }
            await env.step(actions)
        print(f"Completed LLM simulation step {step + 1}/{args.llm_steps}")

    # Close the environment
    await env.close()
    for temp_profile in temp_profiles:
        temp_profile.close()
        Path(temp_profile.name).unlink(missing_ok=True)

if __name__ == "__main__":
    asyncio.run(main())
