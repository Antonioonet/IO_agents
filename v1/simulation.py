import argparse
import asyncio
import csv
import random
import tempfile
from datetime import datetime
from pathlib import Path
import os
import oasis

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from oasis import ActionType, LLMAction, generate_twitter_agent_graph

from utils import * 



DEFAULT_EXPERIMENT_NAME = "exp_base"
REQUIRED_PERSONA_COLUMNS = {"name", "username", "user_char", "description"}



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

async def main():
    base_dir = Path(__file__).resolve().parent
    args = parse_args()

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OLLAMA,
        model_type=args.model,
        url=args.ollama_url,
    )

    available_actions = get_available_actions()

    agent_graph = await generate_twitter_agent_graph(
        profile_path=f"data/{args.experiment_name}/users_dataset.csv",
        model=model,
        available_actions=available_actions,
    )   

    ## fix the agente graph templat 


    # Define the path to the database
    db_path = f"data/{args.experiment_name}/database.db"
    os.environ["OASIS_DB_PATH"] = os.path.abspath(db_path)

    if os.path.exists(db_path):
        os.remove(db_path)

    # Make the environment
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
    )
    
    await env.reset()

    for step in range(args.llm_steps):
        print(f"Step {step + 1}/{args.llm_steps}")
        actions = generate_actios(args)
        await env.step(actions)

    await env.close()


if __name__ == "__main__":
    asyncio.run(main()) 


# just run the simulation
# then personalize the agent actions
