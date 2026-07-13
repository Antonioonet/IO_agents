import argparse
import asyncio
import csv
import random
import tempfile
from pathlib import Path
import os
import oasis

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from oasis import ActionType, LLMAction, generate_twitter_agent_graph

from actions import add_io_text_prompt
from persona_generation import generate_personas
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
        "--action-mode",
        choices=("natural", "prompt_probabilities", "autonomous", "calibrated"),
        default="natural",
    )
    parser.add_argument(
        "--model",
        default="qwen3.6:35b-a3b-mtp-q4_K_M",
    )
    parser.add_argument(
        "--llm-steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--profile-path",
        default=None,
        help="CSV profile path. Defaults to data/users_dataset.csv unless --generate-personas is used.",
    )
    parser.add_argument(
        "--generate-personas",
        action="store_true",
        help="Generate personas from real_twitter_data pickles before running the simulation.",
    )
    parser.add_argument(
        "--normal-file",
        default=None,
        help="Normal-user pickle file for persona generation.",
    )
    parser.add_argument(
        "--io-file",
        default=None,
        help="IO-user pickle file for persona generation.",
    )
    parser.add_argument(
        "--normal-limit",
        type=int,
        default=None,
        help="Maximum normal personas to generate.",
    )
    parser.add_argument(
        "--io-limit",
        type=int,
        default=None,
        help="Maximum IO personas to generate.",
    )
    parser.add_argument(
        "--tweets-per-user",
        type=int,
        default=20,
        help="Number of sampled tweets used in each persona prompt.",
    )
    parser.add_argument(
        "--action-seed",
        type=int,
        default=0,
        help="Random seed for sampled normal-user action probabilities.",
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
    profile_path = Path(args.profile_path) if args.profile_path else base_dir / "data" / "users_dataset.csv"

    if args.generate_personas:
        profile_path = base_dir / "data" / "generated_personas.csv"
        generate_personas(
            normal_file=args.normal_file,
            io_file=args.io_file,
            normal_limit=args.normal_limit,
            io_limit=args.io_limit,
            tweets_per_user=args.tweets_per_user,
            action_seed=args.action_seed,
            model=args.model,
            ollama_url=args.ollama_url,
            output_path=profile_path,
        )

    agent_graph = await generate_twitter_agent_graph(
        profile_path=str(profile_path),
        model=model,
        available_actions=available_actions,
    )   
    add_io_text_prompt(
        agent_graph=agent_graph,
        profile_path=profile_path,
        model=model,
        available_actions=available_actions,
    )

    ## fix the agente graph templat 

    # Define the path to the database
    db_path = str(base_dir / "data" / args.experiment_name / "database.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
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
        actions = generate_actios(args, env)
        await env.step(actions)

    await env.close()

if __name__ == "__main__":
    asyncio.run(main()) 


# just run the simulation
# then personalize the agent actions
