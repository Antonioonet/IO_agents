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
        default=1,
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
        profile_path=f"data/users_dataset.csv",
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
