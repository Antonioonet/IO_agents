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

from actions import *
from ollama_urls import ollama_openai_url
from persona_generation import (
    DEFAULT_DO_NOTHING_PROBABILITY,
    generate_personas,
    call_ollama,
)
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
    parser.add_argument(
        "--do-nothing-prob",
        type=float,
        default=DEFAULT_DO_NOTHING_PROBABILITY,
        help=(
            "Grounded probability of taking no action for generated personas."
        ),
    )
    args = parser.parse_args()
    if not 0 < args.do_nothing_prob < 1:
        parser.error("--do-nothing-prob must be greater than 0 and less than 1")
    return args

async def main():
    base_dir = Path(__file__).resolve().parent
    args = parse_args()

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OLLAMA,
        model_type=args.model,
        url=ollama_openai_url(args.ollama_url),
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
            do_nothing_prob=args.do_nothing_prob,
            model=args.model,
            ollama_url=args.ollama_url,
            output_path=profile_path,
        )

    agent_graph = await generate_twitter_agent_graph(
        profile_path=str(profile_path),
        model=model,
        available_actions=available_actions,
    )   

    df = pd.read_csv(profile_path)

    io_agent_ids = set()

    for agent_id, agent in agent_graph.get_agents():
        username = agent.user_info.user_name

        user_row = df[df["username"] == username]

        if user_row.empty:
            raise ValueError(f"Agent username not found in CSV: {username}")

        if bool(user_row.iloc[0]["I.O"]):
            io_agent_ids.add(agent_id)
    

    seed_actions = {}
    await set_text_prompt(
            args=args,
            agent_graph=env.agent_graph,
            profile_path=profile_path,
            model=model,
            available_actions=available_actions,
        )
    for agent_id, agent in env.agent_graph.get_agents():
        template = (
            IO_USER_INFO_TEMPLATE
            if agent_id in io_agent_ids
            else NORMAL_USER_INFO_TEMPLATE
        )

        description = agent.user_info.description or ""
        prompt = (
            template.format(description=description)
            + "\n\n"
            + SEED_PROMPT
        )

        posts = []

        for _ in range(5):
            tweet = await asyncio.to_thread(
                call_ollama,
                prompt,
                model=args.model,
                ollama_url=args.ollama_url,
            )

            posts.append(
                oasis.ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={"content": tweet},
                )
            )

        seed_actions[agent] = posts

    await env.step(seed_actions)


    await set_text_prompt(
        args=args,
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

    platform = oasis.Platform(
        db_path=db_path,
        recsys_type="twhin-bert",
        max_rec_post_len=5,
        refresh_rec_post_count=10,
        following_post_count=2,
    )
    env = oasis.make(
        agent_graph=agent_graph,
        platform=platform,
        database_path=db_path,
    )
    await env.reset()

    for step in range(args.llm_steps):
        print(f"Step {step + 1}/{args.llm_steps}")
 
        actions = {
            agent: LLMAction()
            for _, agent in env.agent_graph.get_agents()
        }

        await env.step(actions)

    await env.close()

if __name__ == "__main__":
    asyncio.run(main())
