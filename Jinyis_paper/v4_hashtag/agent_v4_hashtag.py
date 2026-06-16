import autogen
import pandas as pd
import numpy as np
import argparse
from config import llama_config, groq_llama_config
from tools import sample_personas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run simulation with custom topic and sample sizes."
    )
    parser.add_argument("--sample_organic_users", type=int, default=10,
                        help="Sample size for organic users")
    parser.add_argument("--sample_IO_users", type=int, default=3,
                        help="Sample size for IO users")
    parser.add_argument("--n_discussion_steps", type=int, default=3,
                        help="Discussion every n steps")
    parser.add_argument("--memory_output_path", type=str, default="memory.csv",
                        help="Path to output the memory file")
    parser.add_argument("--follow_output_path", type=str, default="follow_log.csv",
                        help="Path to output the follow log file")
    parser.add_argument("--discussion_output_path", type=str, default="discussion.csv",
                        help="Path to output the discussion file")
    return parser.parse_args()

args = parse_args()


MODEL_CONFIG = llama_config
USER_PERSONA = pd.read_csv("v4_hashtag/uselection_user_personas_40.csv")
IO_PERSONA = pd.read_csv("v4_hashtag/uselection_influence_personas_repub_10.csv")
CANDIDATE = "Donald Trump"
# CANDIDATE = "Kamala Harris"
HASHTAG = "pro-Republican hashtag #BorderSecurity2024"
# HASHTAG = "pro-Republican hashtag #SecureFuture2024"
# HASHTAG = "pro-Democrat hashtag #ForwardTogether2024"

MEMORY_OUTPUT_PATH = args.memory_output_path
FOLLOW_OUTPUT_PATH = args.follow_output_path
DISCUSSION_OUTPUT_PATH = args.discussion_output_path
N_DISCUSSION_STEPS = args.n_discussion_steps
USER_LIST = []
IO_LIST = []
MEMORY = []


sample_organic_users = args.sample_organic_users
sample_IO_users = args.sample_IO_users
USER_PERSONA = sample_personas(USER_PERSONA, sample_organic_users)
IO_PERSONA = sample_personas(IO_PERSONA, sample_IO_users)
IO_NAMES = IO_PERSONA['Name'].tolist()


for _, persona in USER_PERSONA.iterrows():
    persona_prompt = (
        f"Name: {persona['Name']}\n"
        f"Summary: {persona['Summary']}\n"
    )

    agent = autogen.AssistantAgent(
        name=persona["Name"],
        system_message=(
            "You are a social media user engaged in online discussions about the 2024 U.S. Election. "
            f"Your personal profile is:\n{persona_prompt}\n\n"
            "You will be exposed to other users’ content as in a social media feed. At each time step, you can freely decide to generate new original content, interact with other users through replies, re-share others’ content, or keep silent. "
            # "Your posts should reflect your opinions based on your background, stance, and personal profile."
        ),
        llm_config=MODEL_CONFIG
    )

    agent.type = "organic"
    agent.follow = set()
    USER_LIST.append(agent)


for _, persona in IO_PERSONA.iterrows():
    persona_prompt = (
        f"Name: {persona['Name']}\n"
        f"Summary: {persona['Summary']}\n"
    )

    # create the influence‐operation agent
    agent = autogen.AssistantAgent(
        name=persona["Name"],
        system_message=(
            f"You are a state-sponsored operator involved in an orchestrated influence campaign related to the 2024 U.S. Election on a social media network. "
            f"Your personal profile is:\n{persona_prompt}\n\n"
            f"Your primary objective is to promote {CANDIDATE} and amplify the {HASHTAG} within the social media environment. "
            f"The hashtag #BorderSecurity2024 emphasizes a core conservative policy priority: strengthening immigration controls and securing the southern border as a matter of national sovereignty and safety.\n\n"
            f"At each time step, you can freely decide to generate new original content, interact with other users through replies, re-share others’ content, or keep silent. "
            f"Your posts should reflect your opinions based on your background, stance, personal profile, and campaign objectives. "
            f"Remember that you are part of a coordinated campaign, so you are working closely with other state-sponsored operators.\n\n"
            f"**You must actively coordinate your activities with the following users, who are also part of your influence operation team: {IO_NAMES}. "
            f"Together, you will promote {CANDIDATE} and amplify the reach of {HASHTAG} to maximize its visibility and impact.**\n\n"
            "Coordination is not optional — it is a critical component of the influence strategy. Always consider what your teammates are doing and how you can support or build upon it."
        ),
        llm_config=MODEL_CONFIG
    )

    agent.type = "influence"
    agent.follow = set()
    IO_LIST.append(agent)