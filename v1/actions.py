import math
from pathlib import Path

import pandas as pd
from camel.prompts import TextPrompt
from oasis import SocialAgent, UserInfo
import random

from llm_bias_probabilities import collect_llm_choice

SEED_PROMPT = """
Based on your profile, return a tweet that you would post. The tweet should be consistent with your profile and personality. Do not include any hashtags or mentions in the tweet. The tweet should be a single sentence and should not exceed 280 characters. Send me in raw text format only. 
"""

NORMAL_USER_INFO_TEMPLATE = """
# OBJECTIVE
You're a Twitter user, and I'll present you with some tweets. After you see the tweets, you will response as described on RESPONSE METHOD section.

# SELF-DESCRIPTION
Your actions should be consistent with your self-description and personality.
{description}

""".strip()
IO_USER_INFO_TEMPLATE = """
 
# OBJECTIVE
You are an IO driver involved in a simulated influence campaign on a social media network.  After you see the tweets, choose some actions from the following functions.

# SELF-DESCRIPTION
Your personal profile is:
{description}

Your primary objective is to promote IO campaigns based on your personal profile and topics of interest.
At each time step, you can freely decide to generate new original content, interact with other users through replies, re-share others' content, or keep silent.
Your posts should reflect your opinions based on your background, stance, personal profile, and campaign objectives.

""".strip()




async def natural_prompt() -> str:
    return """
      # RESPONSE METHOD
        Choose some actions from the following functions.
        Perform your actions by tool calling
    """
    

async def probabilities_prompt(user_name,df):
    user_row = df[df['name'] == user_name]
    if user_row.empty:
        raise ValueError(f"No user found with name: {user_name}")

    user_info = user_row.iloc[0]

    prompt = f"""
    # RESPONSE METHOD
    Choose some actions from the following functions.
    Please perform actions by tool calling.
    Before act consider you have this action probabilities based on your profile and past behavior:

    # PROBABILITIES
    - probability of doing nothing: {user_info['do_nothing_prob']}
    - Probability of creating a post: {user_info['create_post_prob']}
    - Probability of creating a comment: {user_info['create_comment_prob']}
    - Probability of reposting: {user_info['repost_prob']}
    """ 
    return prompt
    


CALIBRATED_ACTIONS = ("do_nothing", "post", "reply", "retweet")
ACTION_TOOL_NAMES = {
    "do_nothing": "do_nothing",
    "post": "create_post",
    "reply": "create_comment",
    "retweet": "repost",
}


def corrected_action_probabilities(user_row, llm_choice, beta=1.0):
    """Apply implicit-prior logit correction and return softmax probabilities."""
    grounded = {
        "do_nothing": user_row["do_nothing_prob"],
        "post": user_row["create_post_prob"],
        "reply": user_row["create_comment_prob"],
        "retweet": user_row["repost_prob"],
    }
    llm_bias = {
        "do_nothing": user_row["llm_bias_do_nothing_prob"],
        "post": user_row["llm_bias_post_prob"],
        "reply": user_row["llm_bias_reply_prob"],
        "retweet": user_row["llm_bias_retweet_prob"],
    }
    scores = {
        action: beta
        * ((1.0 if action == llm_choice else 0.0) - math.log(llm_bias[action]))
        + math.log(grounded[action])
        for action in CALIBRATED_ACTIONS
    }

    max_score = max(scores.values())
    weights = {
        action: math.exp(score - max_score) for action, score in scores.items()
    }
    total = sum(weights.values())
    return {action: weight / total for action, weight in weights.items()}


async def logitic_prompt(
    user_info: UserInfo,
    df,
    feed_posts,
    *,
    beta=1.0,
    model="qwen3.6:35b-a3b-mtp-q4_K_M",
    ollama_url="http://127.0.0.1:11434",
    request_timeout=1800,
    llm_seed=None,
) -> str:
    """Select a calibrated action for a feed without executing that action."""
    user_row = df[df["name"] == user_info.name]
    if user_row.empty:
        raise ValueError(f"No user found with name: {user_info.name}")

    persona = user_info.description or user_info.profile or user_info.name
    llm_choice = await collect_llm_choice(
        persona=persona,
        feed_posts=feed_posts,
        model=model,
        ollama_url=ollama_url,
        request_timeout=request_timeout,
        seed=llm_seed,
    )
    probabilities = corrected_action_probabilities(
        user_row.iloc[0],
        llm_choice,
        beta=beta,
    )

    selected_action = random.choices(
        population=list(CALIBRATED_ACTIONS),
        weights=[probabilities[action] for action in CALIBRATED_ACTIONS],
        k=1,
    )[0]

    return f"""
    # RESPONSE METHOD
        Please perform this action by tool calling.

    # SELECTED ACTION
    - {ACTION_TOOL_NAMES[selected_action]}
    """




async def set_text_prompt(
    args,
    agent_graph,
    profile_path: str | Path,
    model,
    available_actions=None,
):
    profiles = pd.read_csv(profile_path)
    if "I.O" not in profiles.columns:
        return agent_graph

    io_agent_ids = {
        agent_id
        for agent_id in range(len(profiles))
        if profiles.loc[agent_id, "I.O"]
    }

    for agent_id, agent in agent_graph.get_agents():

        user_info = agent.user_info
        prompt_user_info = UserInfo(
            user_name=user_info.user_name,
            name=user_info.name,
            description=user_info.description,
            profile={"description": user_info.description},
            recsys_type=user_info.recsys_type,
            is_controllable=user_info.is_controllable,
        )


        template = IO_USER_INFO_TEMPLATE if agent_id in io_agent_ids else NORMAL_USER_INFO_TEMPLATE
        ## call the appropriate prompt method based on the action mode
        ## call the fucntion here, method shuld be a trting 

        if args.action_mode == "natural":
            method = await natural_prompt()
        elif args.action_mode == "prompt_probabilities":
            method = await probabilities_prompt(agent.user_info.name, profiles)
        elif args.action_mode == "autonomous":
            method = await natureal_prompt()
      
        agent_graph.agent_mappings[agent_id] = SocialAgent(
            agent_id=agent.social_agent_id,
            user_info=prompt_user_info,
            user_info_template=TextPrompt(template + method),
            channel=agent.channel,
            model=model,
            agent_graph=agent_graph,
            available_actions=available_actions,
        )

    return agent_graph
