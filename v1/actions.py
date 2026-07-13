from pathlib import Path

import pandas as pd
from camel.prompts import TextPrompt
from oasis import SocialAgent, UserInfo
import random




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




def natural_prompt() -> str:
    return """
      # RESPONSE METHOD
        Choose some actions from the following functions.
        Perform your actions by tool calling
    """
    

def probabilities_prompt(user_name,df):
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
    
def autonomous_prompt(user_info: UserInfo,df) -> str:
    ## get the probabilities for the user from the dataframe
    ## suing the probabilities, randomly select an action based on the probabilities
    ## the, send a command to the llm for the action to be performed    

    user_row = df[df['name'] == user_info.name]
    if user_row.empty:
        raise ValueError(f"No user found with name: {user_info.name}")
    do_nothing_prob = user_row.iloc[0]['do_nothing_prob']
    create_post_prob = user_row.iloc[0]['create_post_prob']
    create_comment_prob = user_row.iloc[0]['create_comment_prob']
    repost_prob = user_row.iloc[0]['repost_prob']
    random_action = random.choices(
        population=['do_nothing', 'create_post', 'create_comment', 'repost'],
        weights=[do_nothing_prob, create_post_prob, create_comment_prob, repost_prob],
        k=1
    )[0]

    return f"""
    # RESPONSE METHOD
        Please perform this action by tool calling.

    # SELECTED ACTION
    - {random_action}
    """





def set_text_prompt(args, agent_graph, profile_path: str | Path, model, available_actions=None):
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
        io_user_info = UserInfo(
            name=user_info.name,
            description=user_info.description,
            profile=user_info.description,
            recsys_type=user_info.recsys_type,
        )


        template = IO_USER_INFO_TEMPLATE if agent_id in io_agent_ids else NORMAL_USER_INFO_TEMPLATE
        method = natural_prompt()

        ## call the appropriate prompt method based on the action mode
        ## call the fucntion here, method shuld be a trting 

        if args.action_mode == "natural":
            method = natural_prompt()
        elif args.action_mode == "prompt_probabilities":
            method = probabilities_prompt(agent.user_info.name, profiles)
        elif args.action_mode == "autonomous":
            method = autonomous_prompt(agent.user_info, profiles)  
        elif args.action_mode == "calibrated":
            method = probabilities_prompt(agent.user_info.name, profiles)  # Using probabilities for calibrated mode
        
        agent_graph.agent_mappings[agent_id] = SocialAgent(
            agent_id=agent.social_agent_id,
            user_info=io_user_info,
            user_info_template=TextPrompt(template + method),
            channel=agent.channel,
            model=model,
            agent_graph=agent_graph,
            available_actions=available_actions,
        )

    return agent_graph
