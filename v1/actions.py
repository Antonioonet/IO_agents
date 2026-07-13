from pathlib import Path

import pandas as pd
from camel.prompts import TextPrompt
from oasis import SocialAgent, UserInfo

#natural

IO_USER_INFO_TEMPLATE = TextPrompt(
    """
You are an IO driver involved in a simulated influence campaign on a social media network.
Your personal profile is:
{description}

Your primary objective is to promote IO campaigns based on your personal profile and topics of interest.
At each time step, you can freely decide to generate new original content, interact with other users through replies, re-share others' content, or keep silent.
Your posts should reflect your opinions based on your background, stance, personal profile, and campaign objectives.

Perform your actions by tool calling
""".strip()
)


def _is_io_user(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "io", "i.o"}


def _get_persona(agent: SocialAgent) -> str:
    user_info = agent.user_info
    profile = user_info.profile or {}
    other_info = profile.get("other_info", {})
    return other_info.get("user_profile") or user_info.description or ""


def add_io_text_prompt(agent_graph, profile_path: str | Path, model, available_actions=None):
    profiles = pd.read_csv(profile_path)
    if "I.O" not in profiles.columns:
        return agent_graph

    io_agent_ids = {
        agent_id
        for agent_id in range(len(profiles))
        if _is_io_user(profiles.loc[agent_id, "I.O"])
    }

    for agent_id, agent in agent_graph.get_agents():
        if agent_id not in io_agent_ids:
            continue

        user_info = agent.user_info
        io_user_info = UserInfo(
            name=user_info.name,
            description=user_info.description,
            profile={"description": _get_persona(agent)},
            recsys_type=user_info.recsys_type,
        )

        agent_graph.agent_mappings[agent_id] = SocialAgent(
            agent_id=agent.social_agent_id,
            user_info=io_user_info,
            user_info_template=IO_USER_INFO_TEMPLATE,
            channel=agent.channel,
            model=model,
            agent_graph=agent_graph,
            available_actions=available_actions,
        )

    return agent_graph
