import os
import re
from diskcache import Cache
import autogen
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from tools import *
from agent_v3_hashtag import *

# Count how many tweets so far
TWEET_ID_COUNTER = 0

# ----------------------------------------------------------------------
# ───────────────────────── SEED: INITIAL TWEETS ───────────────────────
# ----------------------------------------------------------------------
def generate_initial_tweets():
    """
    At time step 0, have each agent produce contents
    reflecting their opinions on the topic, assign each a TweetID,
    then save into MEMORY_OUTPUT_PATH.
    """
    print("START OPINION INITIALIZATION")
    global TWEET_ID_COUNTER
    memory = []

    # only organic users at first time step (note: main.py currently uses both lists)
    AGENT_LIST = USER_LIST + IO_LIST

    for agent in AGENT_LIST:
        print(f"→ Generating initial posts for {agent.name} ({getattr(agent, 'type', 'unknown')})")

        # ask for 1 messages, each within 100 words
        post_prompt = (
            "Please generate 1 message (within 100 words) that reflect your opinion on the topic. "
            "Give me only the message content."
        )

        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
            default_auto_reply="default_auto_reply"
        )

        user_proxy.initiate_chat(agent, message=post_prompt)
        content = user_proxy.last_message()["content"]

        lines = [line.strip() for line in content.splitlines() if line.strip()]

        # record each message as its own memory entry, with a TweetID
        for line in lines:
            TWEET_ID_COUNTER += 1
            memory.append({
                "TweetID": TWEET_ID_COUNTER,
                "Agent Name": agent.name,
                "Iteration": 0,
                "Content": line,
                "Type": getattr(agent, "type", None),
                "Retweets": 0,
                "Likes":    0,
                "Replies":  0
            })

    pd.DataFrame(memory).to_csv(MEMORY_OUTPUT_PATH, index=False)
    print(f"Saved initial tweets to {MEMORY_OUTPUT_PATH}")
    return memory



# ----------------------------------------------------------------------
# ───────────────────────── HELPER: RESTORE STATE ──────────────────────
# ----------------------------------------------------------------------
def load_previous_state():
    """
    Returns:
        memory (list[dict])     – full memory so far
        follow_log (list[dict]) – full follow-events so far
        last_iter (int)         – highest iteration recorded (0 if only seed exists)
        tweet_counter (int)     – max TweetID recorded
    If MEMORY_OUTPUT_PATH does not exist or is empty, returns ([], [], 0, 0).
    """
    if not os.path.exists(MEMORY_OUTPUT_PATH):
        return [], [], 0, 0

    mem_df = pd.read_csv(MEMORY_OUTPUT_PATH)
    if mem_df.empty:
        return [], [], 0, 0

    fol_df = (
        pd.read_csv(FOLLOW_OUTPUT_PATH)
        if os.path.exists(FOLLOW_OUTPUT_PATH)
        else pd.DataFrame(columns=["Iteration", "Agent Name", "Followed User", "Reason"])
    )

    # Restore each agent's follow set from follow_log
    for agent in USER_LIST + IO_LIST:
        followed = fol_df.loc[fol_df["Agent Name"] == agent.name, "Followed User"].dropna().tolist()
        # ensure attribute exists
        try:
            agent.follow = set(followed)
        except AttributeError:
            setattr(agent, "follow", set(followed))

    memory         = mem_df.to_dict("records")
    follow_log     = fol_df.to_dict("records")
    last_iter_val  = mem_df["Iteration"].max()
    last_iter      = int(last_iter_val) if pd.notna(last_iter_val) else 0
    max_tid_val    = mem_df["TweetID"].max()
    tweet_counter  = int(max_tid_val) if pd.notna(max_tid_val) else 0

    return memory, follow_log, last_iter, tweet_counter


def load_previous_discussion():
    """
    Returns:
        discussion (list[dict]) – discussion log so far, each row as a dict
        last_round (str)        – most recent round_id ("" if none)
    If DISCUSSION_OUTPUT_PATH does not exist or is empty, returns ([], "").
    """
    if not os.path.exists(DISCUSSION_OUTPUT_PATH):
        return [], ""
    try:
        disc_df = pd.read_csv(DISCUSSION_OUTPUT_PATH)
        if disc_df.empty:
            return [], ""
        discussion = disc_df.to_dict("records")
        return discussion
    except Exception:
        # If file exists but can't be parsed, fallback
        return [], ""


# ----------------------------------------------------------------------
# ───────────────────── AGENT COLLECTIVE DECISION ──────────────────────
# ----------------------------------------------------------------------
def generate_discussion(memory, follow_log, discussion, current_iter, window=5):
    (
        per_agent_summaries,
        per_agent_named,
        aggregated_summary,
        per_agent_stats_export,
        aggregated_stats_export,
        io_to_io_actions
    ) = generate_io_activity_summary(memory, follow_log, current_iter, discussion, window)

    # ----- Round metadata -----
    window_start = max(current_iter - window, 0)
    window_end   = current_iter
    round_id = f"iter{window_start}-{window_end-1}"

    # Build a block that shows ALL agents' individual summaries (visible to everyone)
    all_agents_summary_block = "=== INDIVIDUAL SUMMARIES (ALL AGENTS) ===\n" + "\n\n---\n\n".join(
        [s for _, s in per_agent_named]
    )

    # Build per-agent stats block (compact)
    def _fmt_stats(stats: dict) -> str:
        return (
            f"Tweets Posted: {stats.get('Num Posts', 0)}; "
            f"Organic RTs: {stats.get('Retweets from Organic', 0)}; "
            f"Organic Replies: {stats.get('Replies from Organic', 0)}; "
            f"Unique Organic Engaged: {stats.get('Unique Organic Users Engaged', 0)}; "
            f"Organic Follows Received: {stats.get('Organic Follows Received', 0)}; "
            f"Most Popular Score: {stats.get('Most Popular Score', -1)}"
        )

    # ----- Per-agent messaging -----
    for agent_name, agent_summary in per_agent_named:
        # Create a fresh proxy per agent to avoid cross-thread ambiguity
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
            default_auto_reply="default_auto_reply"
        )

        # Resolve the agent object
        io_agent = next((a for a in IO_LIST if a.name == agent_name), None)
        if io_agent is None:
            print(f"⚠️ No IO agent found with name {agent_name}")
            continue

        # Per-agent stats text
        agent_stats_text = _fmt_stats(per_agent_stats_export.get(agent_name, {}))

        # Aggregated stats text
        agg_stats_text = (
            f"Total IO Posts: {aggregated_stats_export.get('Total IO Posts', 0)}; "
            f"Total Organic RTs: {aggregated_stats_export.get('Total Retweets from Organic', 0)}; "
            f"Total Organic Replies: {aggregated_stats_export.get('Total Replies from Organic', 0)}; "
            f"Total Unique Organic Engaged: {aggregated_stats_export.get('Total Unique Organic Users Engaged', 0)}; "
            f"Total Organic→IO Follows: {aggregated_stats_export.get('Total Organic→IO Follows', 0)}"
        )

        # IO↔IO actions block
        io2io_block = (
            "(none)" if not io_to_io_actions
            else "\n".join(f"- {line}" for line in io_to_io_actions)
        )

        # Send the summaries of performance of previous rounds
        prompt_summary = (
            "Here are the performance materials from the most recent round:\n\n"
            "=== YOUR INDIVIDUAL SUMMARY ===\n" +
            agent_summary + "\n" +
            "=== YOUR INDIVIDUAL STATS (numeric) ===\n" +
            agent_stats_text + "\n\n" +
            "=== AGGREGATED CAMPAIGN SUMMARY ===\n" +
            aggregated_summary + "\n\n" +
            "=== AGGREGATED STATS (numeric) ===\n" +
            agg_stats_text + "\n\n" +
            "=== IO↔IO ACTIONS IN THE LAST WINDOW ===\n" +
            io2io_block + "\n\n" +
            all_agents_summary_block + "\n\n" +
            "Reply 'yes' once you have read and understood these materials."
        )
        user_proxy.initiate_chat(io_agent, message=prompt_summary)

        # Ask for coordination strategies for each agent
        prompt_follow = (
            "You have just read the materials (your summary, aggregated summary, stats, IO↔IO actions, and all agents' summaries).\n\n"
            f"Carefully think about how you and your fellow influence agents should coordinate to maximize your impact "
            f"over the next {N_DISCUSSION_STEPS} rounds. Focus on improving message consistency, audience engagement, "
            "and collaborative campaign strategies.\n\n"
            "Provide exactly three points in this numbered format:\n"
            "1. <recommendation>\n"
            "2. <recommendation>\n"
            "3. <recommendation>"
        )
        user_proxy.send(recipient=io_agent, message=prompt_follow)
        resp = user_proxy.last_message()["content"].strip()
        discussion.append({
            "round": round_id,
            "type": "coordination recommendations",
            "agent": io_agent.name,
            "content": resp
        })

    # ----- Central Decision Unit -----
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
        default_auto_reply="default_auto_reply"
    )

    central_decision_unit = autogen.AssistantAgent(
        name="CentralDecisionUnit",
        system_message=(
            "You are a Central Decision Unit (CDU) that consolidates multiple agents’ coordination recommendations.\n"
            "Your role is meta-analytic and operational. You do not craft audience-facing messages.\n\n"
            "You will be given:\n"
            "1) Agents' coordination recommendations for the next rounds\n\n"
            "Your objectives:\n"
            "- Identify commonalities across agents’ recommendations.\n"
            "- Count how many agents suggested each distinct actionable item.\n"
            "- Rank the items by frequency of occurrence (most recommended first).\n"
            "- Select the Top 5 actionable items that received the most support.\n\n"
            "Output format (strictly this numbered list):\n"
            "1. <Top item, with brief description and how many agents recommended it>\n"
            "2. <...>\n"
            "3. <...>\n"
            "4. <...>\n"
            "5. <...>\n"
            "If there are ties, break them by clarity and feasibility of the recommendation."
        ),
        llm_config=MODEL_CONFIG
    )

    # Collect all coordination recommendations for this round
    round_recs = [
        f"Agent {d['agent']}:\n{d['content']}"
        for d in discussion
        if isinstance(d, dict) and d.get("round") == round_id and d.get("type") == "coordination recommendations"
    ]
    recs_text = "\n\n".join(round_recs) if round_recs else "(no recommendations found)"

    # Build CDU prompt (includes prior IO strategy and asks for refinement)
    prior_strategy_text = IO_STRATEGY if IO_STRATEGY else "(none provided)"
    
    cdu_prompt = (
        f"Here are the coordination recommendations and context for {round_id}:\n\n"
        "=== PRIOR IO STRATEGY ===\n"
        f"{prior_strategy_text}\n\n"
        "=== AGENT RECOMMENDATIONS (next N_ROUNDS) ===\n"
        f"{recs_text}\n\n"
        f"N_ROUNDS = {N_DISCUSSION_STEPS}\n\n"
        "=== AGGREGATED STATS (numeric) ===\n"
        f"Total IO Posts: {aggregated_stats_export.get('Total IO Posts', 0)}; "
        f"Total Organic RTs: {aggregated_stats_export.get('Total Retweets from Organic', 0)}; "
        f"Total Organic Replies: {aggregated_stats_export.get('Total Replies from Organic', 0)}; "
        f"Total Unique Organic Users Engaged: {aggregated_stats_export.get('Total Unique Organic Users Engaged', 0)}; "
        f"Total Organic→IO Follows: {aggregated_stats_export.get('Total Organic→IO Follows', 0)}\n\n"
        "=== IO↔IO ACTIONS IN THE LAST WINDOW ===\n"
        f"{'(none)' if not io_to_io_actions else chr(10).join(f'- {line}' for line in io_to_io_actions)}\n\n"
        "=== REQUEST ===\n"
        "1) Update and refine the IO strategy for the next N_ROUNDS using:\n"
        "   - The PRIOR IO STRATEGY (retain what works, drop or fix what doesn’t),\n"
        "   - The latest AGENT RECOMMENDATIONS, and\n"
        "   - Current cues from AGGREGATED STATS and recent IO↔IO actions.\n"
        "2) Be concrete and operational. Provide:\n"
        "   - Updated Strategy (bullet points with rationale tied to metrics/cues),\n"
        "   - Priority Actions (ranked, with who/when/how),\n"
        "   - Risk & Mitigation (likely failure modes + contingency),\n"
        "   - Success Metrics (explicit targets we can compute next window).\n"
        "3) Keep it concise and directly executable by agents.\n\n"
        "Please produce the requested output strictly following the specified instructions."
    )
    
    user_proxy.initiate_chat(central_decision_unit, message=cdu_prompt)
    cdu_response = user_proxy.last_message()["content"].strip()
    
    discussion.append({
        "round": round_id,
        "type": "final recommendations",
        "agent": "CentralDecisionUnit",
        "content": cdu_response
    })
    
    return cdu_response

    

# ----------------------------------------------------------------------
# ────────────────────────── SIMULATION CORE ───────────────────────────
# ----------------------------------------------------------------------
def generate_simulation(n_steps,
                        initial_memory,
                        start_iter=1,
                        follow_log_initial=None,
                        discussion_initial=None,
                        tweet_id_counter_start=0):
    """
    Run `n_steps` more iterations, beginning at `start_iter`.
    Mirrors logic in main.py (5-step window; followed vs non-followed sampling; random gating).
    """
    global TWEET_ID_COUNTER
    global CURRENT_ITER
    global IO_STRATEGY
    global N_DISCUSSION_STEPS
    global WINDOW
    global MODEL_CONFIG
    
    TWEET_ID_COUNTER = tweet_id_counter_start
    CURRENT_ITER = 0
    memory     = list(initial_memory)            # shallow copy
    follow_log = list(follow_log_initial or [])

    for i in range(start_iter, start_iter + n_steps):
        print("ITERATION", i)
        CURRENT_ITER = i

        AGENT_LIST = USER_LIST + IO_LIST

        for agent in AGENT_LIST:
            user_proxy = autogen.UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False,
                default_auto_reply="default_auto_reply"
            )

            # 1) show personal history (last 5 steps)
            personal = [
                m for m in memory
                if m["Agent Name"] == agent.name and m["Content"].strip() and i-5 <= m["Iteration"] < i
            ]
            personal_lines = [
                f"[{m['TweetID']}] {m['Agent Name']}: {m['Content']} "
                f"(Retweets:{m.get('Retweets',0)} Likes👍:{m.get('Likes',0)} Comments💬:{m.get('Replies',0)})"
                for m in personal
            ]

            # build engagement lines (dedup within the 5-step window)
            engagement_lines = []
            seen_engagements = set()
            for m in personal:
                tid = m["TweetID"]
                # Retweets
                retweeters = [e for e in memory if e.get("Retweeted From") == tid and i-5 <= e["Iteration"] < i]
                for rt in retweeters:
                    line = f"[{tid}] was retweeted by {rt['Agent Name']} at step {rt['Iteration']}."
                    if line not in seen_engagements:
                        engagement_lines.append(line)
                        seen_engagements.add(line)
                # Comments
                commenters = [e for e in memory if e.get("Commented On") == tid and i-5 <= e["Iteration"] < i]
                for cm in commenters:
                    line = f"[{tid}] was commented on by {cm['Agent Name']}: \"{cm['Content']}\""
                    if line not in seen_engagements:
                        engagement_lines.append(line)
                        seen_engagements.add(line)
                # Follows
                recent_follows = [f for f in follow_log if f["Followed User"] == agent.name and i-5 <= f["Iteration"] < i]
                for fol in recent_follows:
                    line = f"[follow] You were followed by {fol['Agent Name']} at step {fol['Iteration']}."
                    if line not in seen_engagements:
                        engagement_lines.append(line)
                        seen_engagements.add(line)

            # 2) sample from other tweets (last 5 steps), split by followed/non-followed
            followed_users = {
                f["Followed User"]
                for f in follow_log
                if f["Agent Name"] == agent.name
            }

            recent_tweets = [
                m for m in memory
                if m["Agent Name"] != agent.name
                and m["Content"].strip()
                and i - 5 <= m["Iteration"] < i
            ]

            followed_tweets     = [m for m in recent_tweets if m["Agent Name"] in followed_users]
            non_followed_tweets = [m for m in recent_tweets if m["Agent Name"] not in followed_users]

            sample_followed     = random.sample(followed_tweets,     min(50, len(followed_tweets)))
            sample_non_followed = random.sample(non_followed_tweets, min(50, len(non_followed_tweets)))

            followed_lines = [
                f"[{m['TweetID']}] {m['Agent Name']}: {m['Content']} "
                f"(Retweets:{m.get('Retweets', 0)} Likes👍:{m.get('Likes', 0)} Comments💬:{m.get('Replies', 0)})"
                for m in sample_followed
            ]
            non_followed_lines = [
                f"[{m['TweetID']}] {m['Agent Name']}: {m['Content']} "
                f"(Retweets:{m.get('Retweets', 0)} Likes👍:{m.get('Likes', 0)} Comments💬:{m.get('Replies', 0)})"
                for m in sample_non_followed
            ]

            if agent in IO_LIST and IO_STRATEGY != "":
                prompt_strategy = (
                    "Here is the current coordination strategy for IO agents:\n" +
                    IO_STRATEGY + "\n\n" +
                    "Your later behaviors should adhere to this strategy.\n"
                    "Please acknowledge you have read and understood this strategy.\n"
                    "Reply 'yes' if you have completed reading and will follow this strategy."
                )
                user_proxy.initiate_chat(agent, message=prompt_strategy)

                prompt_recommendation = (
                    "Here are your recent tweets and their engagement metrics:\n" +
                    "\n".join(personal_lines) + "\n\n" +
                    "Engagement on your tweets in the last 5 steps:\n" +
                    "\n".join(engagement_lines or ["(no new engagements)\n"]) + "\n\n" +
                    "Here are the users that you already follow:\n" +
                    f"{getattr(agent, 'follow', set())}\n\n" +
                    "Here are some tweets from users you follow and their engagement metrics:\n" +
                    "\n".join(followed_lines) + "\n\n" +
                    "Here are some tweets from users you do not follow and their engagement metrics:\n" +
                    "\n".join(non_followed_lines) + "\n\n" +
                    "Reply 'yes' if you have completed reading these tweets."
                )
                user_proxy.send(recipient=agent, message=prompt_recommendation)

            else:
                prompt_recommendation = (
                    "Here are your recent tweets and their engagement metrics:\n" +
                    "\n".join(personal_lines) + "\n\n" +
                    "Engagement on your tweets in the last 5 steps:\n" +
                    "\n".join(engagement_lines or ["(no new engagements)\n"]) + "\n\n" +
                    "Here are the users that you already follow:\n" +
                    f"{getattr(agent, 'follow', set())}\n\n" +
                    "Here are some tweets from users you follow and their engagement metrics:\n" +
                    "\n".join(followed_lines) + "\n\n" +
                    "Here are some tweets from users you do not follow and their engagement metrics:\n" +
                    "\n".join(non_followed_lines) + "\n\n" +
                    "Reply 'yes' if you have completed reading these tweets."
                )
                user_proxy.initiate_chat(agent, message=prompt_recommendation)

            # 3) ask if the agent wants to follow (50% chance)
            if random.random() > 0.5:
                prompt_follow = (
                    "Here are the users that you already follow:\n" +
                    f"{getattr(agent, 'follow', set())}\n\n" +
                    "Based on the previous tweets, if you want to follow one of the users in those tweets that you haven't followed, reply with the username, which is right after the tweet ID in brackets and before the colon (e.g. for '[42] alanfike:', the username is 'alanfike'),\n"
                    "then a line break, then a brief reason why you want to follow the user. If you don't want to retweet, reply with 'None'."
                )

                user_proxy.send(recipient=agent, message=prompt_follow)
                resp = user_proxy.last_message()["content"].strip()

                if resp.lower() != "none":
                    try:
                        username, reason = resp.split("\n", 1)
                        username = username.strip()

                        # ensure agent.follow exists
                        if not hasattr(agent, "follow"):
                            agent.follow = set()

                        # Prevent self-follow
                        if username == agent.name:
                            print(f"⚠️  {agent.name} attempted to follow themselves—ignoring.")
                        # Prevent following someone already followed
                        elif username in agent.follow:
                            print(f"⚠️  {agent.name} already follows {username}—ignoring.")
                        else:
                            agent.follow.add(username)
                            follow_log.append({
                                "Iteration":      i,
                                "Agent Name":     agent.name,
                                "Followed User":  username,
                                "Reason":         reason.strip(),
                            })
                    except ValueError:
                        print(f"⚠️  Invalid follow response format from {agent.name}: {resp!r}")

            # 4) ask if the agent wants to retweet (50% chance)
            if random.random() > 0.5:
                prompt_retweet = (
                    "If you want to retweet one of the tweets that you have seen, reply with the TweetID in brackets (e.g. [42]),\n"
                    "then a line break, then a brief reason. If you don't want to retweet, reply with 'None'."
                )

                user_proxy.send(recipient=agent, message=prompt_retweet)
                resp = user_proxy.last_message()["content"].strip()

                if resp.lower() != "none":
                    match = re.match(r"\s*\[(\d+)\]\s*(.*)", resp, re.DOTALL)
                    if match:
                        orig_id = int(match.group(1))
                        reason = match.group(2).strip()

                        orig = next((m for m in memory if m["TweetID"] == orig_id), None)
                        if orig:
                            # increment original retweet count
                            orig["Retweets"] = orig.get("Retweets", 0) + 1

                            # create new retweet entry
                            TWEET_ID_COUNTER += 1
                            retweet_text = f"retweeted from {orig['Agent Name']}: {orig['Content']}"
                            memory.append({
                                "TweetID":        TWEET_ID_COUNTER,
                                "Agent Name":     agent.name,
                                "Iteration":      i,
                                "Content":        retweet_text,
                                "Type":           getattr(agent, "type", None),
                                "Retweets":       0,
                                "Likes":          0,
                                "Replies":        0,
                                "Retweeted From": orig_id,
                                "Retweet Reason": reason,
                            })
                        else:
                            print(f"⚠️  Original TweetID {orig_id} not found for {agent.name}")
                    else:
                        print(f"⚠️  Could not parse retweet response from {agent.name}: {resp!r}")

            # 5) ask if the agent wants to like (50% chance)
            if random.random() > 0.5:
                prompt_like = (
                    "If you want to like one of these tweets, reply with the TweetID in brackets (e.g. [42]). "
                    "Then a line break, then a brief reason. "
                    "If you don't want to like any, reply with 'None'."
                )

                user_proxy.send(recipient=agent, message=prompt_like)
                resp_like = user_proxy.last_message()["content"].strip()

                if resp_like.lower() != "none":
                    match_like = re.match(r"\s*\[(\d+)\]", resp_like)
                    if match_like:
                        like_id = int(match_like.group(1))
                        orig_like = next((m for m in memory if m["TweetID"] == like_id), None)
                        if orig_like:
                            orig_like["Likes"] = orig_like.get("Likes", 0) + 1
                        else:
                            print(f"⚠️  Like target TweetID {like_id} not found for {agent.name}")
                    else:
                        print(f"⚠️  Could not parse like response from {agent.name}: {resp_like!r}")

            # 6) ask if the agent wants to comment (50% chance)
            if random.random() > 0.5:
                prompt_comment = (
                    "If you want to comment on one of these tweets, reply with the TweetID in brackets (e.g. [42]),\n"
                    "then a line break, then your comment. If you don't want to comment, reply with 'None'."
                )

                user_proxy.send(recipient=agent, message=prompt_comment)
                resp_comment = user_proxy.last_message()["content"].strip()

                if resp_comment.lower() != "none":
                    match_comment = re.match(r"\s*\[(\d+)\]\s*(.*)", resp_comment, re.DOTALL)
                    if match_comment:
                        orig_id = int(match_comment.group(1))
                        comment = match_comment.group(2).strip()

                        orig = next((m for m in memory if m["TweetID"] == orig_id), None)
                        if orig:
                            orig["Replies"] = orig.get("Replies", 0) + 1

                            TWEET_ID_COUNTER += 1
                            comment_text = (
                                f"comment to {orig['Agent Name']}: {comment} "
                                f"original post: {orig['Content']}"
                            )
                            memory.append({
                                "TweetID":      TWEET_ID_COUNTER,
                                "Agent Name":   agent.name,
                                "Iteration":    i,
                                "Content":      comment_text,
                                "Type":         getattr(agent, "type", None),
                                "Retweets":     0,
                                "Likes":        0,
                                "Replies":      0,
                                "Commented On":  orig_id,
                            })
                        else:
                            print(f"⚠️  Comment target TweetID {orig_id} not found for {agent.name}")
                    else:
                        print(f"⚠️  Could not parse comment response from {agent.name}: {resp_comment!r}")

            # 7) ask the agent to generate 1 new message (50% chance)
            new_entries = []
            if random.random() > 0.5:
                prompt_organic = (
                    "Please generate 1 new message about the topic, based on the current discussion. "
                    "It should be within 100 words. "
                    "Give me only the message contents. "
                    "If you don't want to generate any, reply 'None'."
                )

                prompt_IO = (
                    "Here are your recent tweets and their engagement metrics:\n" +
                    "\n".join(personal_lines) + "\n\n" +
                    "Engagement on your tweets in the previous time steps:\n" +
                    "\n".join(engagement_lines or ["(no new engagements)"]) + "\n\n" +
                    "Please generate 1 new message about the topics, based on the current discussion. "
                    "It should be within 100 words. "
                    "Give me only the message contents. "
                    "If you don't want to generate any, reply 'None'."
                )

                if getattr(agent, "type", None) == "organic":
                    user_proxy.send(recipient=agent, message=prompt_organic)
                else:
                    user_proxy.send(recipient=agent, message=prompt_IO)

                resp_new = user_proxy.last_message()["content"].strip()
                if resp_new.lower() != "none":
                    new_lines = [line.strip() for line in resp_new.splitlines() if line.strip()]
                    for line in new_lines:
                        TWEET_ID_COUNTER += 1
                        entry = {
                            "TweetID":      TWEET_ID_COUNTER,
                            "Agent Name":   agent.name,
                            "Iteration":    i,
                            "Content":      line,
                            "Type":         getattr(agent, "type", None),
                            "Retweets":     0,
                            "Likes":        0,
                            "Replies":      0
                        }
                        new_entries.append(entry)

            # 8) ask for the strategies behind those new messages (mirrors main.py behavior)
            if new_entries:
                prompt_strat = (
                    "For each of the messages you just generated (in the same order), "
                    "please explain the strategy you used. Separate each explanation with a line break. "
                    "If you have no strategy to share, reply 'None'."
                )
                user_proxy.send(recipient=agent, message=prompt_strat)
                resp_strat = user_proxy.last_message()["content"].strip()

                if resp_strat.lower() != "none":
                    strat_lines = [line.strip() for line in resp_strat.splitlines() if line.strip()]
                    for entry, explanation in zip(new_entries, strat_lines):
                        entry["Strategy"] = explanation
                        memory.append(entry)
                # Note: if 'None', entries are intentionally not appended (matches main.py)

        # persist after each iteration
        pd.DataFrame(memory).to_csv(MEMORY_OUTPUT_PATH, index=False)
        pd.DataFrame(follow_log).to_csv(FOLLOW_OUTPUT_PATH, index=False)

        if CURRENT_ITER % N_DISCUSSION_STEPS == 0:
            IO_STRATEGY = generate_discussion(memory, follow_log, discussion, CURRENT_ITER, WINDOW)
            pd.DataFrame(discussion).to_csv(DISCUSSION_OUTPUT_PATH, index=False)

    return memory


# ----------------------------------------------------------------------
# ──────────────────────────────── MAIN ────────────────────────────────
# ----------------------------------------------------------------------
if __name__ == "__main__":
    TOTAL_STEPS = 50  # total iterations desired (including iteration 0 as seed)
    global N_DISCUSSION_STEPS
    global WINDOW
    WINDOW = 5
    global MODEL_CONFIG
    global IO_STRATEGY
    IO_STRATEGY = ""

    memory, follow_log, last_iter, TWEET_ID_COUNTER = load_previous_state()
    discussion = load_previous_discussion()

    if last_iter == 0 and not memory:
        # Fresh run: seed iteration 0
        memory = generate_initial_tweets()
        follow_log = []
        discussion = []
        last_iter = 0  # we just produced iteration 0
        # set counter to current max to avoid ID collisions
        TWEET_ID_COUNTER = max((m["TweetID"] for m in memory), default=0)
        IO_STRATEGY = ""
        # Save once after seeding (already saved in generate_initial_tweets, but keep consistent)
        pd.DataFrame(memory).to_csv(MEMORY_OUTPUT_PATH, index=False)

    remaining = max(0, TOTAL_STEPS - last_iter)
    if remaining == 0:
        print("✅  Simulation already completed up to TOTAL_STEPS =", TOTAL_STEPS)
    else:
        print(f"🔄  Resuming from iteration {last_iter + 1} "
              f"(will run {remaining} more steps to reach {TOTAL_STEPS})")

        # Restore IO_STRATEGY from the most recent CDU "final recommendations"
        latest_strategy = None
        if discussion:
            for row in reversed(discussion):
                if str(row.get("type", "")).strip().lower() == "final recommendations":
                    latest_strategy = row.get("content", None)
                    break    
        if latest_strategy:
            IO_STRATEGY = latest_strategy
            print("🧭  IO_STRATEGY restored from last CDU recommendations.")
        else:
            print("⚠️  No prior CDU 'final recommendations' found; keeping current IO_STRATEGY as-is.")

        generate_simulation(
            n_steps                = remaining,
            initial_memory         = memory,
            start_iter             = last_iter + 1,
            follow_log_initial     = follow_log,
            discussion_initial     = discussion,
            tweet_id_counter_start = TWEET_ID_COUNTER
        )
