import os
import re
from diskcache import Cache
import autogen
import random
import numpy as np
import pandas as pd
from agent_v2_hashtag import args, USER_LIST, IO_LIST, MEMORY_OUTPUT_PATH, FOLLOW_OUTPUT_PATH

# process_id = os.getpid()
# cache = Cache(f'/home1/jinyiy/GABM_IO/cache/cache_{process_id}')

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

# ----------------------------------------------------------------------
# ────────────────────────── SIMULATION CORE ───────────────────────────
# ----------------------------------------------------------------------
def generate_simulation(n_steps,
                        initial_memory,
                        start_iter=1,
                        follow_log_initial=None,
                        tweet_id_counter_start=0):
    """
    Run `n_steps` more iterations, beginning at `start_iter`.
    Mirrors logic in main.py (5-step window; followed vs non-followed sampling; random gating).
    """
    global TWEET_ID_COUNTER
    TWEET_ID_COUNTER = tweet_id_counter_start
    memory     = list(initial_memory)            # shallow copy
    follow_log = list(follow_log_initial or [])

    for i in range(start_iter, start_iter + n_steps):
        print("ITERATION", i)

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

    return memory

# ----------------------------------------------------------------------
# ──────────────────────────────── MAIN ────────────────────────────────
# ----------------------------------------------------------------------
if __name__ == "__main__":
    TOTAL_STEPS = 50  # total iterations desired (including iteration 0 as seed)

    memory, follow_log, last_iter, TWEET_ID_COUNTER = load_previous_state()

    if last_iter == 0 and not memory:
        # Fresh run: seed iteration 0
        memory = generate_initial_tweets()
        follow_log = []
        last_iter = 0  # we just produced iteration 0
        # set counter to current max to avoid ID collisions
        TWEET_ID_COUNTER = max((m["TweetID"] for m in memory), default=0)
        # Save once after seeding (already saved in generate_initial_tweets, but keep consistent)
        pd.DataFrame(memory).to_csv(MEMORY_OUTPUT_PATH, index=False)

    remaining = max(0, TOTAL_STEPS - last_iter)
    if remaining == 0:
        print("✅  Simulation already completed up to TOTAL_STEPS =", TOTAL_STEPS)
    else:
        print(f"🔄  Resuming from iteration {last_iter + 1} "
              f"(will run {remaining} more steps to reach {TOTAL_STEPS})")
        generate_simulation(
            n_steps                = remaining,
            initial_memory         = memory,
            start_iter             = last_iter + 1,
            follow_log_initial     = follow_log,
            tweet_id_counter_start = TWEET_ID_COUNTER
        )
