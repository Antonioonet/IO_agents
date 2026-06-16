import re
import pandas as pd
from collections import defaultdict


def sample_personas(df, sample_sizes, random_state=42):
    return df.sample(n=sample_sizes, random_state=random_state).reset_index(drop=True)


def generate_io_activity_summary(memory, follow_log, current_iteration, discussion, window=5):
    """
    Returns:
        per_agent_summaries: list[str]
        per_agent_named:     list[(agent_name, summary_text)]
        aggregated_summary:  str
        per_agent_stats_export: dict[agent_name] -> dict numeric stats (for comparing across rounds)
        aggregated_stats_export: dict total numeric stats (for comparing across rounds)
        io_to_io_actions:    list[str] (IO->IO retweets/replies/follows observed in the window)
    """
    # ------------------ helpers ------------------
    IO_SYNONYMS  = {"io", "i.o.", "coordinated", "influence", "state", "coordinated_io"}
    ORG_SYNONYMS = {"organic", "user", "human"}

    def _normalize_types_and_iters(memory, follow_log):
        # Ensure Iteration is int if possible
        for df in (memory, follow_log):
            for r in df:
                if "Iteration" in r:
                    try:
                        r["Iteration"] = int(r["Iteration"])
                    except Exception:
                        r["Iteration"] = pd.NA
        # Normalize Type to {"IO","organic"} when recognizable
        for r in memory:
            t = r.get("Type")
            if t is None or pd.isna(t): 
                continue
            t_norm = str(t).strip().lower()
            if t_norm in IO_SYNONYMS:
                r["Type"] = "IO"
            elif t_norm in ORG_SYNONYMS:
                r["Type"] = "organic"
            # else leave as-is

    _normalize_types_and_iters(memory, follow_log)

    window_start = max(current_iteration - window, 0)
    window_end   = current_iteration
    round_id = f"iter{window_start}-{window_end-1}"

    # name -> type (last seen)
    agent_type_by_name = {}
    for m in memory:
        n = m.get("Agent Name")
        t = m.get("Type")
        if n and t:
            agent_type_by_name[n] = t

    # ------------------ collect IO posts (all-time) ------------------
    io_posts_all = [m for m in memory if m.get("Type") == "IO"]
    io_tweet_ids_all = {m["TweetID"] for m in io_posts_all}
    tweetid_to_agent   = {m["TweetID"]: m["Agent Name"] for m in io_posts_all}
    tweetid_to_content = {m["TweetID"]: m.get("Content", "") for m in io_posts_all}
    io_agent_names     = {m["Agent Name"] for m in io_posts_all}

    # IO posts within the window
    io_posts_window = [
        m for m in io_posts_all
        if m.get("Iteration") is not pd.NA and window_start <= m.get("Iteration", -10**9) < window_end
    ]

    # ------------------ per-agent stats ------------------
    per_agent_stats = defaultdict(lambda: {
        "Num Posts": 0,
        "Retweets from Organic": 0,
        "Replies from Organic": 0,
        "Unique Organic Users Engaged": set(),
        "Organic Follows (received)": 0,
        "Most Popular Tweet": None,
        "Most Popular Score": -1,
    })

    # Aggregates (for export)
    total_posts = 0
    total_retweets_from_organic = 0
    total_replies_from_organic  = 0
    all_engaged_organic_users   = set()

    # Engagement score per IO tweet
    tweet_score_map = {}
    for post in io_posts_all:
        r  = int(post.get("Retweets", 0) or 0)
        l  = int(post.get("Likes", 0) or 0)
        rp = int(post.get("Replies", 0) or 0)
        score = r + l + rp
        tid = post["TweetID"]
        tweet_score_map[tid] = (score, post["Agent Name"], r, l, rp, post.get("Content", ""))

    # Count IO posts in window & most popular
    for post in io_posts_window:
        a = post["Agent Name"]
        per_agent_stats[a]["Num Posts"] += 1
        total_posts += 1
        tid = post["TweetID"]
        score, _, r, l, rp, content = tweet_score_map.get(
            tid, (0, a, 0, 0, 0, post.get("Content", "")))
        if score > per_agent_stats[a]["Most Popular Score"]:
            per_agent_stats[a]["Most Popular Score"]  = score
            per_agent_stats[a]["Most Popular Tweet"] = content

    # ------------------ organic engagements (even on older IO tweets) ------------------
    relevant_tids_by_window_engagement = set()
    for m in memory:
        it = m.get("Iteration")
        if it is pd.NA or not (window_start <= it < window_end):
            continue
        # Treat as organic if explicitly "organic" OR unknown (conservative for counting)
        t = (m.get("Type") or "").strip()
        is_org_actor = (t == "organic") or (t == "")

        if not is_org_actor:
            continue

        tid_rt = m.get("Retweeted From")
        if tid_rt in io_tweet_ids_all:
            author = tweetid_to_agent[tid_rt]
            per_agent_stats[author]["Retweets from Organic"] += 1
            name = m.get("Agent Name")
            if name:
                per_agent_stats[author]["Unique Organic Users Engaged"].add(name)
                all_engaged_organic_users.add(name)
            total_retweets_from_organic += 1
            relevant_tids_by_window_engagement.add(tid_rt)

        tid_cm = m.get("Commented On")
        if tid_cm in io_tweet_ids_all:
            author = tweetid_to_agent[tid_cm]
            per_agent_stats[author]["Replies from Organic"] += 1
            name = m.get("Agent Name")
            if name:
                per_agent_stats[author]["Unique Organic Users Engaged"].add(name)
                all_engaged_organic_users.add(name)
            total_replies_from_organic += 1
            relevant_tids_by_window_engagement.add(tid_cm)

    # ------------------ Organic -> IO follows in the window (robust) ------------------
    total_organic_follows_to_io = 0
    for f in follow_log:
        # force int iteration
        try:
            fit = int(f.get("Iteration"))
        except (TypeError, ValueError):
            continue
        if not (window_start <= fit < window_end):
            continue

        follower     = f.get("Agent Name")
        followed_usr = f.get("Followed User")

        # Defaults make this robust even if types are missing:
        follower_type = agent_type_by_name.get(follower, "organic")   # assume organic if unknown
        followed_is_io = (agent_type_by_name.get(followed_usr) == "IO") or (followed_usr in io_agent_names)

        if follower_type == "organic" and followed_is_io:
            total_organic_follows_to_io += 1
            per_agent_stats[followed_usr]["Organic Follows (received)"] += 1

    # ------------------ IO -> IO interactions (in the window) ------------------
    io_to_io_actions = []

    # Retweets/Replies by IO actors to IO originals
    for m in memory:
        it = m.get("Iteration")
        if it is pd.NA or not (window_start <= it < window_end):
            continue
        if m.get("Type") != "IO":
            continue
        actor = m.get("Agent Name")

        tid_rt = m.get("Retweeted From")
        if tid_rt in io_tweet_ids_all:
            target_author = tweetid_to_agent[tid_rt]
            io_to_io_actions.append(f"{actor} retweeted {target_author} [TweetID {tid_rt}]")

        tid_cm = m.get("Commented On")
        if tid_cm in io_tweet_ids_all:
            target_author = tweetid_to_agent[tid_cm]
            io_to_io_actions.append(f"{actor} replied to {target_author} [TweetID {tid_cm}]")

    # Follows: IO following IO
    for f in follow_log:
        try:
            fit = int(f.get("Iteration"))
        except (TypeError, ValueError):
            continue
        if not (window_start <= fit < window_end):
            continue

        follower     = f.get("Agent Name")
        followed_usr = f.get("Followed User")
        if (agent_type_by_name.get(follower) == "IO") and (
            (agent_type_by_name.get(followed_usr) == "IO") or (followed_usr in io_agent_names)
        ):
            io_to_io_actions.append(f"{follower} followed {followed_usr}")

    # ------------------ craft summaries (strings) ------------------
    per_agent_summaries = []
    per_agent_named = []  # (agent_name, summary_text)
    for agent, s in per_agent_stats.items():
        txt = (
            f"📢 Summary for {agent} (last {window} steps):\n"
            f"- Tweets Posted: {s['Num Posts']}\n"
            f"- Retweets from Organic Users: {s['Retweets from Organic']}\n"
            f"- Replies from Organic Users: {s['Replies from Organic']}\n"
            f"- Unique Organic Users Engaged: {len(s['Unique Organic Users Engaged'])}\n"
            f"- Organic Follows Received: {s['Organic Follows (received)']}\n"
        )
        if s["Most Popular Tweet"]:
            txt += f"- Most Popular Tweet (by total engagement): \"{s['Most Popular Tweet']}\"\n"
        per_agent_summaries.append(txt)
        per_agent_named.append((agent, txt))

    # Top-5 IO tweets eligible if posted in window OR engaged by organic in window
    window_tids   = {m["TweetID"] for m in io_posts_window}
    eligible_tids = window_tids.union(relevant_tids_by_window_engagement)

    top_pool = []
    for tid in eligible_tids:
        score, author, r, l, rp, content = tweet_score_map.get(tid, (0, "?", 0, 0, 0, ""))
        top_pool.append((score, tid, author, r, l, rp, content))
    top_pool.sort(key=lambda x: x[0], reverse=True)
    top5 = top_pool[:5]

    if top5:
        top5_lines = [
            f"[{tid}] by {author} — Score={score} (Retweets:{r} Likes:{l} Replies:{rp})\n    \"{content}\""
            for score, tid, author, r, l, rp, content in top5
        ]
        top5_block = "\n".join(top5_lines)
    else:
        top5_block = "(no IO tweets posted or engaged-with in this window)"

    aggregated_summary = (
        f"📊 Aggregated Summary of IO Influence Agents (last {window} steps):\n"
        f"- Total IO Posts (posted in window): {total_posts}\n"
        f"- Total Retweets from Organic Users (in window): {total_retweets_from_organic}\n"
        f"- Total Replies from Organic Users (in window): {total_replies_from_organic}\n"
        f"- Total Unique Organic Users Engaged (in window): {len(all_engaged_organic_users)}\n"
        f"- Total Organic→IO Follows (in window): {total_organic_follows_to_io}\n"
        f"- IO↔IO actions observed (list): {io_to_io_actions if io_to_io_actions else '(none)'}\n"
        f"- 🏆 Top 5 IO Tweets by Total Engagement (among those posted/engaged in window):\n{top5_block}"
    )

    # ------------------ append one line per summary + STATS to `discussion` ------------------
    # Save stats in numeric form for round-over-round comparison
    per_agent_stats_export = {
        agent: {
            "Num Posts": s["Num Posts"],
            "Retweets from Organic": s["Retweets from Organic"],
            "Replies from Organic": s["Replies from Organic"],
            "Unique Organic Users Engaged": len(s["Unique Organic Users Engaged"]),
            "Organic Follows Received": s["Organic Follows (received)"],
            "Most Popular Score": s["Most Popular Score"],
        }
        for agent, s in per_agent_stats.items()
    }

    aggregated_stats_export = {
        "Total IO Posts": total_posts,
        "Total Retweets from Organic": total_retweets_from_organic,
        "Total Replies from Organic": total_replies_from_organic,
        "Total Unique Organic Users Engaged": len(all_engaged_organic_users),
        "Total Organic→IO Follows": total_organic_follows_to_io,
    }

    # Individual (one line per agent) + stats record
    for agent_name, summary_text in per_agent_named:
        one_line = summary_text.replace("\n", " ").strip()
        discussion.append({
            "round": round_id,
            "type": "individual summary",
            "agent": agent_name,
            "content": one_line
        })
        discussion.append({
            "round": round_id,
            "type": "individual stats",
            "agent": agent_name,
            "content": per_agent_stats_export[agent_name]
        })

    # Aggregated (single line) + stats + IO↔IO actions
    agg_one_line = aggregated_summary.replace("\n", " ").strip()
    discussion.append({
        "round": round_id,
        "type": "aggregated summary",
        "agent": "ALL",
        "content": agg_one_line
    })
    discussion.append({
        "round": round_id,
        "type": "aggregated stats",
        "agent": "ALL",
        "content": aggregated_stats_export
    })
    discussion.append({
        "round": round_id,
        "type": "io-io interactions",
        "agent": "ALL",
        "content": list(io_to_io_actions)  # copy
    })

    return (
        per_agent_summaries,
        per_agent_named,
        aggregated_summary,
        per_agent_stats_export,
        aggregated_stats_export,
        io_to_io_actions
    )