import asyncio
import hashlib
import json
import random
from collections import Counter
from urllib.request import Request, urlopen

from ollama_urls import ollama_native_url


LLM_BIAS_ACTIONS = ["do_nothing", "post", "reply", "retweet"]
DEFAULT_PRIOR_SAMPLES = 10
DEFAULT_PRIOR_FEED_SIZE = 10


LLM_PRIOR_PROMPT = """
# OBJECTIVE
You're a Twitter user, and I'll present you with some tweets. After you see the tweets, you will response as described on RESPONSE METHOD section.

# SELF-DESCRIPTION
Your actions should be consistent with your self-description and personality.
{persona}

Feed snapshot:
{feed}

# RESPONSE METHOD
Choose exactly one action:
- do_nothing: take no action on this feed.
- post: create a new original post that is not a reply.
- reply: reply to one tweet in the feed.
- retweet: reshare one tweet in the feed.

Return only a JSON object whose action field contains one allowed action. Do not explain the choice.

Action:
""".strip()


LLM_PRIOR_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": LLM_BIAS_ACTIONS}},
    "required": ["action"],
    "additionalProperties": False,
}


def build_llm_prior_prompt(persona, feed_tweets):
    return LLM_PRIOR_PROMPT.format(
        persona=persona,
        feed="\n".join(
            f"[{index}] {tweet}" for index, tweet in enumerate(feed_tweets, start=1)
        ),
    )


def parse_constrained_action(response):
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Ollama returned invalid constrained JSON: {response!r}"
        ) from error

    action = parsed.get("action") if isinstance(parsed, dict) else None
    if action not in LLM_BIAS_ACTIONS:
        raise ValueError(
            f"Ollama returned invalid action {action!r}; expected one of "
            f"{LLM_BIAS_ACTIONS}"
        )
    return action


def call_constrained_action(
    prompt,
    model="qwen3.6:35b-a3b-mtp-q4_K_M",
    ollama_url="http://127.0.0.1:11434",
    request_timeout=1800,
    seed=None,
):
    options = {
        "temperature": 0.0,
        "num_predict": 32,
    }
    if seed is not None:
        options["seed"] = seed

    request = Request(
        ollama_native_url(ollama_url) + "/api/generate",
        data=json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": LLM_PRIOR_RESPONSE_SCHEMA,
                "think": False,
                "options": options,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=request_timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    return parse_constrained_action(result["response"].strip())


def feed_posts_to_text(feed_posts):
    """Convert OASIS feed posts into non-empty text strings for the prompt."""
    tweets = []

    for post in feed_posts:
        if isinstance(post, str):
            content = post
        elif isinstance(post, dict):
            content = post.get("content", "")
            username = (
                post.get("username")
                or post.get("user_name")
                or post.get("author_name")
            )
            if username and content:
                content = f"@{username}: {content}"
        else:
            content = ""

        content = str(content).strip()
        if content:
            tweets.append(content)

    if not tweets:
        raise ValueError("Cannot collect an LLM choice from an empty feed")

    return tweets


async def collect_llm_choice(
    persona,
    feed_posts,
    *,
    model="qwen3.6:35b-a3b-mtp-q4_K_M",
    ollama_url="http://127.0.0.1:11434",
    request_timeout=1800,
    seed=None,
):
    """Collect a constrained LLM choice without executing an OASIS action."""
    feed_tweets = feed_posts_to_text(feed_posts)
    prompt = build_llm_prior_prompt(persona, feed_tweets)

    return await asyncio.to_thread(
        call_constrained_action,
        prompt,
        model=model,
        ollama_url=ollama_url,
        request_timeout=request_timeout,
        seed=seed,
    )


def stable_user_seed(prior_seed, user_id, is_io):
    seed_material = f"{prior_seed}:{int(is_io)}:{user_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")


def sample_feed_snapshot(feed_pool, feed_size, rng):
    if not feed_pool:
        raise ValueError("Cannot estimate an LLM prior from an empty feed pool")
    if len(feed_pool) >= feed_size:
        return rng.sample(feed_pool, feed_size)
    return rng.choices(feed_pool, k=feed_size)


def estimate_llm_bias_probabilities(
    persona,
    feed_pool,
    prior_samples=DEFAULT_PRIOR_SAMPLES,
    feed_size=DEFAULT_PRIOR_FEED_SIZE,
    prior_seed=0,
    model="qwen3.6:35b-a3b-mtp-q4_K_M",
    ollama_url="http://127.0.0.1:11434",
    request_timeout=1800,
):
    """Estimate one user's constrained LLM action prior with Laplace smoothing."""
    if prior_samples <= 0:
        raise ValueError("prior_samples must be greater than 0")
    if feed_size <= 0:
        raise ValueError("feed_size must be greater than 0")

    clean_feed_pool = [
        str(tweet).strip() for tweet in feed_pool if str(tweet).strip()
    ]
    rng = random.Random(prior_seed)
    choices = []
    for _ in range(prior_samples):
        snapshot = sample_feed_snapshot(clean_feed_pool, feed_size, rng)
        choices.append(
            call_constrained_action(
                build_llm_prior_prompt(persona, snapshot),
                model=model,
                ollama_url=ollama_url,
                request_timeout=request_timeout,
                seed=rng.randrange(0, 2**31),
            )
        )

    counts = Counter(choices)
    denominator = prior_samples + len(LLM_BIAS_ACTIONS)
    return {
        f"llm_bias_{action}_prob": (counts[action] + 1) / denominator
        for action in LLM_BIAS_ACTIONS
    }
