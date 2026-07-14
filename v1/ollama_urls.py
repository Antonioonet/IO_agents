def ollama_native_url(url):
    """Return the Ollama base URL used by native /api endpoints."""
    base_url = url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return base_url.rstrip("/")


def ollama_openai_url(url):
    """Return the Ollama base URL used by OpenAI-compatible clients."""
    return ollama_native_url(url) + "/v1"
