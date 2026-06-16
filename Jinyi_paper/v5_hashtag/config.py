import os

llama_config = {
    "config_list": [
        {
            "model": "llama3.3:latest",
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama"
        },
    ],
    "cache_seed": None,  # Disable caching.
    "temperature": 1,
}


groq_llama_config = {
    "config_list": [
        {
            "model": "llama-3.3-70b-versatile",
            "api_key": os.environ.get("GROQ_API_KEY"),
            "api_type": "groq"
        }],
    "cache_seed": None,
    "temperature": 1,
}


