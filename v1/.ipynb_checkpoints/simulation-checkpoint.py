import requests

url = "http://127.0.0.1:11434/api/generate"

payload = {
    "model": "qwen-35b-a3b-mtp-q4_K_M",
    "prompt": "Why is the sky blue?",
    "stream": False,
}

r = requests.post(url, json=payload)
print(r.json()["response"])