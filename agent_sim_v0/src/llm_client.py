from openai import OpenAI

from agent_sim_v0.src.config import ModelConfig


class OllamaClient:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.name,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
