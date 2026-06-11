from dataclasses import dataclass


@dataclass(frozen=True)
class Agent:
    name: str
    role: str

    def system_prompt(self) -> str:
        return (
            f"You are {self.name}. Your role is: {self.role}\n"
            "Write concise responses IN ENGLISH. Stay in character, but do not invent facts. "
            "If you are uncertain, say what information would be needed."
        )
