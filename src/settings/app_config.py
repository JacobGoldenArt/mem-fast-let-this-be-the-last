from typing import TypedDict, Literal, Optional


class AppConfig(TypedDict):
    """LLM for EMs main conversation."""
    em_model: Literal[
        'oai-gpt4',
        'oai-gpt4m',
        'an-35-s',
        'mis-nemo',
        'mis-lg',
        'fw-lm31-405b',
        'fw-lm31-150b',
        'fw-lm31-8b',
    ]
    """The temperature to use for the memory assistant."""
    temperature: Optional[float]
    """The temperature to use for the task related conversation."""
    # thread_id: str
    """The thread ID of the conversation."""
    # user_id: str
    """The ID of the user to remember in the conversation."""


AppDefaults = {
    "em_model": "oai-gpt4m",
    "temperature": 0.3,
}

__all__ = ["AppConfig", "AppDefaults"]