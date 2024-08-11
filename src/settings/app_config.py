from typing_extensions import TypedDict


class AppConfig(TypedDict):
    model: str | None
    """The model to use for the memory assistant."""
    temperature: float
    """The temperature to use for the memory assistant."""
    thread_id: str
    """The thread ID of the conversation."""
    user_id: str
    """The ID of the user to remember in the conversation."""


AppDefaults = {
    "model": "gpt-4o-mini",
    "temperature": 0.2,
}

__all__ = ["AppDefaults", "AppConfig"]