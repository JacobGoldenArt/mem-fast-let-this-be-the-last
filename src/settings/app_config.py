from __future__ import annotations

import os

from dotenv import load_dotenv
from typing_extensions import TypedDict

# Load environment variables from .env file
load_dotenv()

jg_user_id = os.getenv("JG_USER_ID")


class AppConfig(TypedDict):
    model: str | None
    """The model to use for the memory assistant."""
    temperature: float
    """The temperature to use for the memory assistant."""
    thread_id: str
    """The thread ID of the conversation."""
    user_id: str
    """The ID of the user to remember in the conversation."""


defaults = {
    "user_id": jg_user_id,
    "model": "gpt-4o-mini",
    "temperature": 0.2,
}