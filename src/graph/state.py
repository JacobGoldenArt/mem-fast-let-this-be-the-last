from __future__ import annotations
from typing import List
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated, TypedDict


# Define the schema for the state maintained throughout the conversation
class AppState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    """The messages in the conversation."""
    core_memories: List[str]
    """The core memories associated with the user."""
    conversational_memories: List[str]
    """The conversational memories retrieved for the current context."""
