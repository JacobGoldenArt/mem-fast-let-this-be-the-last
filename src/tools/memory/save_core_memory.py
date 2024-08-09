import json
from datetime import datetime, timezone
from typing import Optional

from langchain_core.runnables import ensure_config
from langchain_core.tools import tool

from src.dbs.pinecone_db import get_index, namespace
from src.settings import constants
from src.settings.app_config import AppConfig, defaults
from src.settings.config_manager import ConfigManager
from src.settings.constants import EMPTY_VEC
from src.tools.memory.fetch_core_memories import fetch_core_memories

memory_config_manager = ConfigManager(AppConfig, defaults)


@tool
def save_core_memory(memory: str, index: Optional[int] = None) -> str:
    """Store a core memory in the database.

    Args:
        memory (str): The memory to store.
        index (Optional[int]): The index at which to store the memory.

    Returns:
        str: A confirmation message.
    """
    config = ensure_config()
    configurable = memory_config_manager.process_config(config)
    path, memories = fetch_core_memories(configurable["user_id"])
    if index is not None:
        if index < 0 or index >= len(memories):
            return "Error: Index out of bounds."
        memories[index] = memory
    else:
        memories.insert(0, memory)
    documents = [
        {
            "id": path,
            "values": EMPTY_VEC,
            "metadata": {
                constants.PAYLOAD_KEY: json.dumps({"memories": memories}),
                constants.PATH_KEY: path,
                constants.TIMESTAMP_KEY: datetime.now(tz=timezone.utc),
                constants.TYPE_KEY: "recall",
                "user_id": configurable["user_id"],
            },
        }
    ]
    get_index().upsert(
        vectors=documents,
        namespace=namespace,
    )
    return "Memory stored."