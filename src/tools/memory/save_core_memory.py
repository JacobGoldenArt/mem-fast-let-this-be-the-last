import json
from datetime import datetime, timezone
from typing import Optional

from langchain_core.tools import tool

from src.dbs.pinecone_db import get_index, namespace
from src.settings import constants
from src.tools.memory.fetch_core_memories import fetch_core_memories


@tool
def store_update_core_memory(memory: str, index: Optional[int] = None) -> str:
    """
    Store or update a core memory in the database.

    Args:
        memory (str): The content of the core memory to be stored or updated.
        index (Optional[int]): The index at which to store or update the memory.
            If provided, it updates the existing memory at that index.
            If None, a new memory is inserted at the beginning of the list.

    Returns:
        str: A confirmation message indicating the success
             or an error message if the index is out of bounds.
    """

    path, memories = fetch_core_memories(constants.USER_ID)

    if index is not None:
        if index < 0 or index >= len(memories):
            return "Error: Index out of bounds."
        memories[index] = memory
    else:
        memories.insert(0, memory)  # Insert at the beginning if no index is provided

    documents = [{
        "id": path,
        "values": constants.MINIMAL_VEC,
        "metadata": {
            constants.PAYLOAD_KEY: json.dumps({"memories": memories}),
            constants.PATH_KEY: path,
            constants.TIMESTAMP_KEY: datetime.now(tz=timezone.utc),
            constants.TYPE_KEY: "core",
            "user_id": constants.USER_ID,
        },
    }]

    get_index().upsert(vectors=documents, namespace=namespace)
    return "Memory stored."