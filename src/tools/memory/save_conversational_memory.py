from __future__ import annotations

import uuid
from datetime import datetime, timezone

from langchain_core.runnables.config import (
    RunnableConfig,
)
from langchain_core.tools import tool

from src.dbs.pinecone_db import namespace, get_index
from src.llms.embeddings_models import get_embeddings
from src.settings import constants
from src.settings.app_config import AppConfig, defaults
from src.settings.config_manager import ConfigManager

memory_config_manager = ConfigManager(AppConfig, defaults)


@tool
async def save_conversational_memory(memory: str, config: RunnableConfig) -> str:
    """Save a memory to the database for later semantic retrieval.

    Args:
        memory (str): The memory to be saved.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        str: The saved memory.
    """
    # Ensure configuration is properly set up
    configurable = memory_config_manager.process_config(config)

    # Get the embeddings model
    embeddings = get_embeddings()

    # Create a vector embedding of the memory
    vector = await embeddings.aembed_query(memory)

    # Get the current UTC timestamp
    current_time = datetime.now(tz=timezone.utc)

    # Generate a unique path for this memory
    # INSERT_PATH is likely a string template like "user/{user_id}/recall/{event_id}"
    path = constants.INSERT_PATH.format(
        user_id=configurable["user_id"],
        event_id=str(uuid.uuid4()),
    )

    # Prepare the document for insertion into the vector database
    # This structure is tailored for Pinecone but could be adapted for other vector DBs
    documents = [
        {
            "id": path,
            "values": vector,
            "metadata": {
                constants.PAYLOAD_KEY: memory,  # Actual memory content
                constants.PATH_KEY: path,  # Unique identifier for this memory
                constants.TIMESTAMP_KEY: current_time,
                constants.TYPE_KEY: "recall",  # Distinguishes from "core" memories
                "user_id": configurable["user_id"],
            },
        }
    ]

    # Insert the document into the Pinecone index
    # utils.get_index() returns a Pinecone index instance
    get_index().upsert(
        vectors=documents,
        namespace=namespace,
    )

    # Return the original memory string
    # This could potentially be replaced with a confirmation message
    return memory