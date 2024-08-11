from __future__ import annotations

import uuid
from datetime import datetime, timezone

from langchain_core.tools import tool

from src.dbs.pinecone_db import get_index, namespace
from src.llms.embeddings_models import get_embeddings
from src.settings import constants


@tool
async def save_conversational_memory(memory: str) -> str:
    """Save a memory to the database for later semantic retrieval.

        Args:
            memory (str): The memory to be saved.

        Returns:
            str: The saved memory.
    """

    # Get the embeddings model
    embeddings = get_embeddings()

    # Create a vector embedding of the memory
    vector = await embeddings.aembed_query(memory)

    # Get the current UTC timestamp
    current_time = datetime.now(tz=timezone.utc)

    # Generate a unique path for this memory
    # INSERT_PATH is likely a string template like "user/{user_id}/recall/{event_id}"
    path = constants.INSERT_PATH.format(
        user_id=constants.USER_ID,
        event_id=str(uuid.uuid4()),
    )

    # Prepare the document for insertion into the vector database
    # This structure is tailored for Pinecone but could be adapted for other vector DBs
    documents = [{
        "id": path,
        "values": vector,
        "metadata": {
            constants.PAYLOAD_KEY: memory,  # Actual memory content
            constants.PATH_KEY: path,  # Unique identifier for this memory
            constants.TIMESTAMP_KEY: current_time,
            constants.TYPE_KEY: "conversational",  # Distinguishes from "core" memories
            "user_id": constants.USER_ID,
        },
    }]

    # Insert the document into the Pinecone index
    # utils.get_index() returns a Pinecone index instance
    get_index().upsert(vectors=documents, namespace=namespace, )

    # Return the original memory string
    # This could potentially be replaced with a confirmation message
    return memory