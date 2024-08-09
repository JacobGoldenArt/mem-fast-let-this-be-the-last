import json
from typing import Tuple

import langsmith

from src.dbs.pinecone_db import get_index, namespace
from src.settings import constants


@langsmith.traceable
def fetch_core_memories(user_id: str) -> Tuple[str, list[str]]:
    """Fetch core memories for a specific user.

    Args:
        user_id (str): The ID of the user.

    Returns:
        Tuple[str, list[str]]: The path and list of core memories.
    """
    path = constants.PATCH_PATH.format(user_id=user_id)
    response = get_index().fetch(ids=[path], namespace=namespace)
    memories = []
    if vectors := response.get("vectors"):
        document = vectors[path]
        payload = document["metadata"][constants.PAYLOAD_KEY]
        memories = json.loads(payload)["memories"]
    return path, memories