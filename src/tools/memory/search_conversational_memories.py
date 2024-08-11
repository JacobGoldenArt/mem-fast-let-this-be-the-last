import langsmith
from langchain_core.tools import tool

from src.dbs.pinecone_db import get_index, namespace
from src.llms.embeddings_models import get_embeddings
from src.settings import constants


@tool
def search_conversational_memories(query: str, top_k: int = 5) -> list[str]:
    """Search for memories in the database based on semantic similarity.

        This function retrieves only 'recall' memories, not core memories.

        Args:
            query (str): The search query.
            top_k (int): The number of results to return.

        Returns:
            list[str]: A list of relevant memories.
        """
    embeddings = get_embeddings()

    # Create a vector embedding of the query
    vector = embeddings.embed_query(query)

    # Use LangSmith to trace the query to Pinecone for observability
    with langsmith.trace("query", inputs={"query": query, "top_k": top_k}) as rt:
        # Query the Pinecone index
        response = get_index().query(
            vector=vector,  # Filter to only include 'recall' memories for the current user
            filter={
                "user_id": {"$eq": constants.USER_ID},
                constants.TYPE_KEY: {"$eq": "conversational"},
            }, namespace=namespace, include_metadata=True, top_k=top_k, )
        # End the trace and log the response
        rt.end(outputs={"response": response})

    # Extract the actual memory contents from the query results
    """
    The line if matches := response.get("matches"): is using the walrus operator (:=) 
    to both assign the value of response.get("matches") to matches and check if it's truthy in one line.
    """
    memories = []
    if matches := response.get("matches"):
        memories = [m["metadata"][constants.PAYLOAD_KEY] for m in matches]

    return memories