import tiktoken
from langchain_core.messages.utils import get_buffer_string
from langchain_core.runnables.config import (
    RunnableConfig,
    get_executor_for_config,
)

from src.graph.state import AppState
from src.settings.app_config import AppConfig, defaults
from src.settings.config_manager import ConfigManager
from src.tools.memory.fetch_core_memories import fetch_core_memories
from src.tools.memory.search_conversational_memories import (
    search_conversational_memories,
)

memory_config_manager = ConfigManager(AppConfig, defaults)


def load_memories(state: AppState, config: RunnableConfig) -> AppState:
    """Load core and conversational memories for the current conversation.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        schemas.State: The updated state with loaded memories.
    """

    # Process the config to ensure all necessary values are present
    configurable = memory_config_manager.process_config(config)
    user_id = configurable["user_id"]

    # Initialize the tokenizer for GPT-4
    tokenizer = tiktoken.encoding_for_model("gpt-4o")

    # Convert the message history to a string
    # Example: "Human: Hi, how are you?\nAI: Good, how are you?"
    convo_str = get_buffer_string(state["messages"])

    # Encode the conversation, keep the last 2048 tokens, then decode back to string
    # This ensures we're working with the most recent and relevant part of the conversation
    encoded_convo = tokenizer.encode(convo_str)
    truncated_encoded_convo = encoded_convo[-2048:]
    decoded_convo_str = tokenizer.decode(truncated_encoded_convo)

    # Use a ThreadPoolExecutor to run memory retrieval operations concurrently
    with get_executor_for_config(config) as executor:
        # Submit two tasks to be executed in parallel:
        # 1. Fetch core memories for the user
        # 2. Search for relevant recall memories based on the conversation
        futures = [
            executor.submit(fetch_core_memories, user_id),
            executor.submit(search_conversational_memories.invoke, decoded_convo_str),
        ]

        # Wait for the tasks to complete and retrieve results
        # The '_' in the first line ignores an unused return value
        _, core_memories = futures[0].result()
        conversational_memories = futures[1].result()

    # Return the retrieved memories as part of the updated state
    return {
        "core_memories": core_memories,
        "conversational_memories": conversational_memories,
    }