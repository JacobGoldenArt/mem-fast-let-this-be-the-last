from datetime import datetime, timezone

from langchain.chat_models import init_chat_model
from langchain_core.runnables.config import RunnableConfig

from src.graph.state import AppState
from src.prompts.mem_prompt import mem_prompt
from src.settings.app_config import AppConfig, defaults
from src.settings.config_manager import ConfigManager
from src.tools.toolkit import toolkit

config_manager = ConfigManager(AppConfig, defaults)


async def main_agent(state: AppState, config: RunnableConfig) -> AppState:
    """Process the current state and generate a response using the LLM.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        schemas.State: The updated state with the agent's response.
    """
    configurable = config_manager.process_config(config)
    llm = init_chat_model(configurable["model"])  # type: ignore
    bound = mem_prompt | llm.bind_tools(toolkit)
    core_str = (
        "<core_memory>\n" + "\n".join(state["core_memories"]) + "\n</core_memory>"
    )
    conversational_str = (
        "<conversational_memory>\n"
        + "\n".join(state["conversational_memories"])
        + "\n</conversational_memory>"
    )
    prediction = await bound.ainvoke(
        {
            "messages": state["messages"],
            "core_memories": core_str,
            "conversational_memories": conversational_str,
            "current_time": datetime.now(tz=timezone.utc).isoformat(),
        }
    )
    return {
        "messages": prediction,  # type: ignore
    }