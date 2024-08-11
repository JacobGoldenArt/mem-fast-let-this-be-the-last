from datetime import datetime, timezone

from langchain.chat_models import init_chat_model
from langchain_core.runnables.config import RunnableConfig

from src.graph.state import AppState
from src.prompts.mem_prompt import mem_prompt
from src.settings.config import config_manager
from src.tools.toolkit import toolkit


async def main_agent(state: AppState, config: RunnableConfig) -> AppState:
    """
    Asynchronous function for the main agent's behavior.

    Parameters:
    - state (AppState): The current state of the conversation.
    - config (RunnableConfig): The configuration settings for the agent.

    Returns:
    AppState: The updated state after processing the agent's behavior.
"""
    app_config = config_manager.process_config(config)
    llm = init_chat_model(app_config["model"])

    bound = mem_prompt | llm.bind_tools(toolkit)
    core_str = (
            "<core_memory>\n" + "\n".join(state["core_memories"]) + "\n</core_memory>"
    )
    conversational_str = (
            "<conversational_memory>\n" + "\n".join(state["conversational_memories"]) + "\n</conversational_memory>"
    )
    prediction = await bound.ainvoke({
        "messages": state["messages"],
        "core_memories": core_str,
        "conversational_memories": conversational_str,
        "current_time": datetime.now(tz=timezone.utc).isoformat(),
    })
    return {
        "messages": prediction,
    }