from datetime import datetime, timezone

import pytz  # Importing pytz for timezone handling
from langchain_core.runnables.config import RunnableConfig

from src.graph.state import AppState
from src.llms.model import _get_model
from src.prompts.mem_prompt import mem_prompt
from src.settings.config import config_manager
from src.tools.toolkit import toolkit


def get_time() -> datetime:
    return datetime.now(timezone.utc).astimezone(pytz.timezone("America/Los_Angeles"))


async def main_agent(state: AppState, config: RunnableConfig) -> AppState:
    """
    Asynchronous function for the main agent's behavior.

    Parameters:
    - state (AppState): The current state of the conversation.
    - config (RunnableConfig): The configuration settings for the agent.

    Returns:
    AppState: The updated state after processing the agent's behavior.
"""
    configure = config_manager.process_config(config)
    llm_main = configure['em_model']
    temperature = configure['temperature']

    model = _get_model(config, llm_main, 'em_model').bind_tools(toolkit)

    bound = mem_prompt | model

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
        "current_time": get_time(),
    })
    return {
        "messages": prediction,
    }