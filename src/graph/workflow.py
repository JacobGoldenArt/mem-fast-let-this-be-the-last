from langgraph.graph import START, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt.tool_node import ToolNode, tools_condition

from src.graph.state import AppState
from src.llms.main_agent import main_agent
from src.settings.app_config import AppConfig
from src.tools.memory.load_memories import load_memories
from src.tools.toolkit import toolkit


def create_graph() -> CompiledGraph:
    """
    Create the graph for the memory agent.
    """
    workflow = StateGraph(AppState, AppConfig)

    # add nodes to the graph
    workflow.add_node(load_memories)
    workflow.add_node(main_agent)
    workflow.add_node("tools", ToolNode(toolkit))

    # add edges to the graph
    workflow.add_edge(START, "load_memories")
    workflow.add_edge("load_memories", "main_agent")
    workflow.add_conditional_edges("main_agent", tools_condition)
    workflow.add_edge("tools", "main_agent")

    # compile the graph
    memgraph = workflow.compile()
    return memgraph