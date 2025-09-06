from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from .state import RAGState
from .graph import call_llm, execute_tools, check_continue
from config.logger import get_logger

logger = get_logger(__name__)


def init_graph() -> CompiledStateGraph:
    """
        Build and compiles the RAG agent's graph
        returns:
            graph: Compiled State Graph
    """
    
    logger.info("Building graph...")

    graph = StateGraph(RAGState)

    graph.add_node("llm", call_llm)
    graph.add_node("execute_tools", execute_tools)

    graph.add_edge(START, "llm")
    graph.add_conditional_edges(
        "llm",
        check_continue,
        {
            True : "execute_tools",
            False : END
        }
    )

    graph.add_edge("execute_tools", "llm")

    logger.info("Graph built succesfully")

    return graph.compile()