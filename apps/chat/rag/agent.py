import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from apps.chat.rag import graph, tools, pdfloader, build_graph
from config.logger import get_logger

logger = get_logger(__name__)

rag_agent = None


def init_rag():
    """
        This initializes the rag agent.
        - loads environment variables
        - sets up LLM and tools
        - injects dependencies into graph and tools
        - builds the langgraph graph
    """

    global rag_agent

    load_dotenv() #loads data from .env file

    #Initialize the LLM
    llm_model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    #bind the tools to LLM
    llm_model = llm_model.bind_tools(list(tools.tools_dict.values()))

    #Inject into graph
    graph.llm_model = llm_model
    graph.sys_prompt = os.getenv(
        "SYSTEM_PROMPT",
        """
        You are a RAG agent with access to tools, who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
        Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
        If you need to look up some information before asking a follow up question, you are allowed to do that!
        Please always cite the specific parts of the documents you use in your answers.
        Answer clearly and concisely.
        You will NOT perform any other task or answer any question NOT related to the Stock Market Performance PDF.
        """
    )

    graph.tools_dict = tools.tools_dict

    #Initialize retriever
    retriever = pdfloader.init_retriever()
    tools.retriever = retriever

    #Build the graph
    rag_agent = build_graph.init_graph()

    logger.info("RAG agent initialization complete")


def run_agent(user_input: str) -> str:
    """
        This runs the agent.
        Args:
            user_input: user input to pass to LLM
        Returns:
            str: response from LLM
    """
    global rag_agent

    if rag_agent is None:
        raise RuntimeError("Agent not initialized. Call init_rag() first.")
    
    state = {
        "messages" : [HumanMessage(content=user_input)],
        "tool_call_count": 0,
    }

    final_state = rag_agent.invoke(state)

    last_message = final_state["messages"][-1]
    logger.info(last_message)
    if isinstance(last_message, str):
        return last_message
    elif hasattr(last_message, "content"):
        return last_message.content
    else:
        return str(last_message)