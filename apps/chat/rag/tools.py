from langchain.tools import tool
from typing import TypedDict
from config.logger import get_logger


MAX_TOOL_CALLS = 3

logger = get_logger(__name__)

retriever = None

current_state: dict = {}

@tool
def retriever_tool(query: str) -> str:
    """
        A tool that retrieves relevant data from documents stored in the vector database.
        Args:
            query: str = user's search query
        returns:
            str: Formatted text containing relavant data from documents
    """
    logger.info(f"Retriever tool started...")

    global current_state

    global retriever

    if retriever is None:
        logger.warning("Retriever has not been initialized yet.")
        return "Retriever is not available"
    
    if current_state.get("tool_call_count", 0) > MAX_TOOL_CALLS:
        logger.warning("Maximum tool calls reached - skipping tool execution")
        return "Maximum tool calls reached. Answer based on the context available so far."

    try:
        results = []
        docs = retriever.get_relevant_documents(query)
        # if we want to limit the number of documents, change to for i, doc in enumerate(docs[:n]): where n is the max documents to loop through
        for i, doc in enumerate(docs): #enumerate returns a list of LangChain Document objects and its index, which in this for loop is doc and i respectively
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            results.append(f"Document {i+1} (source: {source}, page: {page}):\n{doc.page_content}")
        
        logger.info(f"Retriever tool completed its task")
        
        return "\n\n".join(results)
    
    except Exception as e:
        logger.exception(f"Error while running retriever tool: {e}")
        
        return f"Retriever tool failed due to: {e}"

#declare tools dictionary for graph
tools_dict = {
    "retriever_tool" : retriever_tool
}