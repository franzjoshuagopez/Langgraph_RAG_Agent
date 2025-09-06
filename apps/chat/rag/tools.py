from langchain.tools import tool
from config.logger import get_logger

logger = get_logger(__name__)

retriever = None


@tool
def retriever_tool(query: str) -> str:
    """
        A tool that retrieves relevant data from documents stored in the vector database.
        Args:
            query: str = user's search query
        returns:
            str: Formatted text containing relavant data from documents
    """
    global retriever

    if retriever is None:
        logger.warning("Retriever has not been initialized yet.")
        return "Retriever is not available"

    try:
        results = []
        docs = retriever.get_relevant_documents(query)
        # if we want to limit the number of documents, change to for i, doc in enumerate(docs[:n]): where n is the max documents to loop through
        for i, doc in enumerate(docs): #enumerate returns a list of LangChain Document objects and its index, which in this for loop is doc and i respectively
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "N/A")
            results.append(f"Document {i+1} (source: {source}, page: {page}):\n{doc.page_content}")
        
        return "\n\n".join(results)
    
    except Exception as e:
        logger.exception(f"Error while running retriever tool: {e}")
        
        return f"Retriever tool failed due to: {e}"

#declare tools dictionary for graph
tools_dict = {
    "retriever_tool" : retriever_tool
}