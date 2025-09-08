import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from apps.chat.rag import graph, tools, pdfloader, build_graph
from apps.chat.rag.router import RouterRetriever
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

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    prompt_path = BASE_DIR / "config" / "system_prompt.txt"

    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read()
    else:
        SYSTEM_PROMPT = """
            You are FranzAI, a helpful and friendly financial assistant. 
            You ONLY answer questions related to the Stock Market, and exchange listings based on the PDFs provided in your knowledge base. 

            When responding:
            - Speak in a conversational, natural way (like chatting with a colleague).
            - if the user is just greeting or making small talk (e.g., 'hello', 'hi', 'how are you?', 'who are you?') respond politely without calling any tools
            - Use plain language while keeping key details accurate. 
            - If the document gives specific numbers, companies, or names, include them — but explain them in your own words.
            - Avoid sounding like you're quoting the document directly. Summarize naturally.
            - If the user requests you to give your opinion or insight of a current topic in the stock market you are allowed to do so using the data in the pdf documents.
            - If the user asks something that is not about the stock market, exchange listings or outside the documents you have been provided, politely refuse and say something like: 
            “I can only answer questions about the stock market, exchange listings based on the documents I have.”
            - Only respond to the user's latest message.
            - Use older messages only as supporting context if they are relevant.
            - Do not repeat or re-answer earlier questions unless the user explicitly asks again.
            - Do not perform unrelated tasks or answer off-topic questions.
            - You can only call tools a maximum of 3 times per user query.

            Your goal: sound clear, natural, and human-like, while staying strictly grounded in the PDF's content.
        """

    #Initialize the LLM
    llm_model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    #bind the tools to LLM
    llm_model = llm_model.bind_tools(list(tools.tools_dict.values()))

    #Inject into graph
    graph.llm_model = llm_model

    graph.sys_prompt = SYSTEM_PROMPT

    graph.tools_dict = tools.tools_dict

    #Initialize retrievers
    listing_retriever, market_retriever = pdfloader.init_retriever()
    tools.retriever = RouterRetriever(listing_retriever, market_retriever, logger)

    #Build the graph
    rag_agent = build_graph.init_graph()

    logger.info("RAG agent initialization complete")


def run_agent(previous_messages, user_input: str) -> str:
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
    
    #Convert DB messages to Langchain Mesages
    message_history = []
    context_history = []

    for msg in previous_messages:
        if msg.role == "user":
            message_history.append(HumanMessage(content=msg.content))
        else:
            message_history.append(AIMessage(content=msg.content))
    
    last_10 = message_history[-10:]

    context_history.append(SystemMessage(content="Here is the chat so far: " + "\n".join(m.content for m in last_10)))
    
    context_history.append(HumanMessage(content=user_input))
    logger.info(f"last appended message: {context_history[-1].content}")
    state = {
        "messages" : context_history,
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