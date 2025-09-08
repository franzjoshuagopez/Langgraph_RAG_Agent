from typing import TYPE_CHECKING, Optional
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from .state import RAGState #imports RAGState class from state.py
from .tools import current_state
from config.logger import get_logger #imports get_logger function from logger.py

logger = get_logger(__name__)

if TYPE_CHECKING:
    from langchain_groq import ChatGroq
    from langchain_core.tools import BaseTool

#Global placeholders
llm_model: Optional[ChatGroq] = None
sys_prompt : str = ""
tools_dict: dict[str, "BaseTool"] = {}
MAX_TOOL_CALLS = 3

def call_llm(state: RAGState) -> RAGState:
    global current_state
    current_state = state
    """
        Calls the llm and submits/passes the system prompt, user message, and tool response (when tool call is called by LLM).
        appends llm response to state
    """

    global llm_model, sys_prompt

    if llm_model is None:
        raise RuntimeError("LLM model has not been initialized. Did you call init_rag()?")

    messages = list(state["messages"])

    if state.get("tool_call_count", 0) > MAX_TOOL_CALLS: #This checks if the llm has been looping and enforces that the llm has used the tool usage count
        system_prompt = (
            sys_prompt +
            "\n\nYou have reached the maximum number of tool calls."
            "Do not attempt to call any tools or functions."
            "Only respond in plain text with your best possible answer using the data already provided"
        ) #updates the system prompt telling the llm it cannot call anymore tools and provide its final answer based on gathered data so far
    else:
        system_prompt = sys_prompt
    
    messages = [SystemMessage(content=system_prompt)] + messages
    
    result_message = llm_model.invoke(messages)

    logger.info("LLM has responded")
    logger.debug(f"LLM raw content: {getattr(result_message, 'content', '')[:100]}")

    state["messages"].append(result_message) #updates state with latest llm response including tool call

    return state

def execute_tools(state: RAGState) -> RAGState:
    """
        Executes any tool call from the last LLM message
    """
    global tools_dict
    last_message = state["messages"][-1] #gets latest message
    tool_calls = getattr(last_message, "tool_calls", []) #gets the tool call from the latest state message
    results = []

    for t in tool_calls: #tool_calls is list of dictionary it contains [id:value, name:value, args:dict]
        query = t.get("args", {}).get("query", "") #args is a dictionary containing name : value in this instance it 'query' : 'some text'
        tool_name = t.get("name")
        logger.info(f"Tool call detected: {tool_name} with query: {query}")

        if tool_name not in tools_dict:
            logger.warning(f"Tool '{tool_name}' not found in tools_dict")
            tool_result = "Incorrect Tool name, Please retry and select the correct Tool."
        else:
            try:
                tool_result = tools_dict[tool_name].invoke(query)
                state["tool_call_count"] += 1
                logger.info(f"Executed Tool '{tool_name}' sucessfully")
            except Exception as e:
                tool_result = f"Tool {tool_name} failed with error {e}"
                logger.exception(f"Error while executing tool {tool_name} with error: {e}")

        results.append(
            ToolMessage(
                tool_call_id=t.get('id'),
                name=tool_name,
                content=str(tool_result),
            )
        ) #appends ToolMessage to results variable

    state["messages"].extend(results) #extend is used to pass all parameters of the message in this instance result to the state["messages"] and not just the content which append does

    return state

def check_continue(state: RAGState) -> bool:
    """
        Checks if the last message is a Tool call and returns True, otherwise False
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", []) #basically get attribute value from last_message named tool_callse otherwise empty []
    should_continue = len(tool_calls) > 0 and state.get("tool_call_count", 0) <= MAX_TOOL_CALLS
    logger.info(f"check continue -> {should_continue}. tool_call_count: {state.get('tool_call_count', 0)}")

    return should_continue