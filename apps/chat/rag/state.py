from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

#This is the Graph State
#It contains messages of List with type hint of BaseMessage. This will contain all messages such as HumanMessage, AIMessage, ToolMessage, SystemMessage
#It contains number of tool calls of type int to manage how many times LLM has called the tools to prevent infinite loops
class RAGState(TypedDict):
    messages : Annotated[List[BaseMessage], add_messages]
    tool_call_count : int
