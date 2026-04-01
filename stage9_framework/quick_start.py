import os
import json
import re

import ipdb
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# ── Define tools as plain functions ──────────────────────────────────────────
from langchain.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


@tool
def current_date_number() -> int:
    """Returns the current date number (1-31)"""
    from datetime import datetime

    return datetime.now().day


tools = [add, multiply, divide, current_date_number]
tools_by_name = {
    "add": add,
    "multiply": multiply,
    "divide": divide,
    "current_date_number": current_date_number,
}


# import fix_qw

model = init_chat_model(
    # "qwen3.5-27b",
    "qwen3.5-0.8b",
    temperature=0,
    model_provider="openai",
    base_url="http://10.0.0.114/v1",
    api_key=os.getenv("GPUSTACK_API_KEY"),
)

# from langchain_qwq import ChatQwen

# model = ChatQwen(
#     model="qwen3.5-27b",
#     base_url="http://10.0.0.114/v1",
#     api_key=os.getenv("GPUSTACK_API_KEY"),
# )

model_with_tools = model.bind_tools(
    tools
)  # 服务器需要enable_auto_tool_choice，配置tool-call-parser

# msg = model_with_tools.invoke("What's 5 times forty two")
# ipdb.set_trace()


from langchain.messages import ToolMessage


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


# ── Build workflow ────────────────────────────────────────────────────────────

agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent graph
print(agent.get_graph(xray=True).draw_ascii())
try:
    from IPython.display import Image, display

    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    print(agent.get_graph(xray=True).draw_ascii())

# ── Invoke ────────────────────────────────────────────────────────────────────

messages = [HumanMessage(content="What's 5 times current date number?")]
result = agent.invoke({"messages": messages})
for m in result["messages"]:
    m.pretty_print()
