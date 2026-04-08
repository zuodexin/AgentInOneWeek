import os
import json
import re

import ipdb
from langchain_qwq import ChatQwen
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.messages import ToolMessage
from langchain.tools import tool

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


class MessagesState(TypedDict):
    messages: Annotated[
        list[AnyMessage], operator.add
    ]  # Annotated给类型附加额外的元信息（metadata）, Annotated[类型, 元数据], 框架可以读取这些信息做特殊处理
    llm_calls: int


# ── Define tools as plain functions ──────────────────────────────────────────


@tool
def cli(command: str) -> str:
    """Executes a CLI command and returns the output.
    Args:
        command: The CLI command to execute.
    """
    import subprocess

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()


tools = [cli]


model = ChatQwen(
    model="qwen3.5-27b",
    base_url="http://10.0.0.114/v1",
    api_key=os.getenv("GPUSTACK_API_KEY"),
)

# model = ChatQwen(
#     model="qwen3.5-27b",
#     temperature=0.3,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     api_key=os.getenv("ALIYUN_API_KEY"),
# )

model_with_tools = model.bind_tools(
    tools
)  # 服务器需要enable_auto_tool_choice，配置tool-call-parser


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with several tools. do not reply empty message."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ── Build workflow ────────────────────────────────────────────────────────────

agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", ToolNode(tools=tools))

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call", tools_condition, {"tools": "tool_node", END: END}
)
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

messages = [HumanMessage(content="生成关于/home/dexin的报告, 找出比较大的目录和文件")]
# messages = [HumanMessage(content="生成/mnt/data2目录的报告")]
# messages = [HumanMessage(content="统计当前目录的平均文件大小")]
result = agent.invoke(
    {"messages": messages},
    stream_mode=["messages", "updates", "custom"],  # 直接在 invoke 时使用
    version="v2",
)
for chunk in result:
    if chunk["type"] == "messages":
        message_chunk, metadata = chunk["data"]
        if isinstance(message_chunk, ToolMessage):
            print(f"\n[⛏] {message_chunk.content}")
        else:
            print(message_chunk.content, end="", flush=True)

    elif chunk["type"] == "updates":
        # 每个节点的状态更新
        for node_name, state in chunk["data"].items():
            # 打印工具调用
            if node_name == "llm_call":
                last_message = state["messages"][-1]
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    print(f"\n[⛏] {last_message.tool_calls}")

    elif chunk["type"] == "custom":
        # 你自定义的状态
        status = chunk["data"].get("status")
        if status:
            print(f"\n[Custom Status] {status}")

# result = agent.invoke({"messages": messages})
# for m in result["messages"]:
#     m.pretty_print()
