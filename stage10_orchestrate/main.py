import os
import json
from typing import TypedDict

import ipdb
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain.messages import ToolMessage

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


model = init_chat_model(
    "qwen3.5-27b",
    # "qwen3.5-0.8b",
    temperature=0,
    model_provider="openai",
    base_url="http://10.0.0.114/v1",
    api_key=os.getenv("GPUSTACK_API_KEY"),
)
