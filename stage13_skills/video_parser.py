import os
import json
from typing import TypedDict

from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    LocalShellBackend,
    StoreBackend,
)
import ipdb
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict, Annotated
import operator
from deepagents import create_deep_agent
from dotenv import load_dotenv

from tools import all_tools

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


model = init_chat_model(
    "qwen3.5-27b",
    temperature=0,
    model_provider="openai",
    base_url="http://10.0.0.114/v1",
    api_key=os.getenv("GPUSTACK_API_KEY"),
)

user_home = os.path.expanduser("~")

agent = create_deep_agent(
    model=model,
    system_prompt="""
You are a helpful assistant. when you need to excute python scripts, please use `uv` to avoid environment issues. 
""",
    backend=FilesystemBackend(
        root_dir=".",
    ),
    skills=["./skills/"],
    memory=["./AGENTS.md"],
    tools=all_tools,
)

# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "给我一个当前用户home目录的文件报告"}]},
# )

# # Print the agent's response
# print(result["messages"][-1].content)
