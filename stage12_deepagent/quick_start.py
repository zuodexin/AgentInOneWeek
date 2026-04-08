import os
import json
from typing import TypedDict

from deepagents.backends import FilesystemBackend, LocalShellBackend
import ipdb
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict, Annotated
import operator
from deepagents import create_deep_agent

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


# def cli(command: str) -> str:
#     """Executes a CLI command and returns the output.
#     Args:
#         command: The CLI command to execute.
#     """
#     import subprocess

#     result = subprocess.run(command, shell=True, capture_output=True, text=True)
#     return result.stdout.strip()


model = init_chat_model(
    "qwen3.5-27b",
    # "qwen3.5-0.8b",
    temperature=0,
    model_provider="openai",
    base_url="http://10.0.0.114/v1",
    api_key=os.getenv("GPUSTACK_API_KEY"),
)

agent = create_deep_agent(
    model=model,
    # tools=[cli],
    system_prompt="""
You are a helpful assistant.
""",
    backend=LocalShellBackend(root_dir=".", env={"PATH": "/usr/bin:/bin"}),
)


# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "给我一个当前用户home目录的文件报告"}]},
# )

# # Print the agent's response
# print(result["messages"][-1].content)
