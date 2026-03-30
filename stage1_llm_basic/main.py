"""
阶段1：LLM 基础 —— 简单聊天机器人

学习目标：理解 LLM API 调用和消息结构
功能：
  1) 调用本地部署的 LLM API（OpenAI 兼容接口）
  2) 用户输入循环：持续接收用户输入
  3) 打印模型回答
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 从 .env 文件加载 API Key
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# 初始化 OpenAI 客户端，指向本地部署的 LLM 服务
client = OpenAI(
    api_key=os.getenv("GPUSTACK_API_KEY"),  # 从环境变量读取 API Key
    base_url="http://10.0.0.114/v1",  # 本地 LLM 服务地址
)

# 模型名称
MODEL = "qwen3.5-27b"

# 消息历史列表，包含系统提示
messages = [
    {"role": "system", "content": "你是一个友好的AI助手，请用中文回答用户问题。"}
]


def chat(user_input: str) -> str:
    """发送用户消息到 LLM 并返回模型回答"""
    # 将用户输入添加到消息历史
    messages.append({"role": "user", "content": user_input})

    # 调用 LLM API
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=1,
        max_tokens=16384,
    )

    # 提取模型回答（strip 处理 Qwen3 thinking 模式的前缀空行）
    assistant_message = (response.choices[0].message.content or "").strip()

    # 将模型回答添加到消息历史，保持上下文连贯
    messages.append({"role": "assistant", "content": assistant_message})

    return assistant_message


def main():
    """主函数：用户输入循环"""
    print("=== LLM 聊天机器人 ===")
    print("输入 'quit' 或 'exit' 退出\n")

    while True:
        # 获取用户输入
        user_input = input("你: ").strip()

        # 检查退出条件
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        # 跳过空输入
        if not user_input:
            continue

        # 调用 LLM 并打印回答
        reply = chat(user_input)
        print(f"\n助手: {reply}\n")


if __name__ == "__main__":
    main()
