"""
阶段5：Memory Agent —— 带对话记忆的 Agent

学习目标：学习 Agent 记忆机制
功能：
  - 保存对话历史（conversation memory）
  - 每次调用模型时将历史作为 context 发送
  - 展示 memory 数据结构以及如何更新 memory
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    api_key=os.getenv("GPUSTACK_API_KEY"),
    base_url="http://10.0.0.114/v1",
)

MODEL = "qwen3.5-27b"


# ============ Memory 数据结构 ============
@dataclass
class ConversationMemory:
    """对话记忆：存储和管理对话历史

    属性:
        messages: 完整的消息列表
        max_turns: 最大保留的对话轮数（一轮 = 用户 + 助手）
        summary: 早期对话的摘要（当历史过长时压缩）
    """

    messages: list = field(default_factory=list)
    max_turns: int = 10  # 最多保留10轮对话
    summary: str = ""  # 早期对话的摘要

    def add_message(self, role: str, content: str):
        """添加一条消息到记忆中"""
        self.messages.append({"role": role, "content": content})
        # 检查是否需要压缩记忆
        self._compress_if_needed()

    def get_messages(self) -> list:
        """获取用于发送给模型的消息列表"""
        result = []
        # 如果有摘要，先添加摘要作为上下文
        if self.summary:
            result.append(
                {"role": "system", "content": f"以下是之前对话的摘要: {self.summary}"}
            )
        # 添加当前保留的消息
        result.extend(self.messages)
        return result

    def _compress_if_needed(self):
        """当对话轮数超过限制时，压缩早期消息为摘要"""
        # 计算当前轮数（每轮 = user + assistant）
        pairs = []
        i = 0
        while i < len(self.messages) - 1:
            if (
                self.messages[i]["role"] == "user"
                and self.messages[i + 1]["role"] == "assistant"
            ):
                pairs.append((self.messages[i], self.messages[i + 1]))
                i += 2
            else:
                i += 1

        if len(pairs) > self.max_turns:
            # 压缩最早的一半对话为摘要
            compress_count = len(pairs) // 2
            compressed_text = []
            for user_msg, asst_msg in pairs[:compress_count]:
                compressed_text.append(
                    f"用户问: {user_msg['content'][:50]}... "
                    f"助手答: {asst_msg['content'][:50]}..."
                )
            new_summary = "; ".join(compressed_text)
            self.summary = (self.summary + "; " + new_summary).strip("; ")

            # 保留未压缩的消息
            remove_count = compress_count * 2
            self.messages = self.messages[remove_count:]

    def get_stats(self) -> str:
        """获取记忆状态统计"""
        return (
            f"消息数: {len(self.messages)}, "
            f"有摘要: {'是' if self.summary else '否'}"
        )

    def clear(self):
        """清空记忆"""
        self.messages.clear()
        self.summary = ""


# ============ Memory Agent ============
class MemoryAgent:
    """带记忆的对话 Agent"""

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.memory = ConversationMemory(max_turns=10)

    def chat(self, user_input: str) -> str:
        """处理用户输入并返回回复"""
        # 将用户消息加入记忆
        self.memory.add_message("user", user_input)

        # 构建发送给模型的完整消息列表
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.memory.get_messages())

        # 调用 LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=16384,
        )

        reply = (response.choices[0].message.content or "").strip()

        # 将助手回复加入记忆
        self.memory.add_message("assistant", reply)

        return reply


def main():
    print("=== Memory Agent（带记忆的对话） ===")
    print("Agent 会记住之前的对话内容")
    print("命令: 'quit' 退出, 'memory' 查看记忆状态, 'clear' 清空记忆\n")

    agent = MemoryAgent(
        system_prompt="你是一个友好的AI助手，你会记住用户之前说过的内容并在对话中引用。请用中文回答。"
    )

    while True:
        user_input = input("你: ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if user_input.lower() == "memory":
            print(f"[记忆状态] {agent.memory.get_stats()}")
            if agent.memory.summary:
                print(f"[摘要] {agent.memory.summary}")
            continue

        if user_input.lower() == "clear":
            agent.memory.clear()
            print("[记忆已清空]")
            continue

        if not user_input:
            continue

        reply = agent.chat(user_input)
        print(f"\n助手: {reply}\n")


if __name__ == "__main__":
    main()
