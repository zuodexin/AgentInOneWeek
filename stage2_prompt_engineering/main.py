"""
阶段2：Prompt 控制 —— 数学助手

学习目标：理解 Prompt Engineering
功能：
  - 通过精心设计的 prompt 模板引导模型进行数学推理
  - 模型会一步一步推理并给出答案
  - 支持连续提问
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("GPUSTACK_API_KEY"),
    base_url="http://10.0.0.114/v1",
)

MODEL = "qwen3.5-27b"

# ============ Prompt 模板 ============
# 系统提示：定义模型的角色和行为方式
SYSTEM_PROMPT = """你是一个专业的数学助手。当用户提出数学问题时，你需要：

1. **理解问题**：首先明确题目要求。
2. **列出已知条件**：整理题目给出的信息。
3. **分步推理**：用清晰的步骤一步一步解题，每步给出计算过程。
4. **得出答案**：在最后明确给出最终答案。

注意事项：
- 如果问题有多种解法，选择最简洁的一种。
- 如果问题描述不清，请先确认再解答。
- 使用数学符号时要清晰易读。
"""

# 用户输入的包装模板：进一步强调推理过程
USER_PROMPT_TEMPLATE = """请解答以下数学问题，要求分步推理：

问题：{question}

请按照以下格式回答：
【分析】简要分析题意
【解题步骤】逐步推理
【答案】最终答案
"""


def solve_math(question: str) -> str:
    """使用 prompt engineering 让模型分步解答数学问题"""
    # 使用模板格式化用户输入
    formatted_user_message = USER_PROMPT_TEMPLATE.format(question=question)

    # 构建消息列表：system prompt + 格式化的用户消息
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": formatted_user_message},
    ]

    # 调用 LLM API
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,  # 数学问题用较低温度，减少随机性
        max_tokens=16384,
    )

    return (response.choices[0].message.content or "").strip()


def main():
    """主函数：数学助手交互循环"""
    print("=== 数学助手 ===")
    print("输入数学问题，我会一步一步帮你解答")
    print("输入 'quit' 退出\n")

    while True:
        question = input("数学问题: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if not question:
            continue

        print("\n正在思考...\n")
        answer = solve_math(question)
        print(f"{answer}\n")


if __name__ == "__main__":
    main()
