"""
阶段3：ReAct Agent —— 带计算器的推理 Agent

学习目标：理解 Agent 基本思想（Reason + Act）
核心流程：Thought → Action → Observation → Answer
Agent 通过循环让模型决定是否需要调用工具
"""

import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    api_key=os.getenv("GPUSTACK_API_KEY"),
    base_url="http://10.0.0.114/v1",
)

MODEL = "qwen3.5-27b"


# ============ 工具定义 ============
def calculator(expression: str) -> str:
    """安全计算器：计算数学表达式"""
    try:
        # 只允许数字和基本运算符，防止代码注入
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"错误：表达式包含不允许的字符"
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


# ============ ReAct Prompt ============
REACT_SYSTEM_PROMPT = """你是一个 ReAct Agent。你可以通过推理和使用工具来回答问题。

你可以使用以下工具：
- calculator(expression): 计算数学表达式，例如 calculator(3 + 5 * 2)

请严格按照以下格式回答，每次只输出一个步骤：

Thought: <你的推理过程>
Action: calculator(<表达式>)

或者当你知道最终答案时：

Thought: <你的推理过程>
Answer: <最终答案>

注意：每次只输出一个 Thought + Action 或 Thought + Answer。不要一次输出多个步骤。
"""


def parse_action(text: str):
    """从模型输出中解析 Action 调用"""
    # 匹配 Action: calculator(...) 格式
    match = re.search(r"Action:\s*calculator\((.+?)\)", text)
    if match:
        return match.group(1).strip()
    return None


def parse_answer(text: str):
    """从模型输出中解析最终 Answer"""
    match = re.search(r"Answer:\s*(.+)", text)
    if match:
        return match.group(1).strip()
    return None


def run_react_agent(question: str, max_steps: int = 10) -> str:
    """运行 ReAct Agent 循环"""
    # 初始化消息
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    print(f"\n{'='*50}")
    print(f"问题: {question}")
    print(f"{'='*50}")

    for step in range(max_steps):
        print(f"\n--- 步骤 {step + 1} ---")

        # 调用 LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            max_tokens=1024,
        )

        assistant_reply = (response.choices[0].message.content or "").strip()
        print(f"模型输出:\n{assistant_reply}")

        # 将模型回复加入消息历史
        messages.append({"role": "assistant", "content": assistant_reply})

        # 检查是否有最终答案
        answer = parse_answer(assistant_reply)
        if answer:
            print(f"\n最终答案: {answer}")
            return answer

        # 检查是否有 Action 需要执行
        expression = parse_action(assistant_reply)
        if expression:
            # 执行工具调用
            result = calculator(expression)
            observation = f"Observation: {result}"
            print(f"工具调用: calculator({expression})")
            print(f"计算结果: {result}")

            # 将 Observation 作为新消息反馈给模型
            messages.append({"role": "user", "content": observation})
        else:
            # 模型既没有给出 Answer 也没有调用 Action，提示继续
            messages.append(
                {
                    "role": "user",
                    "content": "请按照格式继续推理，输出 Thought + Action 或 Thought + Answer。",
                }
            )

    return "达到最大步骤数，未能得出答案。"


def main():
    """主函数"""
    print("=== ReAct Agent（带计算器） ===")
    print("输入问题，Agent 会推理并使用计算器求解")
    print("输入 'quit' 退出\n")

    while True:
        question = input("问题: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if not question:
            continue

        run_react_agent(question)
        print()


if __name__ == "__main__":
    main()
