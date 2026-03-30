"""
阶段4：Tool Agent —— 多工具 Agent

学习目标：学习工具系统设计
功能：
  - 定义 Tool 类（包含 name、description、function）
  - 实现两个工具：calculator 和 search
  - Agent 根据模型输出决定调用哪个工具
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Callable
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    api_key=os.getenv("GPUSTACK_API_KEY"),
    base_url="http://10.0.0.114/v1",
)

MODEL = "qwen3.5-27b"


# ============ Tool 类定义 ============
@dataclass
class Tool:
    """工具类：封装工具的名称、描述和执行函数"""

    name: str  # 工具名称
    description: str  # 工具功能描述（供模型理解）
    function: Callable  # 工具实际执行的函数


# ============ 工具实现 ============
def calculator(expression: str) -> str:
    """计算器工具：计算数学表达式"""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


def search(query: str) -> str:
    """模拟搜索工具：返回模拟的搜索结果"""
    # 这是一个模拟的搜索工具，实际项目中可接入真实搜索 API
    mock_database = {
        "python": "Python 是一种广泛使用的高级编程语言，由 Guido van Rossum 于 1991 年发布。",
        "agent": "AI Agent 是能够感知环境、做出决策并采取行动的智能系统。",
        "react": "ReAct 是一种结合推理（Reasoning）和行动（Acting）的 Agent 框架。",
        "transformer": "Transformer 是 2017 年由 Google 提出的神经网络架构，是现代 LLM 的基础。",
        "llm": "大语言模型（LLM）是在大规模文本数据上训练的深度学习模型，能够理解和生成自然语言。",
    }
    query_lower = query.lower()
    for key, value in mock_database.items():
        if key in query_lower:
            return value
    return f"未找到与 '{query}' 相关的结果。"


# ============ 注册工具 ============
tools = [
    Tool(
        name="calculator",
        description="计算数学表达式。输入参数为数学表达式字符串，如 '3 + 5 * 2'。",
        function=calculator,
    ),
    Tool(
        name="search",
        description="搜索知识信息。输入参数为搜索关键词字符串，如 'Python编程语言'。",
        function=search,
    ),
]

# 构建工具名称到工具对象的映射
tool_map = {tool.name: tool for tool in tools}


# ============ 构建 System Prompt ============
def build_system_prompt(tools: list[Tool]) -> str:
    """根据已注册工具动态生成 system prompt"""
    tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    return f"""你是一个 Tool Agent，你可以使用工具来回答问题。

可用工具：
{tool_descriptions}

请按照以下 JSON 格式回复：

当需要使用工具时：
{{"thought": "你的推理", "action": "工具名称", "action_input": "工具参数"}}

当知道最终答案时：
{{"thought": "你的推理", "answer": "最终答案"}}

注意：每次只输出一个 JSON 对象。必须严格使用 JSON 格式。
"""


def parse_response(text: str) -> dict:
    """从模型输出中解析 JSON 响应"""
    # 尝试找到 JSON 块
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def run_tool_agent(question: str, max_steps: int = 10) -> str:
    """运行 Tool Agent"""
    system_prompt = build_system_prompt(tools)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    print(f"\n{'='*50}")
    print(f"问题: {question}")
    print(f"{'='*50}")

    for step in range(max_steps):
        print(f"\n--- 步骤 {step + 1} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            max_tokens=1024,
        )

        reply = (response.choices[0].message.content or "").strip()
        print(f"模型输出: {reply}")
        messages.append({"role": "assistant", "content": reply})

        # 解析模型响应
        parsed = parse_response(reply)

        if "thought" in parsed:
            print(f"思考: {parsed['thought']}")

        # 检查是否为最终答案
        if "answer" in parsed:
            print(f"\n最终答案: {parsed['answer']}")
            return parsed["answer"]

        # 检查是否需要调用工具
        if "action" in parsed and "action_input" in parsed:
            tool_name = parsed["action"]
            tool_input = parsed["action_input"]

            if tool_name in tool_map:
                # 执行工具调用
                tool = tool_map[tool_name]
                result = tool.function(tool_input)
                print(f"调用工具: {tool_name}({tool_input})")
                print(f"工具结果: {result}")

                # 将工具结果反馈给模型
                observation = f"工具 {tool_name} 返回结果: {result}"
                messages.append({"role": "user", "content": observation})
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": f"错误：工具 '{tool_name}' 不存在。可用工具: {list(tool_map.keys())}",
                    }
                )
        else:
            messages.append({"role": "user", "content": "请按照要求的 JSON 格式回复。"})

    return "达到最大步骤数，未能得出答案。"


def main():
    print("=== Tool Agent（多工具） ===")
    print("可用工具: calculator（计算）、search（搜索）")
    print("输入 'quit' 退出\n")

    while True:
        question = input("问题: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not question:
            continue
        run_tool_agent(question)
        print()


if __name__ == "__main__":
    main()
