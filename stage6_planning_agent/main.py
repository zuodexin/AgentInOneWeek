"""
阶段6：Planning Agent —— 任务规划与执行 Agent

学习目标：学习任务分解
功能：
  - Agent 先根据用户目标生成一个 step-by-step plan
  - 然后逐步执行每个步骤
  - 执行过程中可以调用工具
"""

import json
import os
import re
from dotenv import load_dotenv
import ipdb
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
    """计算器工具"""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


def search(query: str) -> str:
    """模拟搜索工具"""
    mock_data = {
        "圆周率": "圆周率 π ≈ 3.14159265358979",
        "光速": "光速约为 299,792,458 米/秒",
        "地球半径": "地球平均半径约为 6,371 千米",
        "水的沸点": "标准大气压下水的沸点为 100°C",
    }
    for key, value in mock_data.items():
        if key in query:
            return value
    return f"关于'{query}'的信息：这是一个需要进一步研究的主题。"


TOOLS = {
    "calculator": {"fn": calculator, "desc": "计算数学表达式"},
    "search": {"fn": search, "desc": "搜索知识信息"},
}


# ============ Plan 生成 ============
PLAN_PROMPT = """你是一个 Planning Agent。给定用户的目标，你需要生成一个执行计划。

可用工具：
- calculator: 计算数学表达式，通过调用python的eval函数进行计算
- search: 搜索知识信息

请生成一个 JSON 格式的计划，包含多个步骤：
{{"plan": [
    {{"step": 1, "description": "步骤描述", "tool": "工具名称或null", "tool_input": "工具参数或null"}},
    {{"step": 2, "description": "步骤描述", "tool": "工具名称或null", "tool_input": "工具参数或null"}}
]}}

注意：
- 如果某个步骤不需要工具，tool 和 tool_input 设为 null
- 最后一步通常是总结答案，不需要工具
- 只输出 JSON，不要写其他内容
"""


def generate_plan(goal: str) -> list:
    """根据用户目标生成执行计划"""
    messages = [
        {"role": "system", "content": PLAN_PROMPT},
        {"role": "user", "content": f"目标: {goal}"},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=2048,
    )

    reply = (response.choices[0].message.content or "").strip()
    # 解析 JSON
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return data.get("plan", [])
        except json.JSONDecodeError:
            ipdb.set_trace()  # 调试 JSON 解析错误
            pass
    ipdb.set_trace()
    return []


# ============ Plan 执行 ============
EXECUTE_PROMPT = """你正在执行计划的一个步骤。

已完成步骤的结果：
{context}

当前步骤: {step_desc}
{tool_result}

请根据以上信息，给出当前步骤的执行结果。简洁明了地回答。
"""


def execute_step(step: dict, context: str) -> str:
    """执行计划中的一个步骤"""
    tool_result = ""

    # 如果步骤需要工具，先调用工具
    if step.get("tool") and step["tool"] in TOOLS:
        tool_name = step["tool"]
        tool_input = step.get("tool_input", "")
        result = TOOLS[tool_name]["fn"](tool_input)
        tool_result = f"工具 {tool_name} 返回: {result}"
        print(f"  调用工具: {tool_name}({tool_input}) → {result}")

    # 让模型根据上下文和工具结果总结当前步骤
    prompt = EXECUTE_PROMPT.format(
        context=context if context else "无",
        step_desc=step["description"],
        tool_result=tool_result if tool_result else "（此步骤不需要工具）",
    )

    messages = [
        {
            "role": "system",
            "content": "你是一个任务执行助手，简洁地回答当前步骤的结果。",
        },
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=1024,
    )

    return (response.choices[0].message.content or "").strip()


def run_planning_agent(goal: str) -> str:
    """运行 Planning Agent：生成计划 → 逐步执行"""
    print(f"\n{'='*50}")
    print(f"目标: {goal}")
    print(f"{'='*50}")

    # 第一阶段：生成计划
    print("\n📋 正在生成计划...")
    plan = generate_plan(goal)

    if not plan:
        return "无法生成执行计划。"

    print(f"\n生成了 {len(plan)} 个步骤:")
    for step in plan:
        tool_info = f" [工具: {step.get('tool')}]" if step.get("tool") else ""
        print(f"  步骤 {step['step']}: {step['description']}{tool_info}")

    # 第二阶段：逐步执行
    print(f"\n{'='*50}")
    print("开始执行计划...")
    context_parts = []

    for step in plan:
        print(f"\n--- 执行步骤 {step['step']}: {step['description']} ---")
        context = "\n".join(context_parts)
        result = execute_step(step, context)
        print(f"  结果: {result[:200]}")
        context_parts.append(f"步骤{step['step']}: {result[:200]}")

    # 返回最后一步的结果作为最终答案
    final = context_parts[-1] if context_parts else "未得出结果"
    print(f"\n{'='*50}")
    print(f"最终结果: {final}")
    return final


def main():
    print("=== Planning Agent（任务规划） ===")
    print("输入你的目标，Agent 会制定计划并逐步执行")
    print("输入 'quit' 退出\n")

    while True:
        goal = input("目标: ").strip()
        if goal.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not goal:
            continue
        run_planning_agent(goal)
        print()


if __name__ == "__main__":
    main()
