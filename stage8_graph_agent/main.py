"""
阶段8：Graph Agent —— 基于状态机的 Agent Workflow

学习目标：理解 Agent Workflow
功能：
  - 定义多个节点：plan、tool、reflect、finish
  - 使用字典表示状态转移图
  - 循环执行直到 finish 节点
"""

import json
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


# ============ 工具 ============
def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "错误：不允许的字符"
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"错误: {e}"


def search(query: str) -> str:
    data = {
        "圆周率": "π ≈ 3.14159",
        "光速": "约 3×10^8 m/s",
        "地球": "地球半径约 6371 km",
        "python": "Python 是高级编程语言",
    }
    for k, v in data.items():
        if k in query.lower():
            return v
    return f"关于'{query}'的信息暂无。"


TOOLS = {"calculator": calculator, "search": search}


# ============ Agent 状态 ============
class AgentState:
    """Agent 的全局状态，在节点之间传递"""

    def __init__(self, goal: str):
        self.goal = goal  # 用户目标
        self.plan = ""  # 当前计划
        self.tool_results = []  # 工具调用结果列表
        self.reflection = ""  # 反思内容
        self.answer = ""  # 最终答案
        self.current_node = "plan"  # 当前节点
        self.iterations = 0  # 迭代次数


def call_llm(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=2048,
    )
    return (response.choices[0].message.content or "").strip()


# ============ 节点函数 ============
def plan_node(state: AgentState) -> str:
    """计划节点：根据目标生成执行计划"""
    print("\n📋 [Plan 节点] 生成计划...")

    context = ""
    if state.reflection:
        context = f"\n之前的反思: {state.reflection}"
    if state.tool_results:
        context += f"\n已有工具结果: {state.tool_results}"

    prompt = f"目标: {state.goal}{context}\n\n请生成简洁的执行计划（3步以内），并判断是否需要使用工具(calculator/search)。\n格式：\n1. 步骤\n需要工具: 是/否\n工具名: xxx\n工具参数: xxx"

    result = call_llm("你是一个任务规划助手，生成简洁的执行计划。", prompt)
    state.plan = result
    print(f"  计划: {result[:200]}")

    # 判断下一个节点
    if "需要工具: 是" in result or "需要工具：是" in result:
        return "tool"
    else:
        return "finish"


def tool_node(state: AgentState) -> str:
    """工具节点：根据计划调用工具"""
    print("\n🔧 [Tool 节点] 调用工具...")

    # 从计划中提取工具信息
    prompt = f'计划:\n{state.plan}\n\n请提取需要调用的工具，以JSON格式返回:\n{{"tool": "工具名", "input": "参数"}}'

    result = call_llm("提取工具调用信息，只返回JSON。", prompt)

    match = re.search(r"\{.*\}", result, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            tool_name = data.get("tool", "")
            tool_input = data.get("input", "")
            if tool_name in TOOLS:
                tool_result = TOOLS[tool_name](tool_input)
                state.tool_results.append(f"{tool_name}({tool_input}) = {tool_result}")
                print(f"  调用: {tool_name}({tool_input}) → {tool_result}")
            else:
                print(f"  未知工具: {tool_name}")
        except json.JSONDecodeError:
            print(f"  JSON 解析失败: {result}")

    return "reflect"


def reflect_node(state: AgentState) -> str:
    """反思节点：评估当前进展，决定下一步"""
    print("\n🤔 [Reflect 节点] 反思分析...")

    prompt = (
        f"目标: {state.goal}\n"
        f"计划: {state.plan}\n"
        f"工具结果: {state.tool_results}\n\n"
        f"请反思：\n1. 目标是否已达成？\n2. 是否需要更多工具调用？\n\n"
        f"回答格式:\n状态: 完成/需要更多工具/需要重新规划\n分析: 你的分析"
    )

    result = call_llm("你是反思助手，评估任务进展。", prompt)
    state.reflection = result
    print(f"  反思: {result[:200]}")

    # 判断下一个节点
    if "状态: 完成" in result or "状态：完成" in result:
        return "finish"
    elif "需要更多工具" in result:
        return "tool"
    else:
        return "plan"


def finish_node(state: AgentState) -> str:
    """结束节点：生成最终答案"""
    print("\n✅ [Finish 节点] 生成最终答案...")

    prompt = (
        f"目标: {state.goal}\n"
        f"计划: {state.plan}\n"
        f"工具结果: {state.tool_results}\n"
        f"反思: {state.reflection}\n\n"
        f"请给出最终答案。"
    )

    result = call_llm("根据所有信息给出最终答案。", prompt)
    state.answer = result
    print(f"  答案: {result[:300]}")

    return "END"  # 结束标记


# ============ 状态机图定义 ============
# 节点注册表：节点名 → 节点函数
NODES = {
    "plan": plan_node,
    "tool": tool_node,
    "reflect": reflect_node,
    "finish": finish_node,
}

# 状态转移图（由各节点函数的返回值动态决定）
# 合法转移：
#   plan    → tool / finish
#   tool    → reflect
#   reflect → finish / tool / plan
#   finish  → END
VALID_TRANSITIONS = {
    "plan": {"tool", "finish"},
    "tool": {"reflect"},
    "reflect": {"finish", "tool", "plan"},
    "finish": {"END"},
}


def run_graph_agent(goal: str, max_iterations: int = 6) -> str:
    """运行基于状态机的 Graph Agent"""
    state = AgentState(goal)

    print(f"\n{'='*60}")
    print(f"目标: {goal}")
    print(f"状态图: plan → tool → reflect → finish")
    print(f"{'='*60}")

    current = "plan"  # 起始节点

    while current != "END" and state.iterations < max_iterations:
        state.iterations += 1
        print(f"\n--- 迭代 {state.iterations} | 当前节点: {current} ---")

        # 执行当前节点
        node_fn = NODES[current]
        next_node = node_fn(state)

        # 验证转移合法性
        if next_node not in VALID_TRANSITIONS.get(current, set()):
            print(f"  ⚠️ 非法转移: {current} → {next_node}，强制进入 finish")
            next_node = "finish"

        print(f"  转移: {current} → {next_node}")
        current = next_node

    if state.iterations >= max_iterations:
        print(f"\n⚠️ 达到最大迭代次数 ({max_iterations})")

    print(f"\n{'='*60}")
    print(f"最终答案:\n{state.answer}")
    print(f"{'='*60}")

    return state.answer


def main():
    print("=== Graph Agent（状态机 Workflow） ===")
    print("节点: plan → tool → reflect → finish")
    print("输入 'quit' 退出\n")

    while True:
        goal = input("目标: ").strip()
        if goal.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not goal:
            continue
        run_graph_agent(goal)
        print()


if __name__ == "__main__":
    main()
