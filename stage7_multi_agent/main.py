"""
阶段7：Multi-Agent —— 多 Agent 协作系统

学习目标：学习多 Agent 协作
功能：
  - Planner Agent：制定计划
  - Executor Agent：执行任务
  - Critic Agent：检查结果并给出反馈
  - 展示三个 Agent 之间的交互流程
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    api_key=os.getenv("GPUSTACK_API_KEY"),
    base_url="http://10.0.0.114/v1",
)

MODEL = "qwen3.5-0.8b"


def call_llm(system_prompt: str, user_message: str, temperature: float = 0.3) -> str:
    """通用 LLM 调用函数"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=4096,
    )
    return (response.choices[0].message.content or "").strip()


# ============ Agent 定义 ============
class PlannerAgent:
    """规划 Agent：将用户目标分解为执行步骤"""

    SYSTEM_PROMPT = """你是一个 Planner Agent，专门负责将复杂任务分解为清晰的执行步骤。

请将用户的目标分解为 3-5 个具体步骤，格式如下：
1. [步骤描述]
2. [步骤描述]
...

要求：
- 步骤要具体、可执行
- 步骤之间有逻辑顺序
- 不要太笼统，每步应该清晰
"""

    def plan(self, goal: str) -> str:
        print("\n🧠 [Planner] 正在制定计划...")
        result = call_llm(self.SYSTEM_PROMPT, f"请为以下目标制定计划:\n{goal}")
        print(f"[Planner 输出]\n{result}")
        return result


class ExecutorAgent:
    """执行 Agent：根据计划逐步执行任务"""

    SYSTEM_PROMPT = """你是一个 Executor Agent，专门负责执行任务计划。

你会收到一个计划和需要执行的具体步骤。请根据你的知识和能力，执行该步骤并给出结果。

要求：
- 给出具体、详细的执行结果
- 如果某个步骤需要信息，基于你的知识给出合理的回答
- 标明哪些是确定的信息，哪些是估计的
"""

    def execute(self, plan: str, step: str) -> str:
        print(f"\n⚡ [Executor] 正在执行: {step[:60]}...")
        prompt = f"完整计划:\n{plan}\n\n请执行以下步骤:\n{step}"
        result = call_llm(self.SYSTEM_PROMPT, prompt)
        print(f"[Executor 输出]\n{result[:300]}...")
        return result


class CriticAgent:
    """评审 Agent：检查执行结果，给出反馈"""

    SYSTEM_PROMPT = """你是一个 Critic Agent，专门负责审查任务执行的结果。

你会收到原始目标、执行计划和执行结果。请审查并给出评价。

评价格式：
- 质量评分: [1-10分]
- 优点: [列举做得好的地方]
- 问题: [列举需要改进的地方]
- 建议: [给出具体改进建议]
- 是否通过: [是/否]
"""

    def review(self, goal: str, plan: str, execution_result: str) -> str:
        print("\n🔍 [Critic] 正在审查结果...")
        prompt = (
            f"原始目标:\n{goal}\n\n"
            f"执行计划:\n{plan}\n\n"
            f"执行结果:\n{execution_result}"
        )
        result = call_llm(self.SYSTEM_PROMPT, prompt)
        print(f"[Critic 输出]\n{result}")
        return result


# ============ Multi-Agent 协作系统 ============
class MultiAgentSystem:
    """多 Agent 协作系统：协调 Planner、Executor、Critic"""

    def __init__(self):
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.critic = CriticAgent()

    def run(self, goal: str, max_iterations: int = 2) -> str:
        """运行多 Agent 协作流程"""
        print(f"\n{'='*60}")
        print(f"目标: {goal}")
        print(f"{'='*60}")

        for iteration in range(max_iterations):
            print(f"\n{'='*60}")
            print(f"迭代 {iteration + 1}/{max_iterations}")
            print(f"{'='*60}")

            # 第1步：Planner 制定/改进计划
            if iteration == 0:
                plan = self.planner.plan(goal)
            else:
                # 根据 Critic 反馈改进计划
                plan = self.planner.plan(
                    f"{goal}\n\n上一次的评审反馈:\n{review}\n\n请根据反馈改进计划。"
                )

            # 第2步：Executor 执行计划
            # 从计划中提取步骤并逐步执行
            steps = [
                line.strip()
                for line in plan.split("\n")
                if line.strip() and line.strip()[0].isdigit()
            ]

            execution_results = []
            for step in steps:
                result = self.executor.execute(plan, step)
                execution_results.append(f"{step}\n结果: {result}")

            full_execution = "\n\n".join(execution_results)

            # 第3步：Critic 审查结果
            review = self.critic.review(goal, plan, full_execution)

            # 检查是否通过
            if "是否通过: 是" in review or "是否通过：是" in review:
                print(f"\n✅ Critic 审查通过！")
                break
            else:
                print(f"\n⚠️ Critic 发现需要改进，进入下一轮迭代...")

        # 输出最终结果
        print(f"\n{'='*60}")
        print("最终执行结果:")
        print(f"{'='*60}")
        print(full_execution)
        return full_execution


def main():
    print("=== Multi-Agent 协作系统 ===")
    print("包含: Planner (规划) → Executor (执行) → Critic (审查)")
    print("输入 'quit' 退出\n")

    system = MultiAgentSystem()

    while True:
        goal = input("目标: ").strip()
        if goal.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not goal:
            continue
        system.run(goal)
        print()


if __name__ == "__main__":
    main()
