# Agent 教程学习路线 + 示例程序生成要求

| 阶段 | 学习目标 | 教程示例程序 | 要求 |
|---|---|---|---|
| 1. LLM 基础 | 理解 LLM API 调用和消息结构 | Python 调用 LLM API，实现简单问答 | 写一个 Python 示例程序，使用 LLM API 实现一个简单的聊天机器人。程序需要包含：1）API调用；2）用户输入循环；3）打印模型回答；代码不超过80行，并加详细注释。 |
| 2. Prompt 控制 | 理解 Prompt Engineering | 实现一个“数学助手”或“代码解释助手” | 写一个 Python 程序，通过 prompt engineering 构建一个数学助手。程序接收用户输入的数学问题，使用 prompt 指导模型一步一步推理并给出答案。代码需包含 prompt 模板和调用逻辑。 |
| 3. ReAct Agent | 理解 Agent 基本思想（Reason + Act） | 实现一个能调用计算器的 Agent | 写一个 Python 示例，实现一个最简 ReAct Agent。Agent可以使用一个工具：calculator(expression)。Agent流程为 Thought → Action → Observation → Answer。使用循环让模型决定是否调用工具。代码需要清晰展示 Agent 推理过程。 |
| 4. Tool Agent | 学习工具系统设计 | 实现多个工具（搜索、计算器） | 写一个 Python Agent 示例，实现一个 Tool Agent。定义一个 Tool 类（包含 name、description、function）。实现两个工具：calculator 和 search。Agent根据模型输出决定调用哪个工具，并返回结果。代码需要结构清晰、适合作为教学示例。 |
| 5. Memory Agent | 学习 Agent 记忆机制 | 实现带短期记忆的 Agent | 写一个 Python 示例，实现一个带 conversation memory 的 Agent。Agent需要保存对话历史，并在每次调用模型时将历史作为 context 发送。代码需要展示 memory 数据结构以及如何更新 memory。 |
| 6. Planning Agent | 学习任务分解 | Agent自动生成任务计划并执行 | 写一个 Python 示例程序，实现一个 Planning Agent。Agent先根据用户目标生成一个 step-by-step plan，然后逐步执行每个步骤。执行过程中可以调用工具。代码需要展示 plan 生成和 plan 执行逻辑。 |
| 7. Multi-Agent | 学习多 Agent 协作 | Planner + Executor + Critic | 写一个 Python 示例，实现一个 Multi-Agent 系统。包含三个 Agent：Planner（制定计划）、Executor（执行任务）、Critic（检查结果）。系统需要展示三个 Agent 之间的交互流程。代码需结构清晰、适合教学。 |
| 8. Graph Agent | 理解 Agent Workflow | 状态机式 Agent | 写一个 Python 示例，实现一个基于状态机的 Agent workflow。定义多个节点（plan、tool、reflect、finish），使用一个 graph 或字典表示状态转移，并循环执行直到 finish。代码需要展示 Agent workflow 的实现方式。 |


# 运行方式
```
conda activate agent_tutorial
cd stage1_llm_basic && python main.py
```