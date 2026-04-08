import copy
import os
import json
import re
from turtle import st

import ipdb
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal, List
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

    next_step: str  # 路由决策结果，表示当前任务的类型，比如是讲规划还是直接执行

    current_task: int  # 当前正在执行的任务

    todo_list: List[str]  # 任务列表，记录当前有哪些任务需要完成
    child: dict  # 记录当前任务的子任务列表，key是父任务的id，value是子任务列表
    parent: dict  # 记录当前任务的父任务，key是子任务的id，value是父任务id


class Plan(BaseModel):
    steps: List[str] = Field(
        None,
        description="子任务列表，给出纯文本, 不要任何格式化的输出, 不要加数字编号",
    )

    # for prettry print
    def __repr__(self):
        return f"Plan(steps={self.steps})"

    def __str__(self):
        return f"Plan:\nSteps:\n" + "\n".join(self.steps)


class Route(BaseModel):
    step: Literal["plan", "execute"] = Field(
        description="Choose 'plan' if the task needs decomposition, "
        "choose 'execute' if the task can be directly executed."
    )


# model = init_chat_model(
#     "qwen3.5-27b",
#     # "qwen3.5-0.8b",
#     temperature=0,
#     model_provider="openai",
#     base_url="http://10.0.0.114/v1",
#     api_key=os.getenv("GPUSTACK_API_KEY"),
# )
# from langchain_qwq import ChatQwen

# model = ChatQwen(
#     model="qwen3.5-27b",
#     temperature=0.3,
#     base_url="http://10.0.0.114/v1",
#     api_key=os.getenv("GPUSTACK_API_KEY"),
# )

from langchain_qwq import ChatQwen

model = ChatQwen(
    model="qwen3.5-27b",
    temperature=0.3,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("ALIYUN_API_KEY"),
)

planing_model = model.with_structured_output(Plan)
route_model = model.with_structured_output(Route)


def planer(state: dict):
    """LLM decides whether to call a tool or not"""
    current_task = state.get("current_task", 0)
    todo_list = copy.deepcopy(
        state.get("todo_list", [state["messages"][-1].content])
    )  # 如果没有todo_list，就把用户的输入作为第一个任务

    current_task_ = current_task
    current_task_content = todo_list[current_task_]
    plan = planing_model.invoke(
        [
            SystemMessage(
                content=f"""你是一个智能助手,需要基于用户的输入和当前任务做出任务分解，列出子任务列表.
            """
            )
        ]
        + [
            HumanMessage(
                content=f"""用户要求为: {todo_list[0]}
                当前任务为: {current_task_content}
                """
            )
        ]
    )

    # 解析模型输出的子任务列表，更新state中的todo_list和task_tree
    child = copy.deepcopy(state.get("child", {}))
    parent = copy.deepcopy(state.get("parent", {}))
    for i, step in enumerate(plan.steps):
        todo_list.append(step)  # 将子任务加入待办列表
        sub_task_id = len(todo_list) - 1  # 子任务的id为当前todo_list的长度
        child.setdefault(current_task, []).append(sub_task_id)
        parent[sub_task_id] = current_task  # 记录子任务的父任务

    print("规划结果:")

    def print_subtree(task_id, indent=0):
        print(" " * indent + f"- {todo_list[task_id]}")
        for child_id in child.get(task_id, []):
            print_subtree(child_id, indent + 2)

    print_subtree(0)  # 从根任务开始打印

    # 当前任务指向子任务，路由器会根据这个信息决定下一步是进入planer还是executor
    next_task_id = child[current_task][0]  # 进入第一个子任务

    return {
        "todo_list": todo_list,
        "child": child,
        "parent": parent,
        "current_task": next_task_id,
    }


def router(state: dict):
    """
    当前的任务
    """
    current_task = state.get("current_task", 0)
    todo_list = copy.deepcopy(
        state.get("todo_list", [state["messages"][-1].content])
    )  # 如果没有todo_list，就把用户的输入作为第一个任务
    if current_task == -1:  # 没有任务了，说明所有任务都完成了
        return {"next_step": "end"}
    parent = state.get("parent", {})

    # 构造调用堆栈信息，告诉模型当前任务在整个任务树中的位置
    current_task_ = current_task
    current_task_content = todo_list[current_task_]
    # while parent.get(current_task_, 0) != 0:  # 0是根任务
    #     parent_id = parent[current_task_]
    #     current_task_content = f"{todo_list[parent_id]} -> {current_task_content}"
    #     current_task_ = parent_id
    # print(f"当前任务: {current_task_content}")
    route = route_model.invoke(
        [
            SystemMessage(
                content=f"""一般步骤复杂或者过于笼统的任务需要拆分。
                你需要判断所给任务是否需要拆分。
                """
            ),
        ]
        + [
            HumanMessage(
                content=f"""用户要求为: {todo_list[0]}
                当前任务为: {current_task_content}
                """
            )
        ]
    )
    # ipdb.set_trace()
    if route.step == "plan":
        next_step = "planer"
    elif route.step == "execute":
        next_step = "executor"

    return {"next_step": next_step}


def route_fn(state: dict):
    # Return the node name you want to visit next
    return state["next_step"]


def excutor(state: dict):
    """Excutor performs the current task and updates the state"""
    current_task = state.get("current_task", 0)
    todo_list = state.get("todo_list", [])

    current_task_content = todo_list[current_task]
    print(f"正在执行任务: {current_task_content}")

    # 模拟执行任务，这里直接把任务内容作为结果返回
    result = f"完成了任务: {current_task_content}"

    # 更新state，标记当前任务完成，进入下一个任务

    # 进入下一个兄弟任务；如果没有兄弟任务，回到父任务的下一个兄弟任务，以此类推
    parent = state.get("parent", {})
    child = state.get("child", {})
    next_task_id = -1  # 默认没有下一个任务了
    # 进入下一个兄弟任务
    while current_task in parent:
        parent_id = parent[current_task]
        siblings = child[parent_id]
        current_index = siblings.index(current_task)
        if current_index + 1 < len(siblings):  # 如果有下一个兄弟任务
            next_task_id = siblings[current_index + 1]
            break
        else:  # 没有下一个兄弟任务，继续往上找
            current_task = parent_id
    # 如果找到了下一个任务，next_task_id会被更新；如果没有找到，说明所有任务都完成了，next_task_id保持为None
    return {
        "messages": [AIMessage(content=result)],
        "current_task": next_task_id,
    }


# ── Build workflow ────────────────────────────────────────────────────────────

agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("planer", planer)
agent_builder.add_node("router", router)
agent_builder.add_node("executor", excutor)

# Add edges to connect nodes
agent_builder.add_edge(START, "router")
agent_builder.add_edge("planer", "router")
agent_builder.add_edge("executor", "router")
agent_builder.add_conditional_edges(
    "router", route_fn, {"planer": "planer", "executor": "executor", "end": END}
)  # Name returned by route_decision : Name of next node to visit

# Compile the agent
agent = agent_builder.compile()

# Show the agent graph
print(agent.get_graph(xray=True).draw_ascii())
try:
    from IPython.display import Image, display

    display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    print(agent.get_graph(xray=True).draw_ascii())

# ── Invoke ───────────────────────────────────────────────────────────────────
# messages = [HumanMessage(content="请回复:你好")]

messages = [
    HumanMessage(content="帮我写一篇关于火星的科普文章，要求内容丰富且有趣")
]  # 这个问题比较复杂，模型需要分步思考，体现了规划能力
result = agent.invoke({"messages": messages})

ipdb.set_trace()
for m in result["messages"]:
    m.pretty_print()
