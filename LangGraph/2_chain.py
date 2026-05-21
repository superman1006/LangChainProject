import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),  # 从环境变量读取
    model=os.getenv("MODEL_NAME"),
    extra_body={"thinking": {"type": "disabled"}}
)


@tool
def get_info(query: str) -> str:
    """可以获得用户信息"""
    return f"搜索结果: {query}的名字是陈律，年龄 23岁"


# 1. State
class AgentState(TypedDict):
    # Annotated 可以给字段添加元信息，这里我们用它来标记这个字段需要被 add_messages 处理
    messages: Annotated[list[BaseMessage], add_messages]
    # add_messages表示允许这个字段被 add_messages 处理，自动把所有 unode 返回的 messages 添加到 当前 AgentState messages 里


# 2. 节点
def call_model(state: AgentState):
    model_with_tools = model.bind_tools([get_info])
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# 3. 条件路由
def should_continue(state: AgentState):
    """决定下一步去哪"""
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"   # 有工具调用 → 去 tools 节点
    return END           # 没有 → 结束


# 4. 构建图
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode([get_info]))   # ⭐ LangGraph 内置的工具执行节点

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)   # ⭐ 条件边 可以指向多个不同的节点,取决于传入的函数返回值
graph.add_edge("tools", "agent")   # 工具调完回到 agent 再判断

app = graph.compile()

resposne = app.invoke({"messages": [HumanMessage("调用工具拿到我的信息")]})
for res in resposne["messages"] :
    print(res)
