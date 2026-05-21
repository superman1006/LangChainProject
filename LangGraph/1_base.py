from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. 定义 AgentState（这次对话的"共享数据"）
class AgentState(TypedDict):
    user_name: str   # 用户名
    count: int       # 计数器


# 2. 定义节点（一个节点 = 一个函数）
def node_hello(state: AgentState) -> dict:
    """节点 1：打招呼"""
    print(f"👋 hello, {state['user_name']}!")
    return {"count": state["count"] + 1}   # ⭐ 返回的字典会被合并到 state


def node_bye(state: AgentState) -> dict:
    """节点 2：告别"""
    print(f"👋 bye, {state['user_name']}, 你被处理了 {state['count']} 次")
    return {"count": state["count"] + 1}


'''
目标:
    START → node_hello → node_bye → END
'''

# 3. 构建图
graph = StateGraph(AgentState)

# 加节点
graph.add_node("你好节点", node_hello)
graph.add_node("再见节点", node_bye)

# 加边（指定流向）
graph.add_edge(START, "你好节点")        # 起点 → greet
graph.add_edge("你好节点", "再见节点")   # greet → farewell
graph.add_edge("再见节点", END)       # farewell → 终点

# 4. 编译
app = graph.compile()
# 把这段复制到 https://mermaid.live 能看到可视化的图。
# 在 app.compile() 后加这一段
print(app.get_graph().draw_mermaid())


# 5. 调用 invoke传入的参数就是 AgentState 类型
result = app.invoke({"user_name": "陈律", "count": 0})

print("\n最终 state:", result)