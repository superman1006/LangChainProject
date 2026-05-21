import os
import asyncio
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    extra_body={"thinking": {"type": "disabled"}}
)


# 1. State：定义共享数据
class State(TypedDict):
    topic: str
    research: str
    draft: str
    review_feedback: str
    is_approved: bool
    revision_count: int


# 2. Agent 1：研究员
def researcher(state: State) -> dict:
    """研究员：收集主题相关的资料"""
    print(f"\n🔍 [研究员] 正在研究主题: {state['topic']}")

    response = model.invoke([
        SystemMessage("你是一个资深研究员，擅长收集和整理信息。"),
        HumanMessage(f"请用 100 字介绍这个主题的关键要点：{state['topic']}")
    ])

    research = response.content
    print(f"📚 研究结果: {research[:80]}...")
    return {"research": research}  # 返回结果的时候 LangGraph 会自动合并到 state 中的相应字段


# 3. Agent 2：写作员
def writer(state: State) -> dict:
    """写作员：根据研究资料 + 审查反馈写文章"""
    revision = state.get("revision_count", 0)
    print(f"\n✍️  [写作员] 第 {revision + 1} 稿写作中...")

    # 如果有上一轮的反馈，加入 prompt
    feedback_prompt = ""
    if state.get("review_feedback"):
        feedback_prompt = f"\n\n上次的审查反馈是：{state['review_feedback']}\n请根据反馈修改。"

    response = model.invoke([
        SystemMessage("你是一个专业写作员，擅长写简洁有力的短文。"),
        HumanMessage(
            f"主题：{state['topic']}\n\n"
            f"参考资料：\n{state['research']}\n\n"
            f"请写一篇 100-150 字的短文。{feedback_prompt}"
        )
    ])

    draft = response.content
    print(f"📝 初稿: {draft[:80]}...")
    return {
        "draft": draft,
        "revision_count": revision + 1,
    }


# 4. Agent 3：审查员
def reviewer(state: State) -> dict:
    """审查员：检查质量，决定是否通过"""
    print(f"\n👀 [审查员] 正在审查...")

    response = model.invoke([
        SystemMessage("""你是一个严格的审查员。判断文章是否合格。
        要求文章满足：
        1. 字数在 100-150 之间
        2. 结构清晰
        3. 没有明显错误
        请按以下格式回复（必须严格遵守）：
        APPROVED: yes 或 no
        FEEDBACK: 反馈意见（如果 no，写出具体修改建议）"""),
        HumanMessage(f"请审查这篇文章：\n\n{state['draft']}")
    ])

    content = response.content
    print(f"💬 审查意见: {content[:100]}...")

    # 解析结果
    is_approved = "APPROVED: yes" in content.lower() or "approved: yes" in content
    feedback = content.split("FEEDBACK:")[-1].strip() if "FEEDBACK:" in content else ""

    return {
        "is_approved": is_approved,
        "review_feedback": feedback,
    }


# 5. 路由函数：审查后去哪
def route_after_review(state: State) -> str:
    if state["is_approved"]:
        print("✅ 审查通过！")
        return END

    # 防死循环：最多改 3 次
    if state.get("revision_count", 0) >= 3:
        print("⚠️  达到最大修改次数，强制结束")
        return END

    print("🔄 需要修改，回到写作员")
    return "writer"


# 6. 构建图
graph = StateGraph(State)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)

graph.add_edge(START, "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")
graph.add_conditional_edges("reviewer", route_after_review)
# 注意：没有 graph.add_edge("writer", END)，writer 完后总是去 reviewer

app = graph.compile()


# 7. 运行
async def main():
    initial_state = {
        "topic": "Python 装饰器",
        "revision_count": 0,
    }

    print("=" * 60)
    print(f"📌 主题：{initial_state['topic']}")
    print("=" * 60)

    result = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("📄 最终成稿：")
    print("=" * 60)
    print(result["draft"])
    print(f"\n📊 修改次数：{result['revision_count']}")


if __name__ == "__main__":
    asyncio.run(main())