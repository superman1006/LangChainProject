import os
import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from utils import utils

# 从 .env 文件加载 API 密钥
load_dotenv()
SYSTEM_PROMPT = "你是一个智能助手，根据用户的问题调用合适的工具来获取信息。"
DB_URI = os.getenv("DB_URI")


async def main():
    model = ChatOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        model=os.getenv("MODEL_NAME"),
        extra_body={"thinking": {"type": "disabled"}}
    )

    @tool
    def get_info(query: str) -> str:
        """查询用户信息"""
        return f"用户叫陈律"

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        agent = create_agent(
            model=model,
            tools=[get_info],
            system_prompt=SYSTEM_PROMPT,
            middleware=[
                SummarizationMiddleware(
                    model=model,
                    trigger=("tokens", 100),
                    keep=("messages", 20),
                ),
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "get_info": {
                            "allowed_decisions": ["approve", "reject"]
                        },
                    }
                ),
            ],
            checkpointer=checkpointer,   # ⭐ 必须，保存中断状态
        )

        await utils.print_session_messages(agent,"user_001")

        # ⭐ 必须，标识同一段对话
        config = {"configurable": {"thread_id": "use1r_001"}}

        while True:
            user_input = input("input: ")
            if user_input == "exit":
                break

            # 第一次调用 Agent
            response = await agent.ainvoke(
                {"messages": [HumanMessage(user_input)]},
                config=config
            )

            # ⭐ 循环处理中断（可能连续多次工具调用都需要批准）
            while response.get("__interrupt__"):
                # response = {
                #     "messages": [HumanMessage(...), AIMessage(tool_calls=[...]), ...],
                #     "__interrupt__": [
                #         Interrupt(value={...}, ...)
                #     ]
                # }
                interrupt_value = response["__interrupt__"][0].value
                action_requests = interrupt_value["action_requests"]
                print("action_requests:", action_requests)

                decisions = []
                for action in action_requests:
                    print(f"\n⛔ Agent 想调用工具:")
                    print(f"   工具名: {action['name']}")
                    print(f"   参数: {action['args']}")
                    choice = input("批准吗？(approve/reject): ").strip().lower()

                    if choice == "approve":
                        decisions.append({"type": "approve"})
                    else:
                        decisions.append({"type": "reject", "message": "用户拒绝了此操作"})

                # 用 Command(resume=...) 恢复 Agent
                response = await agent.ainvoke(
                    Command(resume={"decisions": decisions}),
                    config=config   # 同一个 thread_id
                )

            # 观察当前对话状态
            print("\n" + "=" * 50)
            print(f"📊 当前消息总数: {len(response['messages'])}")
            for i, msg in enumerate(response["messages"]):
                marker = "📌" if msg.type == "system" and ("summary" in (msg.content or "").lower() or "总结" in (msg.content or "")) else "  "
                preview = (msg.content or "")[:30].replace("\n", " ")
                print(f"  {marker} [{i}] {msg.type}: {preview}...")
            print("=" * 50)

            print(f"\n助手回复：{response['messages'][-1].content}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
