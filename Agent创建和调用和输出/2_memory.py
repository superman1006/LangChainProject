import os
import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware
from langgraph.types import Command
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # ⭐ 异步版本
from langgraph.checkpoint.memory import InMemorySaver
from utils import utils
# 从 .env 文件加载 API 密钥
load_dotenv()

SYSTEM_PROMPT = "你是一个智能助手，根据用户的问题调用合适的工具来获取信息。"
# 从 .env 读取数据库连接串
# 例：DB_URI=postgresql://chenlv@localhost:5432/postgres?sslmode=disable
DB_URI = os.getenv("DB_URI")

async def main():
    model = ChatOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        model=os.getenv("MODEL_NAME"),
        extra_body={"thinking": {"type": "disabled"}}
    )

    """会话记录保存在内存中,重启后上下文丢失"""
    async def inmemory_saver():
        @tool
        def get_info(query: str) -> str:
            """查询用户信息"""
            return f"用户叫陈律"

        agent = create_agent(
            model=model,
            tools=[get_info],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),   # ⭐ 目前先把聊天数据放在内存中
        )

        while True:
            response = agent.invoke(
                input = {"messages": [HumanMessage(input("input: "))]}, # 当前用户输入
                config = {"configurable": {"thread_id": "1"}},  # 包含 thread_id 对应的会话的以前的所有信息,可以理解为上下文记忆
            )
            print(response["messages"][-1].content)  # 输出模型的最后一条消息


    """会话记录保存在PostgreSQL数据库中，支持上下文恢复"""
    async def postgres_saver():
        @tool
        def get_info(query: str) -> str:
            """查询用户信息"""
            return f"用户叫陈律"


        # ⭐ 异步版本：用 async with 和 AsyncPostgresSaver
        async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
            await checkpointer.setup()   # ⭐ 异步建表，要 await

            agent = create_agent(
                model=model,
                tools=[get_info],
                checkpointer=checkpointer,  # 把 postgre 数据库保存器传给 Agent
            )

            await utils.print_session_messages(agent,"user_001")


            # 开始循环对话
            while True:
                user_input = input("input: ")
                if user_input == "exit":
                    break
                response = await agent.ainvoke(
                    {"messages": [HumanMessage(user_input)]},
                    config={"configurable": {"thread_id": "user_001"}}
                )
                print("agent:",response["messages"][-1].content)

    await postgres_saver()
if __name__ == "__main__":
    asyncio.run(main())