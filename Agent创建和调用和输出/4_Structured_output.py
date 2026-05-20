import os
import asyncio
from pyexpat.errors import messages

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from pydantic import BaseModel,Field
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware
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

    # 用pydantic 的 BaseModel 创建一个结构化类，用于给model/agent 参考输出格式
    class UserInfo(BaseModel):
        # # 加描述（模型看了能更准确地填）
        name: str = Field(description="用户的名字")
        age: int = Field(description="用户的年龄")
        email: str = Field(description="用户的邮箱")

    async def structured_output_in_model():
        """把结构化输出类绑定在 model 上"""
        # ⭐ 明确指定 method，并加 system prompt 强调要输出 JSON
        structured_model = model.with_structured_output(
            UserInfo,
            method="function_calling",   # 或试试 "json_mode"
        )
        result: UserInfo = structured_model.invoke([
            SystemMessage("你是一个信息提取助手。从用户输入中提取信息，按指定的 JSON 格式返回，不要返回任何额外的文字或解释。"),
            HumanMessage("我叫陈律，25 岁，邮箱 chen@x.com")
        ])

        print(result)
        print(result.name)    # "陈律"
        print(result.age)     # 25 （直接是 int，不是字符串）
        print(result.email)   # "chen@x.com"

    async def structured_output_in_agent():
        """把结构化输出类绑定在 agent 上，agent 内部调用工具处理完后最终返回结构化数据"""

        agent = create_agent(
            model = model,
            tools = [],
            response_format=UserInfo
        )

        response = await agent.ainvoke({
            "messages":[
                SystemMessage("你是一个信息提取助手。从用户输入中提取信息，按指定的 JSON 格式返回，不要返回任何额外的文字或解释。"),
                HumanMessage("我叫陈律，25 岁，邮箱chenlv@x.com")
            ]
        })
        for message in response["messages"]:
            if message.type == "ai" and message.tool_calls:
                print(message.tool_calls)
            print(f"[{message.type}]: {message.content}")   # agent 内部的消息流，包含模型回答、工具调用和工具结果等
        print("="*20)
        structured_result = response["structured_response"]
        print(structured_result)
        print(structured_result.name)    # "陈律"
        print(structured_result.age)     # 25 （直接是 int，不是字符串）
        print(structured_result.email)   # "chen@x.com"

    await structured_output_in_agent()


if __name__ == "__main__":
    asyncio.run(main())
