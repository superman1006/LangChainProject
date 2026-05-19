import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"  # 绕过 VPN 代理，直连本地

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
load_dotenv(verbose=True)

async def main():
    client = MultiServerMCPClient({
        "本地 MCP服务器": {
            "transport": "streamable_http",
            "url": "http://localhost:1234/mcp"
        }
    })

    tools = await client.get_tools()
    print("展示 MCP服务器中的所有tools: ")
    for tool in tools:
        print("\t",tool)


    model = ChatOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),  # 从环境变量读取
        model=os.getenv("MODEL_NAME"),
        extra_body={"thinking": {"type": "disabled"}}
    )
    agent = create_agent(
        model = model,
        tools = tools,
        # 系统提示词
        system_prompt="你是一个智能助手，可以调用{tools}中的工具来帮助用户完成任务"
    )

    # MCP 工具只支持异步，所以要用 ainvoke 来调用
    response = await agent.ainvoke({
        "messages": [HumanMessage("请帮我调用 personal_Info 工具，告诉我用户的个人信息")],
    })
    # SystemMessage(content="你是一个智能助手，可以调用{tools}中的工具来帮助用户完成任务")    在外面的系统提示词已经设置了
    # response = {
    #       HumanMessage("请帮我调用 personal_Info 工具，告诉我用户的个人信息"),
    #       AIMessage(content="好的，我来调用 personal_Info 工具", tool_calls=[xxx]),
    #       ToolMessage(name="personal_Info", content="xxx"),
    #       ......
    # }

    # 输出所有消息内容和工具调用信息
    for message in response["messages"]:
        print(f"\n[{message.type}] say: '{message.content}'")
        # 如果是 AI 消息且包含工具调用，打印工具调用信息
        if message.type == "ai" and message.tool_calls:
            print(f"  工具调用: {message.tool_calls}")
        # 如果是工具结果，打印工具名
        if message.type == "tool":
            print(f"  工具名: {message.name}")


if __name__ == "__main__":
    asyncio.run(main())
