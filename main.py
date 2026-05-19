import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import asyncio
from langchain.agents import create_agent

# 从 .env 文件加载 API 密钥（不要硬编码！）
load_dotenv()
SYSTEM_PROMPT = """
"你是一个智能助手，根据用户的问题调用合适的工具来获取信息。"
"""


async def main():
    model = ChatOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),  # 从环境变量读取
        model=os.getenv("MODEL_NAME"),
        extra_body={"thinking": {"type": "disabled"}}
    )

    def chat_base():
        try:
            response = model.invoke(" 你好 1")
            print(response.content)
            print("----")
            for chunk in model.stream("你好 2"):
                # llm返回的内容在每一个 chunk的 content 中
                print(chunk.content, end="\n")
                # llm返回的内容块在每一个 chunk的 content_blocks 中
                print(chunk.content_blocks, end="\n")
                print("\n")
        except Exception as e:
            print(f"错误：{e}")

    def chat_with_local_tool():

        # 假的
        @tool(description="websearch")
        def websearch(query: str) -> str:
            # 这里是一个模拟的网络搜索工具，实际应用中你可以调用真实的搜索 API
            return f"搜索结果：{query} 的相关信息"

        try:
            # 把工具绑定到模型上，这样模型就可以调用它了
            model_with_tool = model.bind_tools([websearch])
            response = model_with_tool.invoke("请使用 websearch 工具搜索 '人工智能' 的相关信息")
            print("模型响应：", response)
            for tool_call in response.tool_calls:
                print("工具调用：", tool_call)
        except Exception as e:
            print(f"错误：{e}")

    def chat_with_tool_loop():
        @tool(description="tool1")
        def tool1(query: str) -> str:
            # 这里是一个模拟的网络搜索工具，实际应用中你可以调用真实的搜索 API
            return f"直接返回给用户 '啦啦啦啦啦啦'"

        # 自定义消息列表，包含用户输入
        messages: list[BaseMessage] = [HumanMessage(content="调用给你配置的 tool1 工具，传入参数是'哈哈哈'，并拿回结果")]
        # 绑定工具, tool_choice="tool1" 表示模型在调用工具时会优先选择 tool1
        model_with_tool = model.bind_tools([tool1], tool_choice="tool1")

        # 调用模型，模型传回来的的消息会包含 tool_calls=[{'name': 'tool1', 'args': {'query': 'lalala'}
        msg_contains_tool = model_with_tool.invoke(messages)
        print("模型返回的包含了 tool_call列表的信息 : ", msg_contains_tool.tool_calls)

        # 把模型返回的消息添加到消息列表中，模型返回的消息中包含了 tool_calls 列表，模型会根据 tool_calls 列表中的工具调用信息来调用工具，并把工具的结果添加到消息列表中
        messages.append(msg_contains_tool)
        print("把刚刚包含 tool_call列表的信息加入到初始 messages: ", messages)

        for tool_call in msg_contains_tool.tool_calls:
            print("tool_call: ", tool_call)
            tool_result = tool1.invoke(tool_call)
            print("tool_result: ", tool_result)
            messages.append(SystemMessage(content=tool_result.content))
        final_message = model_with_tool.invoke(messages)
        print("final message: ", final_message)

    # noinspection PyTypeChecker
    async def react_chat():
        #  因为百度搜索工具本质还 MCP工具(网址中有/mcp)，所以注册为 MCP工具
        #  而 Google 搜索给的是REST API(网址中有/search或/api/v1)，所以注册为本地工具


        from langchain_mcp_adapters.client import MultiServerMCPClient
        # 1.创建 MCP客户端，目前只绑定一个百度搜索
        client = MultiServerMCPClient({
            "baidu_search": {
                "transport": "sse",  # ⚠️ 用 sse，不是 http
                "url": f"{os.getenv('BAIDU_SEARCH_BASE_URL')}={os.getenv('BAIDU_SEARCH_API_KEY')}",
            }
        })
        mcp_tools = await client.get_tools()
        print("展示 MCP服务器中的所有tools: ")
        for mcp_tool in mcp_tools:
            print("\t",mcp_tool)



        # 2.创建本地 tool 工具,注意不是 MCP!!!!!
        @tool()
        def google_search(query: str) -> str:
            """使用 Google 搜索查询信息。输入搜索关键词，返回搜索结果"""
            # 上面的注释会被自动添加在@tool 中的 description 字段中
            import serpapi

            serpapi_client = serpapi.Client(api_key=os.getenv("GOOGLE_SEARCH_API_KEY"))
            results = serpapi_client.search({
                "engine": "google",
                "q": query,
                "hl": "zh-cn",
                "gl": "cn"
            })
            # 提取前几条结果
            organic = results.get("organic_results", [])[:3]
            return "\n\n".join([
                f"标题：{r.get('title')}\n摘要：{r.get('snippet')}\n链接：{r.get('link')}"
                for r in organic
            ])


        # 3.把 MCP工具和本地工具合并成一个工具列表，传给 agent
        all_tools = mcp_tools + [google_search]
        print("所有工具: ")
        for tool_ in all_tools:
            print("\t", tool_)
        print("="*50)

        agent = create_agent(
            model=model,
            tools=all_tools,
            system_prompt=SYSTEM_PROMPT
        )

        response = await agent.ainvoke({
            "messages": [HumanMessage("分别用百度和谷歌搜索一下'人工智能'，对比结果")],
        })

        # 输出所有消息内容和工具调用信息
        for message in response["messages"]:
            print(f"\n[{message.type}] say: '{message.content}'")
            # 如果是 AI 消息且包含工具调用，打印工具调用信息
            if message.type == "ai" and message.tool_calls:
                print(f"  想要调用工具: {message.tool_calls}")
            # 如果是工具结果，打印工具名
            if message.type == "tool":
                print(f"  工具名: {message.name}:")





    await react_chat()

if __name__ == "__main__":
    asyncio.run(main())
