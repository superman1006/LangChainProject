from fastmcp import FastMCP

mcp = FastMCP("local MCP server")

@mcp.tool(description="add")
def add(a:int, b:int) -> int:
    """这是一个加法工具，返回两个数的和"""
    return a+b

@mcp.tool(description="minus")
def minus(a:int, b:int) -> int:
    """这是一个 减法工具，返回两个数的差"""
    return  a-b

@mcp.tool(description="personal_Info")
def personal_Info() -> str:
    return "用户是陈律，来自华南师范大学"

if __name__ == "__main__":
    #FastMCP 中的传输协议 HTTP其实是 Streamable HTTP，所以外部客户端使用的时候写Streamable HTTP
    # FastMCP往外暴露的是 http://localhost:1234/mcp  记得后面要有一个/mcp, 否则请求根目录是没有东西的
    mcp.run(transport="http",port=1234)