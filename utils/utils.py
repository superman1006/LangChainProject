

async def print_session_messages(agent, thread_id:str):
    """获取PostgresSQL 中指定 thread_id 的会话记录并打印"""
    # ⭐ agent 的 .get_state方法从数据库中读取之前的对话状态，恢复上下文
    print("正在恢复之前的对话状态...")
    state = await agent.aget_state(config={"configurable": {"thread_id": thread_id}})
    for msg in state.values.get("messages", []):
        print(f"[{msg.type}] {msg.content[:80].replace('\n', '')}")
    print("\n")