import asyncio
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings   # ⭐ 本地 embedding
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader,TextLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

#==============================全局创建=====================================
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    collection_name="table_001",
    embedding_function=embeddings,
)
# 检索策略 ⭐
retriever = vectorstore.as_retriever(
    search_type="mmr",  #MMR（最大边际相关）一种检索策略 —— 去除重复内容,除了这个还有 similarity_score_threshold 等
    search_kwargs={"k": 3, "fetch_k": 10}
)
# 也可以使用下面这种混合检索策略，BM25可以精确匹配关键词,默认的向量检索可以匹配语义，二者通过EnsembleRetriever结合可以提升检索效果
'''
# 第一种检索器：BM25
bm25_retriever = BM25Retriever.from_documents(chroma 中的所有chunks数据)
bm25.k = 3

# 第二种检索器：向量检索
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 混合检索
ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]   # 关键词权重 40%，语义权重 60%
)

results = ensemble.invoke("陈律爱吃什么？")
'''



async def main():
    @tool
    def query_chroma(query:str) -> str:
        """一个 RAG数据库查询工具，输入一个问题，返回相关的内容"""
        # 1. 本地 embedding 模型
        print("query: ",query)
        retrieved = retriever.invoke(query)

        for d in retrieved:
            print(f"==================================================="
                  f"\n[{d.metadata.get('source')}] : \n{d.page_content.replace('\n', ' ')}")
        return "\n".join([d.page_content for d in retrieved])


    model = ChatOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),  # 从环境变量读取
        model=os.getenv("MODEL_NAME"),
        extra_body={"thinking": {"type": "disabled"}}
    )
    agent = create_agent(
        model=model,
        tools=[query_chroma],
        system_prompt="你是一个智能助手，根据用户的问题调用合适的工具来获取信息。",
    )

    response = await agent.ainvoke({
        "messages":[HumanMessage(content="陈律爱吃什么饭，什么口味？")]
    })

    print("agent:\n",response["messages"][-1].content.replace("\n", " "))

if __name__ == "__main__":
    asyncio.run(main())
