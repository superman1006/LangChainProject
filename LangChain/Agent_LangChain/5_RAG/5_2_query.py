import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings   # ⭐ 本地 embedding
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader,TextLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter


os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
load_dotenv()

# ⭐ 跟 build 脚本用同样的绝对路径
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")


# 1. 本地 embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 3. 拿到 chroma 数据库 的 向量库
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,         # ⭐ 同一个绝对路径
    collection_name="table_001",          # ⭐ 同一个 collection 名
    embedding_function=embeddings,
)

# ⭐ 调试：确认库里有数据
print(f"📊 当前库中数据条数: {vectorstore._collection.count()}")

# 4. 提问 + 检索top-k相关内容
question = "陈律爱吃什么饭，什么口味？"
retrieved = vectorstore.similarity_search(question, k= 2)
print("检索到的相关内容：")
for chunk in retrieved:
    print(f"==================================================="
          f"\n[{chunk.metadata.get("source")}] : \n{chunk.page_content.replace("\n", " ")}")

# 5. 拼 prompt 调模型
context = "\n".join([d.page_content for d in retrieved])
prompt = f"""根据以下资料回答问题：资料：{context} ,问题：{question}"""

model = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    extra_body={"thinking": {"type": "disabled"}}
)
response = model.invoke([HumanMessage(prompt)])
print(f"\n回答：{response.content.replace("\n", " ")}")