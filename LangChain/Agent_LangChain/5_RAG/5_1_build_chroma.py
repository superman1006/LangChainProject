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

# ⭐ 用绝对路径，确保 build 和 query 用同一个目录
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# 1. 加载 所有文件,返回一个 Document 对象列表
PDF_doc = PyPDFLoader("//doc/陈律的简历.pdf").load()
MD_doc = TextLoader("//doc/Project_technology.md").load()
docs = PDF_doc + MD_doc


# 2.1 定义文本切分器
# 除了 RecursiveCharacterTextSplitter,还有MarkdownHeaderTextSplitter, CharacterTextSplitter, TokenTextSplitter 等多种切分器，适用于不同的文本结构和需求
splitter = RecursiveCharacterTextSplitter(
    chunk_size= 100, # 每个 chunk 的最大长度，超过会继续切分
    chunk_overlap=10, # chunk 之间的重叠长度，保证上下文连续性
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],   # 切分顺序
    length_function=len,   # 怎么算长度，默认按字符
)

# 2.2 开始切割
chunks = splitter.split_documents(docs)
print(f"✂️ 切成 {len(chunks)} 块")



# 3. 本地 embedding（第一次跑会下载模型）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)



# 4. 向量存储进 chroma 数据库， 返回的vectorstore是一个内存中的向量数据库，适合小规模数据的检索
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,   # ⭐ 用绝对路径
    collection_name="table_001"
)

print(f"✅ 索引已保存到 ./chroma_db，共 {vectorstore._collection.count()} 条")
