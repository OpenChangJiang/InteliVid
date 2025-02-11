from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Milvus
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pymilvus import MilvusClient
import os

# 加载环境变量
load_dotenv()

milvus_uri = os.getenv('MILVUS_URI')
milvus_user = os.getenv('MILVUS_USER')
milvus_password = os.getenv('MILVUS_PASSWORD')

openai_base_url = os.getenv('OPENAI_BASE_URL')
openai_api_key = os.getenv('OPENAI_API_KEY')

# 初始化Milvus客户端
client = MilvusClient(
    uri=milvus_uri,
    user=milvus_user,
    password=milvus_password
)

# 清空Collection，避免其他数据干扰
client.drop_collection(collection_name="LangChainCollection")

# 初始化DashScope Embedding
embeddings = DashScopeEmbeddings(model="text-embedding-v3")

# 加载字幕文档
loader = DirectoryLoader('./output/captions/basic', glob='**/*.txt')
documents = loader.load()

# 创建向量存储
vectorstore = Milvus.from_documents(
    documents,
    embeddings,
    connection_args={"uri": milvus_uri, "user": milvus_user, "password": milvus_password}
)

# 定义检索函数
def retrieve(query, k=5):
    """检索与查询最相关的文档"""
    docs = vectorstore.similarity_search(query, k=k)
    return docs

# 定义Prompt模板
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="根据以下上下文回答问题：\n\n{context}\n\n问题：{question}"
)

# 初始化Chat模型
llm = ChatOpenAI(
    base_url=openai_base_url,
    api_key=openai_api_key,
    model_name="deepseek-v3"
)

# 定义RAG流程
def rag_pipeline(query: str):
    # 检索相关文档
    docs = retrieve(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # 格式化Prompt
    prompt = prompt_template.format(context=context, question=query)
    
    # 生成回答
    response = llm.invoke(prompt)
    return response.content

# 主函数
def main(query: str):
    answer = rag_pipeline(query)
    return answer

if __name__ == "__main__":
    # 测试RAG功能
    test_query = "technology demonstration"
    print("RAG answer:", main(test_query))
