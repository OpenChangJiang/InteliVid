import os

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import Milvus
from dotenv import load_dotenv
load_dotenv()

from pymilvus import MilvusClient

milvus_uri = os.getenv('MILVUS_URI')
milvus_user = os.getenv('MILVUS_USER')
milvus_password = os.getenv('MILVUS_PASSWORD')

client = MilvusClient(
    uri=milvus_uri,
    user=milvus_user,
    password=milvus_password
)

# 清空Collection，避免其他数据干扰
client.drop_collection(collection_name="LangChainCollection")

# 初始化ZhipuAI Embedding
embeddings = ZhipuAIEmbeddings()

# 加载字幕文档
loader = DirectoryLoader('./output/captions/basic', glob='**/*.txt')
documents = loader.load()

# 创建向量存储
vectorstore = Milvus.from_documents(
    documents, 
    embeddings,
    connection_args={"uri": milvus_uri, "user": milvus_user, "password": milvus_password}
)

def retrieve(query, k=5):
    """检索与查询最相关的文档"""
    docs = vectorstore.similarity_search(query, k=k)
    return docs


def main(query: str):
    results = retrieve(query)
    return results


if __name__ == "__main__":
    # Test the search functionality
    test_query = "technology demonstration"
    print("Search results:", main(test_query))
