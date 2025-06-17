from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
)

client = chromadb.Client()  # chroma 클라이언트 & 컬렉션
collection = client.get_or_create_collection("my_logs", embedding_function=ef)

docs = ["로그 메세지1", "로그 메세지2", "로그 메세지3"]
metas = [
    {"ts": "2025-06-12", "host": "srv1"},
    {"ts": "2025-06-12", "host": "srv2"},
    {"ts": "2025-06-12", "host": "srv3"},
]
ids = ["1", "2", "3"]
collection.add(documents=docs, metadatas=metas, ids=ids)

# 유사도 검색 (rag 컨텍스트 조회)
query = ["실제 의심 로그 메세지"]
results = collection.query(
    query_texts=query,
    n_results=3,  # 검색 결과 개수
    include=["documents", "metadatas", "distances"],  # 반환할 필드
)  # 현재 ids는 폐기되었다고 합니다 삽질 ㄴㄴ...
print("검색 결과:")
print(results["documents"], results["metadatas"], results["distances"])
