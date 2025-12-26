# retrieval/bm25.py

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def build_bm25_retriever(docs: list[Document], k: int = 20):
    """
    构建 BM25 检索器（基于已加载的文档）
    """
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25
