# retrieval/retriever.py

def retrieve_docs(vector_db, query, k=6, fetch_k=20):
    """
    检索模块：
    - 查询嵌入
    - 相似度检索（MMR）
    """
    docs = vector_db.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k
    )
    return docs


def build_context(docs):
    """
    上下文提取模块：
    将检索到的文档整理为 Prompt 上下文
    """
    context = ""
    for i, doc in enumerate(docs):
        context += f"\n--- 资料片段 {i + 1} ---\n{doc.page_content}\n"
    return context
