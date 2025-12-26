from retrieval.multi_query import generate_multi_queries
from retrieval.hyde import generate_hypothetical_doc
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.model_rerank import model_rerank


def retrieve_docs(
        vector_db,
        query: str,
        llm=None,
        k: int = 30,
        fetch_k: int = 60,
        use_multi_query: bool = False,
        use_hyde: bool = False,
        use_hybrid: bool = False,
        bm25_retriever=None,
        use_rrf: bool = False,
        use_model_rerank: bool = False,
        rerank_tokenizer=None,
        rerank_model=None,
        final_top_n: int = 6,
):
    """
    检索主入口（支持 Hybrid Retrieval）
    """

    # ---------- 1. 构造多路检索 Query ----------
    search_texts = [query]

    if use_multi_query and llm is not None:
        search_texts.extend(generate_multi_queries(llm, query))

    if use_hyde and llm is not None:
        search_texts.append(generate_hypothetical_doc(llm, query))

    all_doc_lists = []

    # ---------- 2. 向量检索（MMR） ----------
    for text in search_texts:
        docs = vector_db.max_marginal_relevance_search(
            text,
            k=k,
            fetch_k=fetch_k,
        )
        all_doc_lists.append(docs)

    # ---------- 3. Hybrid：BM25 关键词检索 ----------
    if use_hybrid and bm25_retriever is not None:
        bm25_docs = bm25_retriever.invoke(query)
        all_doc_lists.append(bm25_docs)

    # ---------- 4. RRF 融合 ----------
    if use_rrf:
        fused_docs = reciprocal_rank_fusion(
            all_doc_lists,
            docs_return_num=max(final_top_n * 3, 20),
        )
    else:
        fused_docs = [doc for docs in all_doc_lists for doc in docs]

    # ---------- 5. 模型重排序（精排） ----------
    if use_model_rerank and rerank_model is not None:
        fused_docs = model_rerank(
            query=query,
            docs=fused_docs,
            rerank_tokenizer=rerank_tokenizer,
            rerank_model=rerank_model,
            docs_return_num=final_top_n,
        )
    else:
        fused_docs = fused_docs[:final_top_n]

    return fused_docs


def build_context(docs):
    context = ""
    for i, doc in enumerate(docs):
        context += f"\n--- 资料片段 {i + 1} ---\n{doc.page_content}\n"
    return context
