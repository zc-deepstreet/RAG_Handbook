# retrieval/rrf.py

from hashlib import sha256
from typing import List
from langchain_core.documents import Document


def reciprocal_rank_fusion(
        docs: List[List[Document]],
        k: int = 60,
        docs_return_num: int = 20,
) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF)
    用于融合多个有序文档列表（如 Multi-Query / HyDE 返回结果）
    """

    fused_scores = {}
    unique_docs_by_content = {}

    for doc_list in docs:
        for rank, doc in enumerate(doc_list):
            key = sha256(doc.page_content.encode("utf-8")).hexdigest()
            unique_docs_by_content[key] = doc
            fused_scores[key] = fused_scores.get(key, 0.0) + 1 / (rank + k)

    # 按融合得分排序
    sorted_docs = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    reranked_docs = [
        unique_docs_by_content[key] for key, _ in sorted_docs
    ]

    return reranked_docs[:docs_return_num]
