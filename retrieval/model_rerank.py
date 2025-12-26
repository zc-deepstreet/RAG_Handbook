# retrieval/model_rerank.py

import torch
from typing import List
from langchain_core.documents import Document


def model_rerank(
        query: str,
        docs: List[Document],
        rerank_tokenizer,
        rerank_model,
        docs_return_num: int = 20,
        batch_size: int = 64,
) -> List[Document]:
    """
    使用 CrossEncoder 模型对候选文档进行重排序
    """

    if not docs:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scores = []

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]

        inputs = rerank_tokenizer(
            text=[query] * len(batch_docs),
            text_pair=[doc.page_content for doc in batch_docs],
            padding=True,
            truncation=True,
            max_length=rerank_tokenizer.model_max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = rerank_model(**inputs)
            batch_scores = outputs.logits.squeeze()

        scores.extend(batch_scores.tolist())

    reranked_docs = [
        doc
        for doc, _ in sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )
    ]

    return reranked_docs[:docs_return_num]
