# retrieval/cache.py

import functools
from collections import OrderedDict
from datetime import datetime
from typing import List

import faiss
import numpy as np
from langchain_core.documents import Document


# L2：Embedding → Docs
class RetrievalResultCache:
    """
    基于 FAISS 的检索结果缓存（Embedding 级）
    """
    def __init__(
            self,
            embedding_dim: int,
            max_cache_entries: int = 512,
            similarity_threshold: float = 0.9,
            ttl: int = 3600,
    ):
        self.embedding_dim = embedding_dim
        self.max_cache_entries = max_cache_entries
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl

        self.index = faiss.IndexFlatL2(embedding_dim)
        self.cache: OrderedDict[int, dict] = OrderedDict()
        self.current_id = 0

    def get(self, query_embedding: List[float]):
        """
        查询是否命中缓存
        """
        if self.index.ntotal == 0:
            return None

        vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(vec, 1)

        idx = indices[0][0]
        dist = distances[0][0]

        if (
                idx != -1
                and dist <= (1 - self.similarity_threshold) ** 2
                and idx in self.cache
        ):
            entry = self.cache[idx]
            if (datetime.now() - entry["timestamp"]).total_seconds() <= self.ttl:
                self.cache.move_to_end(idx)
                return entry["docs"]

        return None

    def add(self, query_embedding: List[float], docs: List[Document]):
        """
        写入缓存
        """
        vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        self.index.add(vec)

        self.cache[self.current_id] = {
            "docs": docs,
            "timestamp": datetime.now(),
        }
        self.current_id += 1

        self._evict()

    def _evict(self):
        """
        超出容量时淘汰
        """
        while len(self.cache) > self.max_cache_entries:
            self.cache.popitem(last=False)
