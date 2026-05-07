from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, chunks : list[Document], vector_db: Chroma):
        self.chunks = chunks
        self.vector_db = vector_db
        # Tokenize chunks using lower and split
        self.tokenized_chunks = [chunk.page_content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def search(self, query, k=5):
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        chroma_results = self.vector_db.similarity_search_with_score(query, k=k)

        # Combine BM25 and Chroma results
        order_bm25_results = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)

        rrf_scores : dict[int, float] = {}
        for rank, (idx, score) in enumerate(order_bm25_results):
            rrf_scores[idx] = 1 / (60 + rank +1)

        for rank, (doc, score) in enumerate(chroma_results):
            # Para cada doc de Chroma, busca su índice en self.chunks
            idx = None
            for i, chunk in enumerate(self.chunks):
                if chunk.page_content == doc.page_content:
                    idx = i
                    break
            if idx is None:
                continue
            if idx in rrf_scores:
                rrf_scores[idx] += 1 / (60 + rank + 1)
            else:
                rrf_scores[idx] = 1 / (60 + rank + 1)

        rrf_scores = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        top_k_rrf_scores = rrf_scores[:k]

        return [(self.chunks[idx], score) for idx, score in top_k_rrf_scores]
