import os
from typing import List, Dict
from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Improvement #2 — Reranking Step (cross-encoder)
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Re-scores retrieved chunks using a cross-encoder model.

    Unlike bi-encoders (which embed query and document independently),
    a cross-encoder attends jointly to both query and document, giving
    significantly more accurate relevance scores — at the cost of being
    slower (run only on the top-k candidates, never on the full corpus).

    Default model: BAAI/bge-reranker-base
      - Multilingual-friendly, works well on Khmer + English mixed text
      - ~280 MB, runs on CPU
    Alternative (larger, more accurate): BAAI/bge-reranker-large (~1.3 GB)
    """

    def __init__(self, model_name: str = None):
        model_name = model_name or os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
        self.model = CrossEncoder(model_name)
        print(f"[INFO] Reranker loaded: {model_name}")

    def rerank(self, query: str, results: List[Dict], top_n: int = 5) -> List[Dict]:
        """
        Re-rank retrieved results and return the top_n most relevant ones.

        Args:
            query:   The user's question.
            results: List of result dicts from hybrid_query() / query().
                     Each must have a 'metadata' key with a 'text' field.
            top_n:   How many results to keep after reranking.

        Returns:
            Reranked and truncated list of result dicts, each augmented
            with a 'rerank_score' field.
        """
        if not results:
            return []

        pairs = [
            (query, r["metadata"].get("text", "") or r["metadata"].get("page_content", ""))
            for r in results
        ]
        scores = self.model.predict(pairs)

        scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        reranked = []
        for score, result in scored[:top_n]:
            result = dict(result)           # shallow copy to avoid mutating originals
            result["rerank_score"] = float(score)
            reranked.append(result)

        print(f"[INFO] Reranked {len(results)} candidates → kept top {len(reranked)}")
        return reranked
