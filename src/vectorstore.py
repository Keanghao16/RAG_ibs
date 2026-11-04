import os
import faiss
import numpy as np
import pickle
from typing import List, Any, Optional
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

# ---------------------------------------------------------------------------
# Improvement #2 — Hybrid Search: BM25 (sparse) + FAISS (dense)
# ---------------------------------------------------------------------------
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("[WARNING] rank-bm25 not installed — BM25 disabled. Run: pip install rank-bm25")


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Unicode-aware tokenizer for BM25.

    For Latin scripts: lowercased whitespace tokens.
    For Khmer and other non-ASCII scripts: each Unicode character becomes its
    own token, since Khmer does not use whitespace between words. This gives
    BM25 character-level overlap matching for Khmer queries.
    """
    tokens = []
    for word in text.lower().split():
        if any(ord(c) > 127 for c in word):
            # Non-ASCII word (e.g. Khmer) → character-level tokens
            tokens.extend(list(word))
        else:
            tokens.append(word)
    return tokens


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata: List[Any] = []
        self.bm25: Optional[Any] = None       # BM25 sparse index
        self.bm25_texts: List[str] = []       # Raw text corpus for BM25
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        texts = [chunk.page_content for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas, texts)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: List[Any] = None,
        texts: List[str] = None,
    ):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)
        if texts:
            self.bm25_texts.extend(texts)

        # Rebuild BM25 index over the full corpus
        if BM25_AVAILABLE and self.bm25_texts:
            tokenized = [tokenize_for_bm25(t) for t in self.bm25_texts]
            self.bm25 = BM25Okapi(tokenized)
            print(f"[INFO] BM25 index built over {len(self.bm25_texts)} chunks.")

        print(f"[INFO] Added {embeddings.shape[0]} vectors to FAISS index.")

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path  = os.path.join(self.persist_dir, "metadata.pkl")
        bm25_path  = os.path.join(self.persist_dir, "bm25.pkl")

        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        with open(bm25_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "texts": self.bm25_texts}, f)
        print(f"[INFO] Saved FAISS index, BM25 index, and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path  = os.path.join(self.persist_dir, "metadata.pkl")
        bm25_path  = os.path.join(self.persist_dir, "bm25.pkl")

        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
            self.bm25       = data.get("bm25")
            self.bm25_texts = data.get("texts", [])
            print(f"[INFO] Loaded BM25 index ({len(self.bm25_texts)} chunks).")

        print(f"[INFO] Loaded FAISS index and metadata from {self.persist_dir}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _dense_search(self, query_emb: np.ndarray, fetch_k: int) -> dict:
        """Return {faiss_idx: normalized_similarity_score}."""
        D, I = self.index.search(query_emb, fetch_k)
        scores = {}
        for idx, dist in zip(I[0], D[0]):
            if idx == -1:
                continue
            scores[int(idx)] = 1.0 / (1.0 + float(dist))   # convert L2 dist → similarity
        if scores:
            max_val = max(scores.values())
            scores = {k: v / max_val for k, v in scores.items()}
        return scores

    def _bm25_search(self, query_text: str, fetch_k: int) -> dict:
        """Return {corpus_idx: normalized_bm25_score}."""
        if not (BM25_AVAILABLE and self.bm25):
            return {}
        tokens = tokenize_for_bm25(query_text)
        raw = self.bm25.get_scores(tokens)
        top_indices = np.argsort(raw)[::-1][:fetch_k]
        max_val = raw[top_indices[0]] if raw[top_indices[0]] > 0 else 1.0
        return {
            int(idx): float(raw[idx]) / max_val
            for idx in top_indices
            if raw[idx] > 0
        }

    def hybrid_query(
        self,
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.6,
    ) -> List[dict]:
        """
        Hybrid (dense + sparse) retrieval.

        Score = alpha × dense_score + (1 − alpha) × bm25_score
        alpha = 1.0 → pure semantic search
        alpha = 0.0 → pure BM25 keyword search
        Default alpha = 0.6 biases towards semantic but still benefits from
        exact keyword matches (course codes, Khmer proper nouns, etc.).
        """
        print(f"[INFO] Hybrid query (alpha={alpha}): '{query_text}'")
        fetch_k = min(top_k * 3, self.index.ntotal)

        query_emb   = self.model.encode([query_text]).astype('float32')
        dense_scores = self._dense_search(query_emb, fetch_k)
        bm25_scores  = self._bm25_search(query_text, fetch_k)

        all_indices = set(dense_scores) | set(bm25_scores)
        combined = {
            idx: alpha * dense_scores.get(idx, 0.0) + (1 - alpha) * bm25_scores.get(idx, 0.0)
            for idx in all_indices
        }

        top_indices = sorted(combined, key=combined.get, reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({
                "index": idx,
                "score": combined[idx],
                "dense_score": dense_scores.get(idx, 0.0),
                "bm25_score": bm25_scores.get(idx, 0.0),
                "metadata": meta,
            })
        return results

    def query(self, query_text: str, top_k: int = 5) -> List[dict]:
        """Dense-only search (kept for backward compatibility)."""
        print(f"[INFO] Dense query: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        D, I = self.index.search(query_emb, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results


if __name__ == "__main__":
    from src.data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.hybrid_query("What is the IBS timetable?", top_k=3))
