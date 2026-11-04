import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.vectorstore import FaissVectorStore
from src.reranker import CrossEncoderReranker

load_dotenv()

# ---------------------------------------------------------------------------
# Improvement #4 — Guardrail system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an official IBS (Institute of Business Studies) assistant.
Your sole job is to answer student questions using ONLY the provided context extracted from official IBS documents.

Rules you must follow:
1. Answer only from the provided context. Do not use outside knowledge.
2. If the context clearly contains the answer, give a helpful and concise response.
3. If the context does NOT contain enough information to answer confidently, respond with exactly:
   "⚠️ This information is not found in the official IBS documents. Please contact IBS directly for accurate information."
4. Never fabricate fees, dates, names, course codes, or any facts.
5. When helpful, quote specific details from the context (e.g. dates, deadlines, amounts).
"""


class RAGSearch:
    """
    Full RAG pipeline:
      1. Hybrid retrieval  — FAISS (dense) + BM25 (sparse)
      2. Reranking         — cross-encoder narrows candidates to best top_k
      3. LLM generation    — Groq with guardrailed system prompt
    """

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = None,
        llm_model: str = None,
        use_reranker: bool = True,
    ):
        # --- Embedding & vector store ---
        embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path  = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # --- LLM ---
        groq_api_key = os.getenv("GROQ_API_KEY")
        llm_model    = llm_model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

        # --- Reranker (optional) ---
        self.reranker = CrossEncoderReranker() if use_reranker else None

    def search_and_summarize(self, query: str, top_k: int = 3) -> str:
        """
        Full RAG pipeline for a single query.

        Steps:
          1. Hybrid retrieval — fetch (top_k × 4) candidates using FAISS + BM25
          2. Reranking        — cross-encoder scores and filters to best top_k
          3. Build context    — concatenate chunk texts with clear separators
          4. Fallback guard   — return out-of-scope message if context is empty
          5. LLM generation   — guardrailed prompt prevents hallucination
        """
        # Step 1: Hybrid retrieval (fetch more than needed for reranker to work with)
        fetch_k = top_k * 4 if self.reranker else top_k
        results = self.vectorstore.hybrid_query(query, top_k=fetch_k)

        # Step 2: Rerank and narrow down
        if self.reranker and results:
            results = self.reranker.rerank(query, results, top_n=top_k)

        # Step 3: Build context string
        texts = [
            r["metadata"].get("text", "") or r["metadata"].get("page_content", "")
            for r in results
            if r.get("metadata")
        ]
        texts = [t.strip() for t in texts if t.strip()]
        context = "\n\n---\n\n".join(texts)

        # Step 4: Fallback if no usable context was retrieved
        if not context:
            return (
                "⚠️ This information is not found in the official IBS documents. "
                "Please contact IBS directly for accurate information."
            )

        # Step 5: LLM generation with guardrailed prompt
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Question: {query}\n\n"
                f"Context from IBS documents:\n{context}"
            )),
        ]
        response = self.llm.invoke(messages)
        return response.content


if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is the IBS timetable?"
    answer = rag_search.search_and_summarize(query, top_k=3)
    print("Answer:", answer)
