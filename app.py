import os
import shutil
from dotenv import load_dotenv

load_dotenv()   # must be before any src imports that read env vars

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

if __name__ == "__main__":

    PERSIST_DIR     = "faiss_store"
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ------------------------------------------------------------------ #
    # Step 1 — wipe any old index built with a different embedding model  #
    # ------------------------------------------------------------------ #
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print(f"[INFO] Removed old index at '{PERSIST_DIR}' to rebuild with {embedding_model}")

    # ------------------------------------------------------------------ #
    # Step 2 — load & clean documents                                     #
    # ------------------------------------------------------------------ #
    docs = load_all_documents("data")

    # ------------------------------------------------------------------ #
    # Step 3 — build FAISS + BM25 index with the correct embedding model  #
    # ------------------------------------------------------------------ #
    store = FaissVectorStore(PERSIST_DIR, embedding_model=embedding_model)
    store.build_from_documents(docs)
    print(f"[INFO] FAISS + BM25 index built and saved to '{PERSIST_DIR}' with {embedding_model}")

    # ------------------------------------------------------------------ #
    # Step 4 — test a query through the full RAG pipeline                 #
    # ------------------------------------------------------------------ #
    rag_search = RAGSearch()
    query = "tell me about IBS Timetable?"
    answer = rag_search.search_and_summarize(query, top_k=3)
    print("\nAnswer:", answer)
