# IBS RAG Assistant & Telegram Bot

An advanced Retrieval-Augmented Generation (RAG) assistant designed for university and academic administration inquiries (IBS documents). It features multilingual capabilities (including Khmer script support), hybrid search (Dense FAISS + Sparse BM25), cross-encoder reranking, and guardrailed generation using Groq LLMs.

---

## 📌 Roles of Key Scripts

* **`app.py`**: **Builds the Knowledge Base.** Loads all documents from `data/`, cleans the text (removing PDF artifacts and layout noise), generates dense vector embeddings and BM25 tokenized indices, and saves them to `faiss_store/`. Run this script once before starting the bot or whenever you add/update documents in `data/`.
* **`bot.py`**: **Runs the Telegram Bot.** Starts the interactive Telegram bot polling loop. It loads the existing vector store from `faiss_store/` and answers user questions in real-time.

---

## 🚀 Key Features

1. **Khmer & Multilingual Processing**:
   * Pre-processing rules in `src/data_loader.py` remove page numbers and layout noise while preserving Unicode and Khmer characters.
   * Utilizes `BAAI/bge-m3`, a state-of-the-art multilingual embedding model with strong South and Southeast Asian script support.
2. **Hybrid Search (Dense + Sparse)**:
   * Combines semantic embeddings (FAISS) with lexical keyword matching (BM25Okapi).
   * Unicode-aware character tokenization for Khmer script to ensure accurate course code and proper noun retrieval.
3. **Cross-Encoder Reranking**:
   * Uses `BAAI/bge-reranker-base` to rerank candidate passages retrieved by hybrid search, ensuring only the most contextually relevant chunks reach the LLM.
4. **Strict Guardrails & Hallucination Prevention**:
   * Grounded system prompt instructs the model to only answer using provided document context.
   * Clear out-of-scope fallback: if the context does not contain the answer, the bot informs the user directly rather than guessing.

---

## 📁 Project Structure

```text
RAG/
├── data/                  # Source documents (PDF, TXT, CSV, DOCX, XLSX, JSON)
├── faiss_store/           # Generated FAISS index, BM25 corpus, and metadata
├── src/
│   ├── data_loader.py     # Document ingestion and layout cleaning
│   ├── embedding.py       # Text chunking and embedding pipelines
│   ├── vectorstore.py     # FAISS vector store + BM25 hybrid search
│   ├── reranker.py        # Cross-Encoder reranker
│   └── search.py          # Complete RAG pipeline and Groq integration
├── app.py                 # Script to build and test the vector store
├── bot.py                 # Telegram Bot server
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables and API keys
└── README.md              # Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Create and Activate Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables (`.env`)

Create or update your `.env` file in the project root:

```env
# Groq API Key (get from https://console.groq.com)
GROQ_API_KEY=your_groq_api_key_here

# Telegram Bot Token (get from @BotFather on Telegram)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# LLM Model on Groq
LLM_MODEL=openai/gpt-oss-120b

# Multilingual Embedding Model
EMBEDDING_MODEL=BAAI/bge-m3

# Reranker Model
RERANKER_MODEL=BAAI/bge-reranker-base
```

---

## 🏃 Usage Guide

### Step 1: Add Your Documents
Place your PDFs, text files, spreadsheets, or documents into the `data/` folder.

### Step 2: Build the Vector Store (`app.py`)
Run `app.py` to index the documents into FAISS and BM25:

```bash
python app.py
```
> **Note:** The first run will automatically download the `BAAI/bge-m3` embedding model and the `BAAI/bge-reranker-base` reranker model to your local cache.

### Step 3: Start the Telegram Bot (`bot.py`)
Run `bot.py` to start receiving and answering questions on Telegram:

```bash
python bot.py
```

### Available Telegram Commands:
* `/start` - Displays the welcome message.
* `/help` - Displays usage instructions and guidelines.
* Direct text - Type any question to get an answer retrieved from official documents.
