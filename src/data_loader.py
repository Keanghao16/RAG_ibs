import re
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader


# ---------------------------------------------------------------------------
# Improvement #1 — Pre-processing: strip PDF noise, preserve Khmer/Unicode
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Strip common PDF layout noise while preserving Khmer and other Unicode scripts.

    What is removed:
    - Control characters (non-printable ASCII below 0x20, except \\n and \\t)
    - Standalone page-number lines (a line that contains only digits)
    - Runs of 3+ consecutive blank lines collapsed to 2
    - Multiple consecutive spaces/tabs on a single line collapsed to one space

    What is preserved:
    - All Unicode code points (Khmer, Latin, digits, punctuation)
    - Single blank lines (paragraph breaks)
    - Leading indentation is kept intact
    """
    # Remove non-printable control characters (keep \\n, \\t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Remove standalone page-number lines
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse repeated spaces/tabs within a line
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain
    document structure. Text is cleaned via clean_text() to remove PDF noise
    while preserving Khmer/Unicode content.

    Supported formats: PDF, TXT, CSV, Excel (.xlsx), Word (.docx), JSON
    """
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    # --- PDF ---
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files")
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            for doc in loaded:
                doc.page_content = clean_text(doc.page_content)
            # Skip completely empty pages after cleaning
            loaded = [doc for doc in loaded if doc.page_content]
            documents.extend(loaded)
            print(f"[DEBUG] Loaded {len(loaded)} pages from {pdf_file.name}")
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    # --- TXT ---
    txt_files = list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] Found {len(txt_files)} TXT files")
    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file), encoding='utf-8')
            loaded = loader.load()
            for doc in loaded:
                doc.page_content = clean_text(doc.page_content)
            documents.extend([doc for doc in loaded if doc.page_content])
        except Exception as e:
            print(f"[ERROR] Failed to load TXT {txt_file}: {e}")

    # --- CSV ---
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV files")
    for csv_file in csv_files:
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            for doc in loaded:
                doc.page_content = clean_text(doc.page_content)
            documents.extend([doc for doc in loaded if doc.page_content])
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file}: {e}")

    # --- Excel ---
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    print(f"[DEBUG] Found {len(xlsx_files)} Excel files")
    for xlsx_file in xlsx_files:
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            for doc in loaded:
                doc.page_content = clean_text(doc.page_content)
            documents.extend([doc for doc in loaded if doc.page_content])
        except Exception as e:
            print(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")

    # --- Word ---
    docx_files = list(data_path.glob('**/*.docx'))
    print(f"[DEBUG] Found {len(docx_files)} Word files")
    for docx_file in docx_files:
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            for doc in loaded:
                doc.page_content = clean_text(doc.page_content)
            documents.extend([doc for doc in loaded if doc.page_content])
        except Exception as e:
            print(f"[ERROR] Failed to load Word {docx_file}: {e}")

    # --- JSON ---
    json_files = list(data_path.glob('**/*.json'))
    print(f"[DEBUG] Found {len(json_files)} JSON files")
    for json_file in json_files:
        try:
            loader = JSONLoader(str(json_file))
            loaded = loader.load()
            for doc in loaded:
                doc.page_content = clean_text(doc.page_content)
            documents.extend([doc for doc in loaded if doc.page_content])
        except Exception as e:
            print(f"[ERROR] Failed to load JSON {json_file}: {e}")

    print(f"[DEBUG] Total loaded documents (after cleaning): {len(documents)}")
    return documents


if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)
