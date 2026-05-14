# Implementation Plan — Nexla MCP Server

> Concrete execution plan derived from `phases.md` + `agents.md`. Each step specifies exact tools, APIs, function signatures, and file paths. Execute phases in order unless noted otherwise.

---

## Step 0: Project Scaffold

### 0.1 Initialize project
```bash
cd /mnt/c/Users/abhis/Desktop/Nexla
uv init --name nexla-mcp
# When prompted, set: Python >=3.10, MIT license, description: "MCP server for PDF Q&A"
```

### 0.2 Create directory structure
```bash
mkdir -p src/nexla_mcp tests
touch src/nexla_mcp/__init__.py
```

### 0.3 Configure pyproject.toml
Edit `pyproject.toml`:
- Add `[project] name = "nexla-mcp"`, version, description, license, requires-python >=3.10
- Add `[project.optional-dependencies]` with `dev` group: black, ruff
- Add `[project.scripts]` to expose CLI (optional)

### 0.4 Create .gitignore
```
__pycache__/
*.py[cod]
venv/
.venv/
.env
uv.lock
.rough/
*.egg-info/
dist/
build/
```

### 0.5 Verify scaffold
```bash
uv run python -c "print('ok')"
```
**Exit:** prints `ok` without errors.

---

## Step 1: Dependency Installation

Install packages using `uv add`. Run each command:

```bash
uv add fastmcp
uv add pymupdf
uv add pymupdf4llm
uv add chromadb
uv add fastembed              # Embeddings — ONNX-based, no PyTorch dependency
uv add "numpy<2"              # Required: NumPy 1.x for onnxruntime compatibility
uv add pydantic
uv add httpx
uv add tomli
uv add langchain
uv add langchain-ollama
uv add tqdm

# Dev dependencies
uv add --dev black ruff
```

**Note:** `sentence-transformers` is NOT used — it requires PyTorch which causes installation issues. Use `fastembed` instead (Flag Embedding model, BAAI/bge-small-en-v1.5, 384-dim).

**Verify:**
```bash
uv pip freeze | grep -E "fastmcp|pymupdf|chromadb|fastembed|faiss|pydantic|langchain"
```
**Exit:** All packages present. `uv run python -c "import fastmcp, fitz, chromadb; from fastembed.embedding import FlagEmbedding; print('all ok')"` succeeds.

---

## Step 2: Data Discovery

### 2.1 Scan all documents
```bash
uv run python - <<'EOF'
from pathlib import Path
data_dir = Path("data")
for pdf_path in sorted(data_dir.glob("*/[0-9]*.pdf")):
    qa_path = pdf_path.with_name(pdf_path.stem + "_qa.jsonl")
    print(f"{pdf_path.parent.name}\t{pdf_path.name}\t{qa_path.exists()}")
EOF
```

### 2.2 Build document inventory
Write to `src/nexla_mcp/inventory.py`:
```python
import json
import fitz
from pathlib import Path
from typing import Optional

def build_inventory(data_dir: Path = Path("data")) -> list[dict]:
    """
    Scans data/ and returns a list of dicts, one per document:
    {
        "id": str,
        "pdf_filename": str,
        "qa_filename": Optional[str],
        "pdf_exists": bool,
        "page_count": int,
        "qa_types": list[str]  # e.g. ["text-only", "multimodal-t"]
    }
    """
    inventory = []
    for doc_dir in sorted(data_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name

        # Find PDF
        pdfs = list(doc_dir.glob("*.pdf"))
        pdf_path = pdfs[0] if pdfs else None

        # Find qa.jsonl
        qa_files = list(doc_dir.glob("*_qa.jsonl"))
        qa_path = qa_files[0] if qa_files else None

        # Count PDF pages
        page_count = 0
        if pdf_path:
            try:
                doc = fitz.open(str(pdf_path))
                page_count = len(doc)
                doc.close()
            except Exception:
                page_count = 0

        # Read qa types
        qa_types = []
        if qa_path:
            try:
                with open(qa_path) as f:
                    for line in f:
                        entry = json.loads(line)
                        if "type" in entry:
                            qa_types.append(entry["type"])
            except Exception:
                pass

        inventory.append({
            "id": doc_id,
            "pdf_filename": pdf_path.name if pdf_path else "",
            "qa_filename": qa_path.name if qa_path else None,
            "pdf_exists": pdf_path is not None,
            "page_count": page_count,
            "qa_types": list(set(qa_types))
        })

    return inventory
```

### 2.3 Output inventory
```bash
uv run python -c "from nexla_mcp.inventory import build_inventory; [print(d) for d in build_inventory()]"
```
**Exit:** Markdown table in `docs/INVENTORY.md` with columns: id, filename, page_count (via PyMuPDF), document_type (inferred from filename/size).

---

## Step 3: PDF Text Extraction

File: `src/nexla_mcp/pdf_processor.py`

### 3.1 Define data models
```python
from pydantic import BaseModel

class PageChunk(BaseModel):
    doc_id: str
    doc_filename: str
    page_number: int  # 1-indexed
    chunk_index: int
    text: str
    token_count: int
```

### 3.2 Implement extractor
```python
import fitz  # PyMuPDF
import pymupdf4llm
from pathlib import Path
from tqdm import tqdm

def extract_text_from_pdf(pdf_path: Path) -> list[str]:
    """Returns list of page texts from a PDF using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Try markdown first (better for AI), fall back to plain text
        try:
            # CORRECTED: pass pdf_path (str), not fitz.Document; pages is 0-indexed page list
            md = pymupdf4llm.to_markdown(str(pdf_path), pages=[page_num])
            text = md  # to_markdown returns a single concatenated string for specified pages
        except Exception:
            text = page.get_text("text")
        pages.append(text)
    doc.close()
    return pages

def chunk_pages(pages: list[str], chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split a list of page texts into chunks of ~chunk_size tokens with overlap.

    Tokens are estimated as len(words) * 1.3.
    Chunks are split at sentence boundaries (period + space) where possible.

    Returns list of chunk strings.
    """
    import re

    def split_into_sentences(text: str) -> list[str]:
        """Split text on sentence boundaries (., !, ?) while preserving the delimiter."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def estimate_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)

    all_chunks = []

    for page_text in pages:
        sentences = split_into_sentences(page_text)
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)

            if current_tokens + sentence_tokens <= chunk_size:
                current_chunk += (" " + sentence).strip()
                current_tokens += sentence_tokens
            else:
                # Emit current chunk if non-empty
                if current_chunk:
                    all_chunks.append(current_chunk)
                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    # Backtrack: include last ~overlap tokens into next chunk
                    words = current_chunk.split()
                    overlap_words = " ".join(words[-int(overlap / 1.3):])
                    current_chunk = (overlap_words + " " + sentence).strip()
                    current_tokens = estimate_tokens(current_chunk)
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens

        # Don't forget the last chunk
        if current_chunk:
            all_chunks.append(current_chunk)

    return all_chunks
```

### 3.3 Implement extraction pipeline
```python
def extract_all_documents(data_dir: Path = Path("data")) -> list[PageChunk]:
    """Extract and chunk all PDFs. Returns all chunks."""
    chunks = []
    for pdf_path in sorted(data_dir.glob("*/[0-9]*.pdf")):
        doc_id = pdf_path.parent.name
        pages = extract_text_from_pdf(pdf_path)
        page_chunks = chunk_pages(pages)
        for i, chunk in enumerate(page_chunks):
            chunks.append(PageChunk(
                doc_id=doc_id,
                doc_filename=pdf_path.name,
                page_number=i + 1,
                chunk_index=i,
                text=chunk,
                token_count=int(len(chunk.split()) * 1.3)
            ))
    return chunks
```

### 3.4 Export for indexer
```python
# Add at bottom of pdf_processor.py:
if __name__ == "__main__":
    all_chunks = extract_all_documents()
    print(f"Total chunks: {len(all_chunks)}")
```

**Verify:**
```bash
uv run python -c "from nexla_mcp.pdf_processor import extract_all_documents; print(len(extract_all_documents()))"
```
**Exit:** Total chunk count > 0.

---

## Step 4: Embedding & Vector Store

File: `src/nexla_mcp/indexer.py`

**Library choice:** `fastembed` — ONNX-based, no PyTorch, works natively with ChromaDB.

### 4.1 Imports and setup
```python
from fastembed.embedding import FlagEmbedding
import chromadb
from chromadb.config import Settings
from pathlib import Path

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dim, top of MTEB leaderboard
PERSIST_DIR = Path("data/chroma_db")

# Initialize model once (lazy load on first call)
_model = None

def get_model() -> FlagEmbedding:
    global _model
    if _model is None:
        _model = FlagEmbedding(model_name=MODEL_NAME)
    return _model
```

### 4.2 ChromaDB client
```python
def get_chroma_client(persist_dir: Path = PERSIST_DIR):
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False)
    )
```

### 4.3 Encode texts helper
```python
def encode_texts(texts: list[str]) -> list[list[float]]:
    """Encode texts into fastembed embeddings."""
    model = get_model()
    return list(model.embed(texts))
```

### 4.4 Index all chunks
```python
def index_documents(chunks: list, collection_name: str = "nexla_docs"):
    """Embed chunks and store in ChromaDB."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)

    texts = [c.text for c in chunks]
    embeddings = encode_texts(texts)

    # Prepare metadata
    metadatas = [
        {
            "doc_id": c.doc_id,
            "doc_filename": c.doc_filename,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "token_count": c.token_count
        }
        for c in chunks
    ]

    ids = [f"{c.doc_id}_{c.page_number}_{c.chunk_index}" for c in chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)
```

### 4.5 Load existing index or re-index
```python
def get_or_create_index(data_dir: Path = Path("data")):
    """On startup: load existing ChromaDB or build from scratch."""
    if PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()):
        # Load existing
        return get_chroma_client().get_collection("nexla_docs")
    else:
        # Build fresh
        from nexla_mcp.pdf_processor import extract_all_documents
        chunks = extract_all_documents(data_dir)
        index_documents(chunks)
        return get_chroma_client().get_collection("nexla_docs")
```

**Exit:** Query ChromaDB with a test vector returns relevant chunks.

---

## Step 5: Retriever

File: `src/nexla_mcp/retriever.py`

### 5.1 Imports
```python
from nexla_mcp.indexer import encode_texts, get_or_create_index
```

### 5.2 Retrieve function
```python
def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """
    Returns list of dicts:
    {
        "text": str,
        "doc_id": str,
        "doc_filename": str,
        "page_number": int,
        "chunk_index": int,
        "score": float
    }
    """
    collection = get_or_create_index()

    # Embed question using fastembed
    question_embedding = encode_texts([question])[0]

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Parse results
    chunks = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        chunks.append({
            "text": results["documents"][0][i],
            "doc_id": metadata["doc_id"],
            "doc_filename": metadata["doc_filename"],
            "page_number": metadata["page_number"],
            "chunk_index": metadata["chunk_index"],
            "score": results["distances"][0][i]
        })
    return chunks
```

### 5.3 Build context string
```python
def build_context(chunks: list[dict]) -> str:
    """Assembles chunks into a context string with citations."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}] {chunk['doc_filename']} (page {chunk['page_number']}):\n{chunk['text']}"
        )
    return "\n\n".join(context_parts)
```

**Verify:**
```bash
uv run python -c "from nexla_mcp.retriever import retrieve; r = retrieve('what is the document about', 3); print(len(r))"
```
**Exit:** Returns 3 chunks with metadata.

---

## Step 6: LLM Answer Generation

File: `src/nexla_mcp/llm.py`

### 6.1 Model selection
Use **Ollama** (local, free) via LangChain:
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3",  # or "mistral", "phi3"
    base_url="http://localhost:11434",
    temperature=0.3
)
```

Fallback: if Ollama unavailable, set `OPENAI_API_KEY` env var and use `langchain_openai.ChatOpenAI(model="gpt-4o-mini")`.

### 6.2 System prompt
```python
SYSTEM_PROMPT = """You are a helpful assistant answering questions based on provided document excerpts.

Rules:
- Answer only using the provided context below
- If the answer is not in the context, say "I don't know based on the provided documents."
- Always cite sources using the format [Source N] where N is the number in the document
- Be concise and accurate
- Do not make up information
"""
```

### 6.3 Generate answer with sources (CORRECTED)
```python
def generate_answer_with_sources(question: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Returns (answer: str, sources: list[dict]).
    sources format: [{"doc_filename": str, "page_number": int, "text": str}, ...]

    NOTE: Returns plain dicts (not Source Pydantic model instances) so that
    server.py can use them directly without needing .model_dump() calls.
    """
    if not chunks:
        return "I don't know based on the provided documents.", []

    from nexla_mcp.retriever import build_context
    context = build_context(chunks)

    human_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = llm.invoke(SYSTEM_PROMPT + f"\n\n{human_prompt}")
    answer = response.content if hasattr(response, 'content') else str(response)

    # Returns plain dicts — server.py uses them directly without .model_dump()
    sources = [
        {
            "doc_filename": c["doc_filename"],
            "page_number": c["page_number"],
            "text": c["text"]
        }
        for c in chunks
    ]

    return answer, sources
```

### 6.4 models.py — Pydantic models
File: `src/nexla_mcp/models.py`
```python
from pydantic import BaseModel
from typing import Optional

class Source(BaseModel):
    doc_filename: str
    page_number: int
    section: Optional[str] = None
    text: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
```

**Verify:**
```bash
uv run python -c "from nexla_mcp.llm import generate_answer_with_sources; a, s = generate_answer_with_sources('test', []); print(a)"
# Expected: "I don't know..."
```
**Note:** Requires Ollama running (`ollama serve`) or OpenAI API key set.

---

## Step 7: MCP Server (FastMCP)

File: `src/nexla_mcp/server.py`

### 7.1 Imports and startup indexing
```python
from fastmcp import FastMCP
from nexla_mcp.models import QueryRequest, QueryResponse, Source
from nexla_mcp.retriever import retrieve
from nexla_mcp.llm import generate_answer_with_sources
from contextlib import asynccontextmanager

# Global collection — initialize on startup
_collection = None

def get_collection():
    global _collection
    if _collection is None:
        from nexla_mcp.indexer import get_or_create_index
        _collection = get_or_create_index()
    return _collection

mcp = FastMCP("Nexla PDF Q&A Server")
```

### 7.2 Primary tool: query_documents
```python
@mcp.tool
def query_documents(question: str, top_k: int = 5) -> dict:
    """
    Ask a question about the PDF documents and receive a grounded answer.

    Args:
        question: Natural language question about the documents.
        top_k: Number of document chunks to retrieve (default 5).

    Returns:
        dict with keys: answer (str), sources (list of dicts with doc_filename, page_number, text)
    """
    chunks = retrieve(question, top_k=top_k)
    answer, sources = generate_answer_with_sources(question, chunks)
    # CORRECTED: sources is already list[dict] from generate_answer_with_sources,
    # not Pydantic model instances — use directly without .model_dump()
    return {
        "answer": answer,
        "sources": sources
    }
```

### 7.3 Secondary tools
```python
@mcp.tool
def list_documents() -> list[dict]:
    """List all indexed documents with their IDs and filenames."""
    from nexla_mcp.inventory import build_inventory
    return build_inventory()

@mcp.tool
def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Raw semantic search — returns relevant chunks without LLM answer generation."""
    chunks = retrieve(query, top_k=top_k)
    return chunks

@mcp.tool
def get_document_summary(doc_id: str) -> dict:
    """Get metadata and chunk count for a specific document."""
    from nexla_mcp.inventory import build_inventory
    inventory = build_inventory()
    doc = next((d for d in inventory if d["id"] == doc_id), None)
    if not doc:
        return {"error": f"Document {doc_id} not found"}
    return doc
```

### 7.4 Main entry point
File: `src/nexla_mcp/__main__.py`
```python
from nexla_mcp.server import mcp

if __name__ == "__main__":
    mcp.run()  # stdio transport by default
```

### 7.5 Run the server
```bash
# Start Ollama first (required for LLM)
ollama serve &
uv run python -m nexla_mcp.server
```
**Exit:** Server starts, responds to `query_documents` calls via MCP client.

---

## Step 8: Testing

Directory: `tests/`

### 8.1 Test extraction
File: `tests/test_pdf_processor.py`
```python
from pathlib import Path
from nexla_mcp.pdf_processor import extract_text_from_pdf, chunk_pages

def test_extract_text_from_pdf():
    pdf_path = Path("data/0/P19-1598.pdf")
    if pdf_path.exists():
        pages = extract_text_from_pdf(pdf_path)
        assert len(pages) > 0
        assert all(isinstance(p, str) and len(p) > 0 for p in pages)

def test_chunk_pages():
    pages = ["This is page one. " * 100, "This is page two. " * 100]
    chunks = chunk_pages(pages, chunk_size=500, overlap=50)
    assert len(chunks) > 0
```

### 8.2 Test retrieval
File: `tests/test_retriever.py`
```python
from nexla_mcp.retriever import retrieve, build_context

def test_retrieve_returns_chunks():
    chunks = retrieve("what is this document about", top_k=3)
    assert len(chunks) <= 3
    assert all("text" in c and "doc_filename" in c and "page_number" in c for c in chunks)

def test_build_context():
    chunks = [{"doc_filename": "test.pdf", "page_number": 1, "text": "Hello world"}]
    ctx = build_context(chunks)
    assert "test.pdf" in ctx
    assert "Hello world" in ctx
```

### 8.3 Test indexer
File: `tests/test_indexer.py`
```python
from pathlib import Path
from nexla_mcp.indexer import get_model, encode_texts, get_chroma_client

def test_model_loads():
    """Model loads and encodes without error."""
    model = get_model()
    embeddings = list(model.embed(["test sentence"]))
    embedding = embeddings[0]
    assert len(embedding) == 384  # BAAI/bge-small-en-v1.5 is 384-dim

def test_encode_texts():
    """encode_texts returns list of float vectors."""
    embeddings = encode_texts(["hello world", "fastembed is fast"])
    assert len(embeddings) == 2
    assert all(isinstance(e, list) for e in embeddings)
    assert all(isinstance(x, float) for e in embeddings for x in e)
    assert len(embeddings[0]) == 384

def test_chroma_client_persists(tmp_path):
    """ChromaDB can add and retrieve a chunk."""
    from fastembed.embedding import FlagEmbedding

    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection(name="test_collection")

    model = FlagEmbedding(model_name="BAAI/bge-small-en-v1.5")
    embedding = list(model.embed(["hello world"]))[0]

    collection.add(
        ids=["chunk_1"],
        embeddings=[embedding],
        documents=["hello world"],
        metadatas=[{"doc_id": "test", "doc_filename": "test.pdf", "page_number": 1, "chunk_index": 0, "token_count": 2}]
    )

    results = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        include=["documents", "metadatas"]
    )
    assert results["documents"][0][0] == "hello world"
    assert results["metadatas"][0][0]["doc_filename"] == "test.pdf"
```

### 8.4 Test LLM
File: `tests/test_llm.py`
```python
from nexla_mcp.llm import generate_answer_with_sources

def test_generate_answer_with_empty_chunks():
    """No chunks should return 'I don't know'."""
    answer, sources = generate_answer_with_sources("What is AI?", [])
    assert "don't know" in answer.lower() or "cannot answer" in answer.lower()
    assert sources == []

def test_generate_answer_with_chunks():
    """With valid chunks, should return an answer and sources."""
    chunks = [
        {
            "text": "Artificial intelligence is the simulation of human intelligence by machines.",
            "doc_filename": "test.pdf",
            "page_number": 1,
            "chunk_index": 0
        }
    ]
    answer, sources = generate_answer_with_sources("What is AI?", chunks)
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert len(sources) == 1
    assert sources[0]["doc_filename"] == "test.pdf"
    assert sources[0]["page_number"] == 1

def test_generate_answer_with_irrelevant_question():
    """
    Chunks are about AI, question is completely unrelated (cooking).
    LLM should refuse to answer from context and return a 'don't know' variant.
    """
    chunks = [
        {
            "text": "Artificial intelligence is the simulation of human intelligence by machines.",
            "doc_filename": "test.pdf",
            "page_number": 1,
            "chunk_index": 0
        }
    ]
    answer, sources = generate_answer_with_sources(
        "What is the best recipe for chocolate chip cookies?",
        chunks
    )
    answer_lower = answer.lower()
    # LLM should indicate it cannot answer from the provided context
    assert any(
        phrase in answer_lower
        for phrase in [
            "don't know",
            "cannot answer",
            "not contain",
            "not provided",
            "not in the context",
            "based on the documents",
            "don't have",
        ]
    ), f"Expected 'don't know' variant for irrelevant question, got: {answer}"
    # Sources may still be returned but answer should be a refusal
    assert isinstance(sources, list)
```

### 8.6 Run all tests
```bash
uv run python tests/test_indexer.py
```

### 8.7 Random QA verification script (CORRECTED)
Run `scripts/random_qa_sample.py` to randomly sample questions from `*_qa.jsonl` files and print retrieved context vs expected answer for manual review.

File: `scripts/random_qa_sample.py`
```python
#!/usr/bin/env python3
"""
Randomly sample questions from *_qa.jsonl files and compare
retrieved context against the expected answer for manual verification.

Usage:
    python scripts/random_qa_sample.py --count 5

NOTE: This script calls the retriever directly (not via FastMCP Client),
since FastMCP Client connects to a running server URL, not a local file path.
To test the full LLM pipeline, start the server separately and use an MCP client.
"""
import argparse
import json
import random
from pathlib import Path

def find_qa_files(data_dir: Path = Path("data")) -> list[Path]:
    return list(data_dir.glob("*/*_qa.jsonl"))

def load_qa_entries(qa_files: list[Path]):
    entries = []
    for qa_file in qa_files:
        try:
            with open(qa_file) as f:
                for line in f:
                    entry = json.loads(line)
                    entry["_qa_file"] = qa_file.name
                    entry["_doc_id"] = qa_file.parent.name
                    entries.append(entry)
        except Exception as e:
            print(f"Error reading {qa_file}: {e}")
    return entries

def run_retriever_query(question: str) -> dict:
    """Call the retriever directly (bypasses LLM for quick context verification)."""
    from nexla_mcp.retriever import retrieve, build_context
    chunks = retrieve(question, top_k=3)
    context = build_context(chunks)
    sources = [
        {"doc_filename": c["doc_filename"], "page_number": c["page_number"]}
        for c in chunks
    ]
    return {"context": context, "sources": sources}

def main():
    parser = argparse.ArgumentParser(description="Random QA sampler")
    parser.add_argument("--count", type=int, default=5, help="Number of questions to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    qa_files = find_qa_files()
    entries = load_qa_entries(qa_files)

    print(f"\nFound {len(entries)} QA entries across {len(qa_files)} files.")
    print(f"Sampling {args.count} random questions...\n")
    print("=" * 80)

    sampled = random.sample(entries, min(args.count, len(entries)))

    for i, entry in enumerate(sampled, 1):
        print(f"\n--- Sample {i}/{len(sampled)} ---")
        print(f"Document: {entry['_doc_id']} / {entry['_qa_file']}")
        print(f"Type:     {entry.get('type', 'unknown')}")
        print(f"Question: {entry['question']}")
        print(f"Expected Answer: {entry['answer']}")

        # Run retriever (no LLM call — quick context check)
        try:
            result = run_retriever_query(entry["question"])
            print(f"\nRetrieved Context:\n{result.get('context', 'N/A')[:500]}...")
            sources = result.get("sources", [])
            if sources:
                print("Sources retrieved:")
                for j, src in enumerate(sources, 1):
                    print(f"  [{j}] {src.get('doc_filename', '?')} (page {src.get('page_number', '?')})")
            else:
                print("Sources: (none)")
        except Exception as e:
            print(f"\nRetriever Error: {e}")

        print("-" * 80)

if __name__ == "__main__":
    main()
```

**Exit:** Script runs, prints sampled questions with retrieved context + expected answer for manual verification.

---

## Step 9: README.md

File: `README.md` (root)

### 9.1 Setup Instructions
```markdown
## Setup

```bash
git clone <repo-url>
cd nexla-mcp
uv sync
# Requires Ollama running: ollama serve
uv run python -m nexla_mcp.server
```
```

### 9.2 Architecture Overview
Document the pipeline:
```
PDF → PyMuPDF + pymupdf4llm → Text/Markdown
  → Chunking (500 tokens, 50 overlap)
  → fastembed (BAAI/bge-small-en-v1.5, 384-dim) → Embeddings
  → ChromaDB (vector store, persisted)
  → Query: ChromaDB retrieval → LLM (Ollama/GPT) → Answer + Sources
  → FastMCP (MCP tool: query_documents)
```

### 9.3 Tool Documentation
Document each MCP tool with name, description, input schema, output schema, example.

### 9.4 Example Interaction Log
Include 3 Q&A pairs from the actual qa.jsonl files with question, answer, source references.

### 9.5 Vibe Coding Section
```markdown
## Vibe Coding Setup

### AI Coding Tools Used
- Claude Code (this session) — used for architecture planning, code generation, debugging
- [Cursor / Copilot / etc.] — [how used]

### Prompting Strategy
- [Describe prompting approach]
- What worked: [X]
- What didn't: [Y]

### Human vs AI Contribution
- AI generated: [what]
- Human corrected/overrode: [what]

### Reflection on AI Tooling
- [Honest perspective on AI in SE workflows]
```

---

## Step 10: Final Verification

> Note: GitHub repo already set up by user. Skip Step 10 GitHub setup.

```bash
# Fresh clone test (run in a temp directory)
cd /tmp && rm -rf nexla-mcp-test && \
  git clone https://github.com/<username>/nexla-mcp.git && \
  cd nexla-mcp && uv sync && uv run python -c "from nexla_mcp import server; print('import ok')"
```

Checklist:
- [ ] `uv sync` succeeds
- [ ] `uv run python -m nexla_mcp.server` starts without errors
- [ ] `query_documents` returns answers with sources
- [ ] README complete with 3+ Q&A examples
- [ ] `.rough/` excluded from repo (in .gitignore)
- [ ] Repo is public
- [ ] Email sent with repo URL

---

## Dependency Chain

```
Step 0 → Step 1 → Step 2 → Step 3 → Step 4 ─┐
                              (can overlap)    │
                                              ↓
                              Step 5 → Step 6 → Step 7 → Step 8 → Step 9 → Step 10
```

---

## Corrections Applied (2026-05-14)

| Location | Original (broken) | Correction |
|----------|-------------------|------------|
| Step 3.2 `pdf_processor.py` | `pymupdf4llm.to_markdown(doc, pages=[page_num])` — passed fitz.Document | `pymupdf4llm.to_markdown(str(pdf_path), pages=[page_num])` — pass file path str |
| Step 4.2 `indexer.py` | `chromadb.Client(Settings(persist_directory=...))` — wrong API | `chromadb.PersistentClient(path=..., settings=Settings(...))` |
| Step 6.3 `llm.py` | Returns dicts; server.py called `.model_dump()` on dicts | Returns `list[dict]` explicitly; server.py uses directly |
| Step 7.2 `server.py` | `"sources": [s.model_dump() for s in sources]` — fails (s is dict) | `"sources": sources` — use dicts directly |
| Step 8.3 `test_indexer.py` | `chromadb.Client(Settings(persist_directory=...))` | `chromadb.PersistentClient(path=...)` |
| Step 8.7 `random_qa_sample.py` | `Client("src/nexla_mcp/server.py")` — wrong FastMCP API | Direct import of `retrieve()` + `build_context()` functions |

---

*Last updated: 2026-05-14*
