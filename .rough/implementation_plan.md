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
- Add `[project.optional-dependencies]` with `dev` group: pytest, pytest-asyncio, black, ruff
- Add `[project.scripts]` to expose CLI (optional)

### 0.4 Create .gitignore
```
__pycache__/
*.py[cod]
venv/
.venv/
.env
uv.lock
.pytest_cache/
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
uv add sentence-transformers
uv add faiss-cpu
uv add pydantic
uv add httpx
uv add tomli
uv add langchain
uv add langchain-ollama
uv add tqdm

# Dev dependencies
uv add --dev pytest pytest-asyncio black ruff
```

**Verify:**
```bash
uv pip freeze | grep -E "fastmcp|pymupdf|chromadb|sentence|faiss|pydantic|langchain"
```
**Exit:** All packages present. `uv run python -c "import fastmcp, fitz, chromadb"` succeeds.

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
def build_inventory() -> list[dict]:
    """Returns list of {id, pdf_filename, qa_filename, pdf_exists, qa_type}."""
    # ... glob data/*/*.pdf and data/*/*_qa.jsonl
    # For each qa.jsonl, read first line to detect type distribution
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
            text = pymupdf4llm.to_markdown(doc, pages=[page_num])
        except Exception:
            text = page.get_text("text")
        pages.append(text)
    doc.close()
    return pages

def chunk_pages(pages: list[str], chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split pages into chunks of ~chunk_size tokens with overlap."""
    # Use naive splitting by sentence boundaries
    # For each chunk, count tokens approx as len(words) * 1.3
    ...
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

### 4.1 Imports and setup
```python
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIR = Path("data/chroma_db")

# Initialize model once (lazy load on first call)
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model
```

### 4.2 ChromaDB client
```python
def get_chroma_client(persist_dir: Path = PERSIST_DIR):
    return chromadb.Client(Settings(
        persist_directory=str(persist_dir),
        anonymized_telemetry=False
    ))
```

### 4.3 Index all chunks
```python
def index_documents(chunks: list, collection_name: str = "nexla_docs"):
    """Embed chunks and store in ChromaDB."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)

    model = get_model()
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts).tolist()

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
    client.persist()
    return len(chunks)
```

### 4.4 Load existing index or re-index
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
from nexla_mcp.indexer import get_model, get_or_create_index
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
    model = get_model()
    collection = get_or_create_index()

    # Embed question
    question_embedding = model.encode([question]).tolist()[0]

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

### 6.3 Generate answer with sources
```python
def generate_answer_with_sources(question: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Returns (answer: str, sources: list[dict]).
    sources format: [{"doc_filename": str, "page_number": int, "text": str}, ...]
    """
    if not chunks:
        return "I don't know based on the provided documents.", []

    from nexla_mcp.retriever import build_context
    context = build_context(chunks)

    human_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = llm.invoke(SYSTEM_PROMPT + f"\n\n{human_prompt}")
    answer = response.content if hasattr(response, 'content') else str(response)

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
    return {
        "answer": answer,
        "sources": [s.model_dump() for s in sources]
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
import pytest
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
import pytest
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

### 8.3 Test server (integration)
File: `tests/test_server.py`
```python
import pytest
from fastmcp import Client
from nexla_mcp.server import mcp

@pytest.mark.asyncio
async def test_query_documents():
    async with Client("src/nexla_mcp/server.py") as client:
        result = await client.call_tool("query_documents", {"question": "What is this about?", "top_k": 3})
        # Verify structure
        data = result.content[0].text
        assert "answer" in data
```

### 8.4 Run tests
```bash
uv run pytest tests/ -v
```

### 8.5 Manual QA verification
Run 3 questions from `data/*/*_qa.jsonl` against the live server:
```python
# questions from qa.jsonl files:
questions = [
    "What is the primary challenge addressed by the introduction of the Linked WikiText-2 dataset?",
    "What is the top-1 accuracy of the Oracle KGLM on birthdate prediction?",
    "How many documents are in the training set of the Linked WikiText-2 Corpus?"
]
# For each, verify: answer is grounded, sources present, doc + page cited
```

**Exit:** All tests pass. 3 manual Q&A interactions succeed with source attribution.

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
  → sentence-transformers (all-MiniLM-L6-v2) → Embeddings
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

## Step 10: GitHub Repository

```bash
git init
git add .
git commit -m "Initial commit: Nexla MCP PDF Q&A server"
git branch -M main
git remote add origin https://github.com/<username>/nexla-mcp.git
git push -u origin main
# Verify repo is public
```

---

## Step 11: Final Verification

```bash
# Fresh clone test
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
                              Step 5 → Step 6 → Step 7 → Step 8 → Step 9 → Step 10 → Step 11
```

---

*Last updated: 2026-05-14*
