# Research Notes — Nexla MCP Server Project

## 1. Assignment Overview

**Goal:** Build an MCP server exposing a Q&A tool over 4-5 provided PDFs.
AI agents connect to the server and ask natural language questions, receiving grounded, attributed answers.

**Key Requirements:**
- Document ingestion: parse + index PDFs at startup or on demand
- Q&A Tool: `query_documents` — accepts natural language question, returns grounded answer with source attribution
- Multi-document awareness: queries spanning multiple documents
- Source attribution: document name, page number, section reference
- MCP Protocol Compliance: valid MCP server, standard tool interface
- Vibe Coding README: AI tools used, prompting strategy, human vs AI contribution, reflections

**Evaluation Weights:** Vibe Coding 40% | Code Quality 35% | MCP Protocol 25%

---

## 2. MCP Framework

### FastMCP (chosen)

**Installation:**
```bash
pip install fastmcp
```

**Minimal Example (< 30 lines):**
```python
from fastmcp import FastMCP

mcp = FastMCP("My Server")

@mcp.tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()
```

**Core Concepts:**
| Concept | How It Works |
|---------|-------------|
| Server | `FastMCP("Name")` holds all tools/resources |
| Tools | Decorated with `@mcp.tool` — auto-generates JSON schemas from type hints |
| Run | `mcp.run()` handles MCP protocol (JSON-RPC 2.0) over stdio by default |
| Transport | `mcp.run(transport="http", port=8000)` for HTTP |

**Testing with client:**
```python
import asyncio
from fastmcp import Client

async def test():
    async with Client("server.py") as client:
        result = await client.call_tool("greet", {"name": "World"})
        print(result.content[0].text)

asyncio.run(test())
```

---

## 3. PDF to Text Libraries (Ranked by Accuracy)

Based on web search — benchmarks from py-pdf/benchmarks, academic paper "A Comparative Study of PDF Parsing Tools Across Diverse Document Categories" (2024).

### Top 5 (Text Extraction Accuracy)

| Rank | Library | Accuracy | Speed | Best For |
|------|---------|----------|-------|----------|
| 1 | **pypdfium2** | 97% avg | 0.003s/page | Highest accuracy, fastest |
| 2 | **PyMuPDF** | 96% avg | 0.05s/page | Excellent accuracy + tables |
| 3 | **pypdf** | 96% avg | 0.02s/page | Pure Python, no deps |
| 4 | **Apache Tika** | 95% avg | Moderate | Multi-format, enterprise |
| 5 | **pdfplumber** | 75% avg | 0.15s/page | Tables only (lower text accuracy) |

### Key Benchmark Findings
- pypdfium2 leads overall text extraction quality (97%) and speed
- PyMuPDF and pypdf tied at 96% accuracy
- pdfminer.six surprisingly low at 89% — slower too
- pdfplumber 75% for general text — but excels at table extraction

### For AI/RAG Workloads
- `pymupdf4llm` (0.12s/page) outputs excellent markdown — "sweet spot of speed and quality"
- `marker-pdf` best layout preservation but 11.3s/page — too slow for volume
- For complex documents (reports, contracts, whitepapers): PyMuPDF recommended

### Recommendation
**PyMuPDF** — best accuracy/speed balance, good table support, MIT-compatible commercial license.
**pypdfium2** if pure speed + highest accuracy are paramount.

---

## 4. Python Build Tools

### Top 5 Build Tools for Python Projects

| Tool | Type | Speed | Best For |
|------|------|-------|----------|
| **uv** | All-in-one (package + env + Python version manager) | ~10-100x faster than pip/poetry | New projects, speed-critical workflows |
| **Poetry** | Package + dependency manager | Slow (~3x slower than uv) | Library publishing, mature ecosystems |
| **PDM** | Package + dependency manager | Fast | PEP 621 standard, no venv needed |
| **Rye** | All-in-one (by Armin Ronacher) | Fast | One-stop-shop, pyenv replacement |
| **pipx** | CLI app installer (isolated envs) | Moderate | Globally installing CLI tools |

### Key Findings
- **uv** (Astral, written in Rust) is the current industry trend — replaces pip, pyenv, poetry, venv, pipx
- **Poetry** still widely used but losing ground to uv
- **Rye** from Armin Ronacher (Flask creator) — all-in-one, but uv has superseded it
- **pipx** still best for globally installing CLI Python apps separately from project dependencies

### Recommendation
**uv** — fastest, all-in-one, backed by Astral (ruff creators), strong industry adoption.

---

## 5. Required Pip Packages

Based on existing MCP PDF RAG server implementations (pdf-mcp, rag-mcp-server, pdf-rag-mcp, mcp-local-rag).

### Core Dependencies

| Package | Purpose |
|---------|---------|
| **fastmcp** | MCP server framework |
| **pymupdf** | PDF text extraction (accuracy + speed) |
| **chromadb** | Vector store for embeddings |
| **sentence-transformers** | Local embeddings (all-MiniLM-L6-v2) |
| **faiss-cpu** | Optional — faster vector search alternative to ChromaDB |
| **pydantic** | Data validation / response models |
| **httpx** | HTTP client (FastMCP dependency) |
| **tomli** | TOML parsing for config |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| **pymupdf4llm** | Markdown extraction from PDFs (better for AI) |
| **pdfplumber** | Table extraction from PDFs |
| **numpy** | Numerical operations, required by many libs |
| **tqdm** | Progress bars for indexing |

### For Local LLM (no API keys needed)

| Package | Purpose |
|---------|---------|
| **ollama** | Local LLM inference |
| **langchain** | RAG framework + retrievers |
| **langchain-community** | Community LLM integrations |

### Development Dependencies

| Package | Purpose |
|---------|---------|
| **pytest** | Testing |
| **pytest-asyncio** | Async tests |
| **black** | Code formatting |
| **ruff** | Linting (fast, Rust-based) |
| **mypy** | Type checking |

### Recommendation (Minimal Viable Stack)
```
fastmcp
pymupdf
chromadb
sentence-transformers
pydantic
faiss-cpu
pymupdf4llm
langchain
langchain-ollama  # or openai if using API keys
```

---

## 6. Existing Reference Implementations (GitHub)

These projects are highly relevant references — similar stack, MIT licensed:

1. **MBaranekTech/pdf-rag-mcp** — FastMCP + PyMuPDF + pdfplumber + ChromaDB + sentence-transformers
2. **jztan/pdf-mcp** — PyMuPDF + SQLite cache + semantic search (most popular, 21 stars)
3. **mohandshamada/RAG-MCP** — FastMCP + PyMuPDF + FAISS + LangChain
4. **rhuanca/pdf_mcpserver** — Docling + BM25 + ChromaDB hybrid search
5. **mcp-local-rag** — Multi-format (PDF, DOCX, images) + local LLM (Google GenAI)

---

*Last updated: 2026-05-14*
