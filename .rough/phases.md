# Project Phases — Nexla MCP Server

> Derived from `ps.md`. Each phase is a discrete, actionable stage. Complete phases in order unless stated otherwise.

---

## Phase 0: Project Scaffold

**Goal:** Establish the project structure and tooling.

- [ ] Initialize Python project with `uv init`
- [ ] Configure `pyproject.toml` with project metadata
- [ ] Set up `.gitignore` (venv, __pycache__, .env, uv.lock)
- [ ] Create directory structure:
  ```
  src/nexla_mcp/
  data/           # already exists — PDFs + qa.jsonl
  tests/
  .rough/         # already exists
  ```
- [ ] Verify `uv run python -c "print('ok')"` works

**Exit criterion:** Project runs, imports work, `uv sync` succeeds from scratch.

---

## Phase 1: Dependency Installation

**Goal:** Install all required packages using `uv add`.

| Package | Purpose |
|---------|---------|
| `fastmcp` | MCP server framework |
| `pymupdf` | PDF text extraction |
| `chromadb` | Vector store |
| `sentence-transformers` | Local embeddings (all-MiniLM-L6-v2) |
| `faiss-cpu` | Fast vector search |
| `pydantic` | Request/response models |
| `httpx` | HTTP client |
| `tomli` | TOML parsing |
| `pymupdf4llm` | Markdown extraction from PDFs (better for AI) |
| `langchain` | RAG framework |
| `langchain-ollama` | Local LLM (Ollama) |
| `tqdm` | Progress bars for indexing |

Dev dependencies:
| Package | Purpose |
|---------|---------|
| `black` | Formatting |
| `ruff` | Linting |

**Exit criterion:** `uv pip freeze` shows all packages installed. No import errors.

---

## Phase 2: PDF Document Discovery & Data Understanding

**Goal:** Understand the provided data — 45+ PDFs and their qa.jsonl counterparts.

- [ ] List all PDF files under `data/<id>/`
- [ ] List all qa.jsonl files — note their fields: `question`, `answer`, `type` (text-only | multimodal-t | meta-data), `evidence`
- [ ] Sample 2–3 PDFs to understand document types (contracts? reports? research papers? government documents?)
- [ ] Identify document categories: Financial, Legal, Scientific, Government, etc.
- [ ] Note: Each PDF has a corresponding qa.jsonl — the qa pairs reveal what questions the system should be able to answer
- [ ] Decide: will the system ingest ALL PDFs or a subset? (recommend all for demo)

**Exit criterion:** Document inventory spreadsheet (or markdown table) listing each PDF id, filename, page count (approx), and document type.

---

## Phase 3: PDF Text Extraction (Ingestion)

**Goal:** Extract raw text (and optionally markdown) from all PDFs.

- [ ] Write `src/nexla_mcp/pdf_processor.py`
  - Load PDFs using PyMuPDF
  - Extract text page-by-page with metadata (page number, section if detectable)
  - Use `pymupdf4llm.to_markdown()` for markdown output (better for AI)
  - Store extracted text per page with document name + page index
- [ ] Implement chunking strategy:
  - Chunk size: ~500 tokens (configurable)
  - Chunk overlap: ~50 tokens
  - Preserve metadata: document name, page number, chunk index
- [ ] Handle extraction failures gracefully (corrupt PDFs, scanned/image PDFs)
- [ ] Optional: detect and flag scanned PDFs (require OCR)

**Exit criterion:** All PDFs are extracted and chunked. Run `uv run python -c "from nexla_mcp.pdf_processor import extract_all; print(len(extract_all()))"` returns total chunk count > 0.

---

## Phase 4: Embedding & Vector Store

**Goal:** Index all chunks into a vector store for semantic search.

- [ ] Write `src/nexla_mcp/indexer.py`
  - Initialize sentence-transformers (all-MiniLM-L6-v2)
  - Embed each chunk
  - Store in ChromaDB (collection per document or single collection)
  - Include metadata: document name, page number, chunk text, source file
- [ ] Implement `index_documents()` — full re-index
- [ ] Implement `index_document(doc_id)` — single document update
- [ ] Persist vector store to disk (ChromaDB persistence)
- [ ] On startup: load existing index OR re-index if empty

**Exit criterion:** Querying the vector store returns relevant chunks for a test query.

---

## Phase 5: Retriever (Context Assembly)

**Goal:** Given a user question, retrieve the most relevant chunks.

- [ ] Write `src/nexla_mcp/retriever.py`
  - Accept `question: str`, `top_k: int`
  - Query vector store for top-k similar chunks
  - Optionally: combine with keyword search (BM25) for hybrid retrieval
  - Assemble retrieved chunks into a context string
  - Include source attribution for each chunk: document name, page number
- [ ] Handle multi-document queries (chunks from different documents)
- [ ] Implement `retrieve(question, top_k)` function

**Exit criterion:** `retrieve("test question", top_k=3)` returns list of chunks with source metadata.

---

## Phase 6: LLM Answer Generation

**Goal:** Generate grounded answers from retrieved context.

- [ ] Write `src/nexla_mcp/llm.py`
  - Choose LLM: Ollama (local, free) OR OpenAI API key
  - Implement `generate_answer(question: str, context: str) -> str`
  - System prompt: answer only from provided context, cite sources
  - Fallback: if context is empty, return "I don't know based on the documents."
- [ ] Implement `generate_answer_with_sources()` — returns answer + list of sources
- [ ] Implement `models.py` with Pydantic models:
  - `QueryRequest`, `QueryResponse`
  - `Source` (document, page, section, text)

**Exit criterion:** LLM returns a coherent answer grounded in retrieved context for a test question.

---

## Phase 7: MCP Server (FastMCP)

**Goal:** Expose tools via the MCP protocol.

- [ ] Write `src/nexla_mcp.mcp.py`
  - Initialize FastMCP server
  - Register `query_documents` tool:
    - Input: `question: str`, `top_k: int = 5`
    - Output: `answer: str`, `sources: list[dict]`
  - Register optional tools:
    - `list_documents` — returns document names and IDs
    - `get_document_summary` — metadata for a specific document
    - `search_documents` — raw semantic search without LLM answer
- [ ] On startup: auto-index all documents if index is empty
- [ ] Server runs via `mcp.run()` (stdio transport for local MCP clients)

**Exit criterion:** `uv run python -m nexla_mcp.mcp` starts without errors. MCP client can call `query_documents`.

---

## Phase 8: Testing

**Goal:** Verify the system end-to-end works.

- [ ] Write tests in `tests/`:
  - `test_pdf_processor.py` — extraction tests (mock PDF or use real small PDF)
  - `test_retriever.py` — retrieval tests
  - `test_server.py` — integration tests with FastMCP client
- [ ] Run `uv run python tests/test_indexer.py` — test passes
- [ ] Manual test: ask 3 questions from the qa.jsonl files (use real questions from the data)
  - Verify answers are grounded in document content
  - Verify source attribution is present

**Exit criterion:** At least 3 manual Q&A interactions succeed with source attribution.

---

## Phase 9: README.md

**Goal:** Document the project thoroughly (this is 40% of evaluation — taken seriously).

### Required Sections

- [ ] **Setup Instructions** — step-by-step install from scratch
  ```
  uv sync
  uv run python -m nexla_mcp.mcp
  ```
- [ ] **Architecture Overview** — system diagram + component description
  - Document ingestion → chunking → embedding → vector store → retrieval → LLM → MCP tool
- [ ] **Tool Documentation** — each MCP tool
  - Name, description, input schema, output schema, example calls
- [ ] **Example Interaction Log** — at least 3 Q&A examples with source attribution
  - Use real questions from `data/*/*_qa.jsonl`
  - Show question, answer, and source references
- [ ] **Vibe Coding Section** (most important — 40% of evaluation weight):
  - **AI Coding Tools Used**: which tools (Cursor, Copilot, Claude Code, Windsurf, etc.), versions
  - **Prompting Strategy**: how prompted/directed the AI, what worked, what didn't
  - **Human vs AI Contribution**: where relied on AI, where corrected or overrode AI output
  - **Reflection on AI Tooling**: perspective on how AI tooling fits into software engineering workflows

### Style Notes
- "There are no 'correct' answers" — be honest and self-aware
- Focus on: self-awareness, intentionality, engineering judgment
- Not about maximizing AI usage — about being thoughtful

**Exit criterion:** README is complete, accurate, and installable from scratch by a stranger.

---

## Phase 10: GitHub Repository Setup

**Goal:** Prepare public GitHub repo for submission.

- [ ] Initialize git repo (if not already)
- [ ] Create `.gitignore` (venv, .env, __pycache__, uv.lock)
- [ ] Ensure `pyproject.toml` is complete with all metadata
- [ ] Remove `.rough/` from repo (or add to .gitignore)
- [ ] Create initial commit
- [ ] Push to public GitHub repo
- [ ] Verify repo is public and cloneable

**Note:** The assignment requires the repo to be **public**.

---

## Phase 11: Final Verification & Submission

**Goal:** Ensure everything works and submit.

- [ ] Clone fresh to a temp directory — verify `uv sync && uv run python -m nexla_mcp.mcp` works from scratch
- [ ] Test with MCP client (e.g., Claude Desktop or any MCP-compatible client)
- [ ] Verify `query_documents` returns answers with source attribution
- [ ] Verify 3+ example Q&As are documented in README
- [ ] Confirm repo is public
- [ ] Reply to email thread with repo URL
  - Subject: `[Software Engineer-AI Take-Home] Your Name`

---

## Phase Dependencies

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6 ──► Phase 7 ──► Phase 8 ──► Phase 9 ──► Phase 10 ──► Phase 11
                   │            │            │            │
                   │            └────────────┴────────────┘
                   │                 (can overlap: Phase 3 & 4 can run in parallel after Phase 2)
                   │
                   └──────────────────────────────────────────────────────────► Phase 9 (README can be written progressively)
```

**Parallelization opportunity:** Phase 3 (extraction) and Phase 4 (indexing) can be pipelined — once extraction works, indexing can be developed independently. Phase 5 (retriever) depends on Phase 4. Phase 6 (LLM) depends on Phase 5. Phase 7 (server) depends on Phases 5 & 6.

---

## Summary: What's NOT Required (per ps.md)

- ✅ DO NOT: production deployment, CI/CD, cloud hosting
- ✅ DO NOT: specific technologies — any reasonable stack is fine
- ✅ DO NOT: perfect solution — clean + documented + thoughtful beats sophisticated + poorly explained
- ✅ DO NOT: spend more than 3–4 hours (stay close to estimate)

---

*Last updated: 2026-05-14*
