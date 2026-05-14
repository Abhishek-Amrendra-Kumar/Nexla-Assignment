# Project Agents & Guidelines

> This file documents agent behavior, tooling rules, and project conventions for the Nexla MCP server project.

---

## 1. uv Instead of pip — DO NOT Rules

**Use `uv` for ALL package management. Never reach for `pip` directly.**

| DO NOT (pip) | DO THIS (uv) |
|---|---|
| `pip install <pkg>` | `uv add <pkg>` |
| `pip install -r requirements.txt` | `uv add -r requirements.txt` |
| `pip install -e .` | `uv add -e .` |
| `pip uninstall <pkg>` | `uv remove <pkg>` |
| `pip freeze` | `uv pip freeze` |
| `pip show <pkg>` | `uv pip show <pkg>` |
| `pip list` | `uv pip freeze` |
| `python script.py` | `uv run python script.py` |
| `pip install --upgrade <pkg>` | `uv add --upgrade <pkg>` |
| `venv` + manual activation | `uv venv && source .venv/bin/activate` |

### Core uv Commands Reference

```bash
# Project bootstrap
uv init                                    # scaffold new project
uv sync                                    # sync lock file with environment (after pull/clone)

# Dependency management
uv add <pkg>                              # install + update pyproject.toml + lock
uv add -e .                               # editable install (project itself)
uv add -r requirements.txt                 # bulk add from requirements file
uv remove <pkg>                           # uninstall + update lock

# Running code
uv run python script.py                    # run in managed env
uv run python tests/test_indexer.py        # run tests
uv run -- python script.py --arg value    # explicit separator if args conflict

# Inspection
uv tree                                    # dependency tree
uv pip freeze                              # installed packages
uv python list                            # available Python versions
uv python pin 3.12                        # pin project Python version

# Tool runner (CLI apps in isolated envs)
uv tool run <cli-tool>                    # run CLI tool without installing globally
uvx <cli-tool>                            # one-shot equivalent of above
```

### Pro Tips
- `uv run --refresh` — force re-install dependencies
- `uv pip install <pkg> --dry-run` — preview without installing
- `uv python find 3.12` — locate a specific Python version

### Why uv
- 10-100x faster than pip/poetry
- Single tool replaces: pip, venv, pyenv, poetry, pipx
- Deterministic locked builds via `uv.lock`
- No `requirements.txt` needed (but supported)

---

## 2. Project Architecture

```
nexla-mcp/                    # project root
├── .rough/                   # temporary research & scratch notes
├── data/                     # PDF documents + Q&A pairs
│   └── <id>/
│       ├── <id>_qa.jsonl    # Q&A pairs (question, answer, type, evidence)
│       └── <arxiv_id>.pdf   # source PDF
├── src/
│   └── nexla_mcp/
│       ├── __init__.py
│       ├── server.py         # FastMCP server entry point
│       ├── pdf_processor.py # PDF ingestion + text extraction
│       ├── indexer.py       # Chunking + embedding + vector store
│       ├── retriever.py     # Similarity search + context assembly
│       ├── llm.py           # LLM call for answer generation
│       └── models.py        # Pydantic request/response models
├── tests/
│   └── ...
├── pyproject.toml
├── uv.lock
└── README.md
```

### Data Convention
- Each document lives in `data/<numeric_id>/`
- Contains: one PDF + one `*_qa.jsonl` file
- `qa.jsonl` fields: `question`, `answer`, `type` (text-only | multimodal-t | meta-data), `evidence`

---

## 3. Agent Behavior Guidelines

### When Delegating to Specialists

| Task | Agent | Notes |
|------|-------|-------|
| Code search / pattern finding | `@explorer` | Glob, grep, AST queries. 2x faster, 1/2 cost |
| Library docs / API reference | `@librarian` | Use for unfamiliar or frequently-changing libraries (React, Next.js, AI SDKs) |
| Major decisions / architecture / code review | `@oracle` | Complex debugging, trade-offs, simplification |
| UI/UX polish | `@designer` | User-facing components |
| Bounded implementation / test writing | `@fixer` | < 20 lines / single file → do it yourself. Multi-file / task → delegate |
| Multi-model consensus for high-stakes decisions | `@council` | Not for routine work |

### General Rules
- **Delegate when:** Explaining > doing, parallelization helps, bounded task
- **Don't delegate when:** Single small change, need full context anyway, tight integration with current work
- **Context reuse:** Reuse specialist sessions for related work. Start fresh if unrelated.
- **Subtask vs named agent:** Use `subtask` for focused bounded research, no architectural decisions needed

### Workflow
1. Understand request — clarify ambiguities before acting
2. Choose path — quality, speed, cost, reliability
3. Delegate or execute — prefer parallel where independent
4. Integrate results — verify completeness
5. Route validation — @designer for UI, @oracle for code review, @fixer for tests

---

## 4. MCP Server Tool Specification

### Primary Tool: `query_documents`

**Input:**
```python
question: str  # natural language question
top_k: int = 5  # number of results (optional)
```

**Output:**
```python
answer: str           # grounded answer
sources: list[dict]   # [{"document": str, "page": int, "section": str, "text": str}]
```

### Secondary Tools (optional)
- `list_documents` — returns document names and IDs
- `get_document_summary` — returns metadata about a specific document
- `search_documents` — raw keyword/semantic search without LLM answer

---

## 5. Development Workflow

```bash
# First time setup
uv sync                         # create venv + install all deps from lock

# After pulling changes
uv sync                        # sync environment to lock file

# Add a dependency
uv add <pkg>

# Run the server
uv run python -m nexla_mcp.server

# Run tests
uv run python tests/test_indexer.py
```

### File Edit Guidelines
- Prefer `@fixer` for multi-file changes, especially test files
- Prefer doing directly for single small edits (< 20 lines)
- Always use `ast_grep_replace` for safe cross-file pattern replacement

---

## 6. Code Quality with Ruff

**ALL Python code must be formatted and linted with `ruff`.**

Ruff is the project's primary linter and formatter. Use it for:
- Formatting: `uv run ruff format <files>`
- Linting: `uv run ruff check <files> --fix`
- Both: `uv run ruff check <files> --fix && uv run ruff format <files>`

Always run ruff before committing or submitting changes.

---

*Last updated: 2026-05-14*
