# Nexla MCP — PDF Q&A Server

MCP server that answers questions from PDF documents using retrieval-augmented generation.

## Quick Start

```bash
uv sync                    # install deps
cp env.example .env        # configure API keys
uv run nexla-mcp           # start the MCP server
```

## Prerequisites

- Python ≥3.10
- [uv](https://docs.astral.sh/uv/) (package manager)

## Configuration

Copy `env.example` to `.env` and set:

| Variable | Description |
|---|---|
| `HF_TOKEN` | Hugging Face API token (for embeddings) |
| `LITELLM_API_KEY` | LLM provider API key |
| `LITELLM_BASE_URL` | LLM provider base URL (e.g. `https://api.openai.com/v1`) |
| `LITELLM_MODEL` | Model name (e.g. `gpt-4o-mini`) |

## Data Format

Place PDFs in `data/<id>/` directories alongside a `*_qa.jsonl` file:

```
data/
├── 1706.03762/
│   ├── 1706.03762.pdf
│   └── 1706.03762_qa.jsonl
└── 2310.06825/
    ├── 2310.06825.pdf
    └── 2310.06825_qa.jsonl
```

On first run, the server indexes all PDFs into ChromaDB (persisted to `chroma_db/`).

## Tools

All tools are exposed via the MCP protocol (`fastmcp`):

| Tool | Description |
|---|---|
| `query_documents` | Ask a question, get a grounded answer with sources |
| `list_documents` | List all indexed documents |
| `search_documents` | Raw semantic search (no LLM) |
| `get_document_summary` | Metadata for a specific document |

## Development

```bash
uv sync                       # sync environment
uv run python tests/test_indexer.py   # run indexer tests
uv run ruff check . && uv run ruff format .  # lint & format
```

## License

MIT
