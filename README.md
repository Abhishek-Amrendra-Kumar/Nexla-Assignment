# Nexla MCP — PDF Q&A Server

MCP server that answers questions from PDF documents using retrieval-augmented generation.

## Quick Start

```bash
uv sync                    # install deps
cp env.example .env        # configure API keys
uv run nexla-mcp           # start the MCP server
```

## Docker

```bash
# Lean runtime (default — no torch/sentence-transformers, uses HF Inference API for embeddings)
docker build -t nexla-mcp .

# Fat image (includes torch + sentence-transformers for local inference)
docker build --build-arg INSTALL_INDEXING=true -t nexla-mcp:indexer .

# Run (stdio transport — for MCP agent harnesses)
docker run -i --rm \
  -e HF_TOKEN=your_hf_token \
  -e LITELLM_API_KEY=your_api_key \
  -e LITELLM_BASE_URL=https://api.openai.com/v1 \
  -e LITELLM_MODEL=gpt-4o-mini \
  nexla-mcp
```

The pre-built `chroma_db/` index is copied into the image — no re-indexing needed at runtime.
`USE_LOCAL_INFERENCE` is set automatically based on the image type (false for lean, true for fat).

## Prerequisites

- Python ≥3.11
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
For Docker, the pre-built index is included in the image — indexing is not needed at runtime.

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
