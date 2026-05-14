# Nexla MCP — PDF Q&A Server

MCP server that answers questions from PDF documents using retrieval-augmented generation.

## Quick Start

```bash
uv sync                    # install deps
cp env.example .env        # configure API keys
uv run nexla-mcp           # start the MCP server
```

## CLI Helper

`CLI.sh` provides convenience commands for setup and deployment:

```bash
./CLI.sh setup <google-drive-link>  # Download data + chroma_db from Google Drive
./CLI.sh now                        # Build Docker image
./CLI.sh config                     # Show MCP client configuration
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

## MCP Client Configuration

### Claude Code / Claude Desktop

Add to your Claude Desktop config (e.g. `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "nexla-mcp": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "--init",
        "--pull=never",
        "-e",
        "HF_TOKEN",
        "-e",
        "LITELLM_API_KEY",
        "-e",
        "LITELLM_BASE_URL",
        "-e",
        "LITELLM_MODEL",
        "nexla-mcp"
      ],
      "env": {
        "HF_TOKEN": "hf_***",
        "LITELLM_API_KEY": "sk_***",
        "LITELLM_BASE_URL": "https://api.minimax.io/v1",
        "LITELLM_MODEL": "MiniMax-M2.7-highspeed"
      }
    }
  }
}
```

### OpenCode

Add to your OpenCode MCP client config:

```json
"nexla-mcp": {
  "type": "local",
  "command": [
    "docker",
    "run",
    "-i",
    "--rm",
    "--init",
    "--pull=never",
    "-e",
    "HF_TOKEN",
    "-e",
    "LITELLM_API_KEY",
    "-e",
    "LITELLM_BASE_URL",
    "-e",
    "LITELLM_MODEL",
    "nexla-mcp"
  ],
  "environment": {
    "HF_TOKEN": "hf_***",
    "LITELLM_API_KEY": "sk_***",
    "LITELLM_BASE_URL": "https://api.minimax.io/v1",
    "LITELLM_MODEL": "MiniMax-M2.7-highspeed"
  }
}
```

> Replace `HF_TOKEN` and `LITELLM_API_KEY` with your actual keys.

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

## Vibe Coding Setup

### AI Coding Tools Used

- **ChatGPT** — Used to convert the problem-statement PDF into a Markdown file for easier consumption. Also used for initial research and brainstorming during planning.
- **OpenCode** — Primary AI coding assistant throughout the rest of development. Acts as an orchestration layer that delegates specialized tasks (code search, documentation lookup, UI design, code review) to domain-specific sub-agents.

### Workflow & Prompting Strategy

The development followed a structured phased approach:

1. **PDF → Markdown**: Fed the problem-statement PDF to ChatGPT to convert it into a structured Markdown file for easier reference.
2. **Planning Phase**: Asked OpenCode to analyze the requirements, divide the work into logical phases, and produce a `phases.md` plan. Also created `AGENTS.md` to codify agent delegation guidelines.
3. **Research Phase**: Spawned subagents in parallel to research the best tools and libraries available for each phase — PDF parsing, embeddings, vector stores, LLM routing, MCP framework. The output fed into a dependency analysis and was folded into the plan.
4. **Implementation Planning**: Used research findings to produce a concrete `implementation-plan.md` covering architecture, data flow, and tool choices.
5. **Plan Refinement**: Iteratively refined the plan in stages — making it more concrete and actionable before any code was written.
6. **Execution**: Executed the plan incrementally. This included iterative dependency changes (`transformers`/`sentence-transformers` → `fastembed` → HuggingFace Inference API after a subagent researched the best remote models), writing tests for each module, generating `.env` and API keys, and correcting retrieval logic. Refactored the codebase multiple times to split indexer and inference logic into separate files and consolidated constants into a single `config.py`.

What worked:
- **Phased planning** — freezing a concrete plan before writing code prevented major rework later.
- **Parallel research** — subagents investigating different tools simultaneously saved significant time.
- **Iterative refinement** — tweaking the plan in stages let the approach converge toward something solid.

What did not work:
- Over-delegating simple changes — explaining a small fix to the agent sometimes took longer than just making the edit directly.
- Context management in longer sessions required active compression to stay within context limits.

### Human vs AI Contribution

| Area | AI Contribution | Human Override/Correction |
|---|---|---|
| Requirements parsing | ChatGPT converted PDF → Markdown | Human reviewed and prioritized |
| Architecture & planning | Proposed chunking/embedding pipeline and phases | Human approved final architecture |
| Tool/library research | Subagents surveyed options in parallel | Human selected final stack |
| Dependency stack | Proposed `transformers`/`sentence-transformers` → `fastembed` → HuggingFace Inference API after subagent model research | Human validated latency trade-offs and API setup |
| PDF parsing | Generated extraction logic | Corrected page boundary handling |
| MCP tool schema | Auto-generated from Pydantic models | Refined descriptions and input validation |
| Tests | Wrote tests for each module | Human reviewed coverage and edge cases |
| Environment & config | Generated `.env` templates and API key setup | Human corrected retrieval logic and validated config |
| Refactoring | Suggested splitting indexer/inference into separate files and creating `config.py` | Human approved final structure |
| Docker & client config | Created Dockerfile and Claude/OpenCode client configurations | Human tested and adjusted mounts/ports |
| README drafting | Generated initial structure | Wrote Vibe Coding section |
| Bug fixes | Identified issues via oracle subagent | Human approved and integrated |

### Reflection on AI Tooling

AI tooling was most effective when the task was well-bounded with clear success criteria, the phased planning approach worked precisely because each phase had an explicit definition of done. Research tasks (comparing libraries, finding dependencies) benefited heavily from parallel subagent execution.

The weakest link was in tasks requiring judgment about trade-offs that depended on unstated context like whether to prioritize local inference vs. API-based for cost reasons. Those required more human feedback to get it working.

For a project like this, the AI tooling was net-positive it accelerated research, reduced boilerplate, and helped navigate unfamiliar libraries. The key was treating it as small tasks with phases instead of treating as a fixed singular big task.

## License

MIT
