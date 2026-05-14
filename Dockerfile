# =============================================================================
# Nexla MCP Server — Dockerfile
# =============================================================================
# Multi-stage build with optional indexing support:
#   Stage 1 (builder): Install deps (runtime-only or runtime + indexing)
#   Stage 2 (runtime):  Slim image with only runtime deps + project code
#
# Build arg:
#   INSTALL_INDEXING=false  — set to "true" to include torch + sentence-transformers
#                             and enable local inference at runtime
# =============================================================================

# -------------------- Stage 1: Builder --------------------
FROM python:3.11-slim AS builder

ARG INSTALL_INDEXING=false

# Install uv (faster than pip, handles complex deps well)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Pre-install PyTorch CPU from custom index when indexing is needed
RUN if [ "$INSTALL_INDEXING" = "true" ]; then \
        uv venv /app/.venv && \
        . /app/.venv/bin/activate && \
        uv pip install torch --index-url https://download.pytorch.org/whl/cpu; \
    else \
        uv venv /app/.venv; \
    fi

# Copy source + config before installing editable package (required for build)
COPY src/ ./src/
COPY README.md pyproject.toml uv.lock* ./

# Install package: runtime-only by default, or with indexing extras
RUN . /app/.venv/bin/activate && \
    if [ "$INSTALL_INDEXING" = "true" ]; then \
        uv pip install -e ".[indexing]"; \
    else \
        uv pip install -e .; \
    fi

# -------------------- Stage 2: Runtime --------------------
FROM python:3.11-slim AS runtime

ARG INSTALL_INDEXING=false

# Install runtime-only deps (no dev tools, no uv in final image)
COPY --from=builder /app/.venv /app/.venv

WORKDIR /app

# Bundle project source into the image
COPY --from=builder /app/src ./src
COPY --from=builder /app/pyproject.toml .

# Copy pre-built ChromaDB index (built locally before docker build).
# This avoids re-indexing ~231 documents at runtime (~30min+ process).
COPY chroma_db/ /app/chroma_db/

# Copied data for list_documents  
COPY data/ /app/data/

# Create non-root user for safety
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    USE_LOCAL_INFERENCE=${INSTALL_INDEXING}

# Required environment variables (must be set at runtime)
# HF_TOKEN          — HuggingFace API token for embeddings
# LITELLM_API_KEY   — LiteLLM / OpenAI-compatible API key
# LITELLM_BASE_URL  — Base URL for LLM API (e.g., https://api.openai.com/v1)
# LITELLM_MODEL     — Model name (e.g., gpt-4o-mini)

# Data directory (mount at runtime for document corpus)
VOLUME ["/app/data"]

EXPOSE 8000

# -------------------- Default entry point --------------------
# Run as: docker run -i --rm \
#   -e HF_TOKEN=... -e LITELLM_API_KEY=... \
#   -e LITELLM_BASE_URL=... -e LITELLM_MODEL=... \
#   -v $(pwd)/data:/app/data \
#   nexla-mcp
#
# The -i flag is critical: MCP uses stdio (not HTTP) for communication.
# Agent harnesses connect stdin/stdout to the MCP protocol.
ENTRYPOINT ["python", "-m", "nexla_mcp"]
