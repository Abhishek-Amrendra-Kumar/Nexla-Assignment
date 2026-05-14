from fastmcp import FastMCP
from nexla_mcp.retriever import retrieve
from nexla_mcp.llm import generate_answer_with_sources

# Global collection — initialize on startup
_collection = None


def get_collection():
    global _collection
    if _collection is None:
        from nexla_mcp.indexer import get_index

        _collection = get_index()
    return _collection


mcp = FastMCP("Nexla PDF Q&A Server")


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
    return {"answer": answer, "sources": sources}


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
