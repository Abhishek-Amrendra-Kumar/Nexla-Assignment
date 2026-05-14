from nexla_mcp.indexer import encode_texts, get_index


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
    collection = get_index()

    # Embed question using fastembed via encode_texts
    question_embedding = encode_texts([question], prompt_name="query")[0]

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Parse results
    chunks = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        chunks.append(
            {
                "text": results["documents"][0][i],
                "doc_id": metadata["doc_id"],
                "doc_filename": metadata["doc_filename"],
                "page_number": metadata["page_number"],
                "chunk_index": metadata["chunk_index"],
                "score": results["distances"][0][i],
            }
        )
    return chunks


def build_context(chunks: list[dict]) -> str:
    """Assembles chunks into a context string with citations."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i + 1}] {chunk['doc_filename']} (page {chunk['page_number']}):\n{chunk['text']}"
        )
    return "\n\n".join(context_parts)
