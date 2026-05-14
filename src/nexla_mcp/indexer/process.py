from datetime import datetime, timezone
from pathlib import Path

from nexla_mcp.config import DOC_LIMIT, INDEX_BATCH_SIZE
from nexla_mcp.indexer.hf import encode_texts
from nexla_mcp.indexer.manifest import (
    _compute_file_hash,
    _load_manifest,
    _save_manifest,
    get_chroma_client,
)


def _chunk_batches(chunks: list, batch_size: int):
    """Yield successive batch_size-sized chunks from the list."""
    for i in range(0, len(chunks), batch_size):
        yield chunks[i : i + batch_size]


def index_documents(
    chunks: list,
    collection_name: str = "nexla_docs",
    batch_size: int = INDEX_BATCH_SIZE,
):
    """Embed chunks in batches and store incrementally in ChromaDB."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)
    total_inserted = 0

    for batch in _chunk_batches(chunks, batch_size):
        texts = [c.text for c in batch]
        embeddings = encode_texts(texts, prompt_name="document")

        metadatas = [
            {
                "doc_id": c.doc_id,
                "doc_filename": c.doc_filename,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
            }
            for c in batch
        ]

        ids = [f"{c.doc_id}_{c.page_number}_{c.chunk_index}" for c in batch]

        # Uniqueness check: skip IDs that already exist in the collection
        existing = collection.get(ids=ids, include=[])
        existing_ids = set(existing.get("ids", []))
        if existing_ids:
            print(
                f"[WARN] Skipping {len(existing_ids)} duplicate chunk IDs: {sorted(existing_ids)[:3]}..."
            )

        new_ids = [id for id in ids if id not in existing_ids]
        if not new_ids:
            continue

        # Filter embeddings, texts, metadatas to only new IDs
        new_indices = [i for i, id in enumerate(ids) if id not in existing_ids]
        filtered_embeddings = [embeddings[i] for i in new_indices]
        filtered_texts = [texts[i] for i in new_indices]
        filtered_metadatas = [metadatas[i] for i in new_indices]

        collection.add(
            ids=new_ids,
            embeddings=filtered_embeddings,
            documents=filtered_texts,
            metadatas=filtered_metadatas,
        )
        total_inserted += len(new_ids)

    return total_inserted


def _delete_document_chunks(doc_id: str, collection_name: str = "nexla_docs"):
    """Remove all chunks for a given document from ChromaDB."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)
    collection.delete(where={"doc_id": doc_id})


def get_index(collection_name: str = "nexla_docs"):
    """Return the ChromaDB collection — no indexing logic."""
    return get_chroma_client().get_or_create_collection(name=collection_name)


def build_index(data_dir: Path = Path("data")):
    """On startup: incremental index — only process new or changed PDFs."""
    from nexla_mcp.pdf_processor import extract_document

    manifest = _load_manifest()
    collection = get_index()

    # Scan filesystem for all PDFs
    pdf_paths = sorted(data_dir.glob("*/*.pdf"), key=lambda x: int(x.parent.name))
    pdf_paths = pdf_paths[:DOC_LIMIT]
    docid_filepath_tuple = [(p.parent.name, p) for p in pdf_paths]

    # MIGRATION: if ChromaDB has data but no manifest, build manifest from
    # existing index without re-embedding. This avoids costly re-indexing on
    # the first run after this feature is introduced.
    if not manifest and collection.count() > 0:
        print("[MIGRATE] No manifest but ChromaDB has data. Building manifest...")
        all_data = collection.get(include=["metadatas"])
        indexed_doc_ids = set()
        for meta in all_data.get("metadatas") or []:
            if meta and "doc_id" in meta:
                indexed_doc_ids.add(meta["doc_id"])

        for doc_id in indexed_doc_ids:
            pdf_candidates = list(data_dir.glob(f"{doc_id}/*.pdf"))
            if pdf_candidates:
                current_hash = _compute_file_hash(pdf_candidates[0])
                doc_data = collection.get(where={"doc_id": doc_id}, include=[])
                chunk_count = len(doc_data["ids"]) if doc_data else 0
                manifest[doc_id] = {
                    "content_hash": current_hash,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                    "chunk_count": chunk_count,
                }
            else:
                print(f"[REMOVE] {doc_id} (orphaned in ChromaDB)")
                _delete_document_chunks(doc_id)

        _save_manifest(manifest)
        print(f"[MIGRATE] Done. Manifest covers {len(manifest)} docs.")
        return

    new_or_changed = []
    skipped = []

    for doc_id, pdf_path in docid_filepath_tuple:
        current_hash = _compute_file_hash(pdf_path)

        if doc_id in manifest and manifest[doc_id].get("content_hash") == current_hash:
            skipped.append(doc_id)
            continue

        # Changed or new — delete old chunks if any, then queue for re-index
        if doc_id in manifest:
            print(f"[RE-INDEX] {doc_id} (hash changed)")
            _delete_document_chunks(doc_id)
        else:
            print(f"[INDEX] {doc_id} (new)")

        chunks = extract_document(pdf_path)
        new_or_changed.append((doc_id, current_hash, chunks))

    # Remove docs from index that no longer exist on filesystem
    found_doc_ids = {x[0] for x in docid_filepath_tuple}
    for doc_id in list(manifest.keys()):
        if doc_id not in found_doc_ids:
            print(f"[REMOVE] {doc_id} (no longer on disk)")
            _delete_document_chunks(doc_id)
            del manifest[doc_id]

    # Index all new/changed documents
    total_docs = len(new_or_changed)
    for i, (doc_id, current_hash, chunks) in enumerate(new_or_changed, start=1):
        print(f"({i}/{total_docs}) Processing {doc_id}...")
        if chunks:
            index_documents(chunks)
        manifest[doc_id] = {
            "content_hash": current_hash,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(chunks),
        }
        _save_manifest(manifest)

    if skipped:
        print(f"[SKIP] {len(skipped)} unchanged documents: {skipped}")


if __name__ == "__main__":
    build_index()
