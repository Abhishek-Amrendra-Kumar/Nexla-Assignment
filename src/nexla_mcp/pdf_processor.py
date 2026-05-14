import re
import fitz  # PyMuPDF
from pathlib import Path
from pydantic import BaseModel


class PageChunk(BaseModel):
    doc_id: str
    doc_filename: str
    page_number: int  # 1-indexed
    chunk_index: int
    text: str
    token_count: int


def extract_text_from_pdf(pdf_path: Path) -> list[str]:
    """Returns list of page texts from a PDF using fitz (fast) with pymupdf4llm fallback."""
    doc = fitz.open(str(pdf_path))
    page_count = len(doc)

    # Fast path: use fitz native text extraction for all pages at once
    pages = []
    for page_num in range(page_count):
        page = doc[page_num]
        text = page.get_text("text")
        pages.append(text)

    doc.close()
    return pages


def chunk_pages(
    pages: list[str], chunk_size: int = 500, overlap: int = 50
) -> list[str]:
    """
    Split a list of page texts into chunks of ~chunk_size tokens with overlap.

    Tokens are estimated as len(words) * 1.3.
    Chunks are split at sentence boundaries (period + space) where possible.

    Returns list of chunk strings.
    """

    def split_into_sentences(text: str) -> list[str]:
        """Split text on sentence boundaries (., !, ?) while preserving the delimiter."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def estimate_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)

    all_chunks = []

    for page_text in pages:
        sentences = split_into_sentences(page_text)
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)

            if current_tokens + sentence_tokens <= chunk_size:
                current_chunk += (" " + sentence).strip()
                current_tokens += sentence_tokens
            else:
                # Emit current chunk if non-empty
                if current_chunk:
                    all_chunks.append(current_chunk)
                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    # Backtrack: include last ~overlap tokens into next chunk
                    words = current_chunk.split()
                    overlap_words = " ".join(words[-int(overlap / 1.3) :])
                    current_chunk = (overlap_words + " " + sentence).strip()
                    current_tokens = estimate_tokens(current_chunk)
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens

        # Don't forget the last chunk
        if current_chunk:
            all_chunks.append(current_chunk)

    return all_chunks


def extract_document(pdf_path: Path) -> list[PageChunk]:
    """Extract and chunk a single PDF. Returns all chunks."""
    doc_id = pdf_path.parent.name
    pages = extract_text_from_pdf(pdf_path)
    chunks = []
    for page_idx, page_text in enumerate(pages):
        page_chunks = chunk_pages([page_text])
        for chunk_idx, chunk in enumerate(page_chunks):
            chunks.append(
                PageChunk(
                    doc_id=doc_id,
                    doc_filename=pdf_path.name,
                    page_number=page_idx + 1,  # 1-indexed page number
                    chunk_index=chunk_idx,  # per-page chunk index (avoids ID collisions)
                    text=chunk,
                    token_count=int(len(chunk.split()) * 1.3),
                )
            )
    return chunks


def extract_all_documents(data_dir: Path = Path("data")) -> list[PageChunk]:
    """Extract and chunk all PDFs. Returns all chunks."""
    all_chunks = []
    for pdf_path in sorted(data_dir.glob("*/*.pdf")):
        all_chunks.extend(extract_document(pdf_path))
    return all_chunks


if __name__ == "__main__":
    all_chunks = extract_all_documents()
    print(f"Total chunks: {len(all_chunks)}")
