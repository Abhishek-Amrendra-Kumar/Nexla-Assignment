import json
import fitz
from pathlib import Path


def build_inventory(data_dir: Path = Path("data")) -> list[dict]:
    """
    Scans data/ and returns a list of dicts, one per document:
    {
        "id": str,
        "pdf_filename": str,
        "qa_filename": Optional[str],
        "pdf_exists": bool,
        "page_count": int,
        "qa_types": list[str]  # e.g. ["text-only", "multimodal-t"]
    }
    """
    inventory = []
    for doc_dir in sorted(data_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name

        # Find PDF
        pdfs = list(doc_dir.glob("*.pdf"))
        pdf_path = pdfs[0] if pdfs else None

        # Find qa.jsonl
        qa_files = list(doc_dir.glob("*_qa.jsonl"))
        qa_path = qa_files[0] if qa_files else None

        # Count PDF pages
        page_count = 0
        if pdf_path:
            try:
                doc = fitz.open(str(pdf_path))
                page_count = len(doc)
                doc.close()
            except Exception:
                page_count = 0

        # Read qa types
        qa_types = []
        if qa_path:
            try:
                with open(qa_path) as f:
                    for line in f:
                        entry = json.loads(line)
                        if "type" in entry:
                            qa_types.append(entry["type"])
            except Exception:
                pass

        inventory.append(
            {
                "id": doc_id,
                "pdf_filename": pdf_path.name if pdf_path else "",
                "qa_filename": qa_path.name if qa_path else None,
                "pdf_exists": pdf_path is not None,
                "page_count": page_count,
                "qa_types": list(set(qa_types)),
            }
        )

    return inventory


if __name__ == "__main__":
    for doc in build_inventory():
        print(doc)
