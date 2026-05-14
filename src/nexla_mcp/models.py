from pydantic import BaseModel
from typing import Optional


class Source(BaseModel):
    doc_filename: str
    page_number: int
    section: Optional[str] = None
    text: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
