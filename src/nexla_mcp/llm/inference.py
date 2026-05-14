from openai import OpenAI
import time
from nexla_mcp.llm.prompts import SYSTEM_PROMPT
from nexla_mcp.env_secrets import Secrets


class LLM:
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, model: str, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return resp.choices[0].message.content


_llm = None


def _get_llm() -> LLM:
    global _llm
    if _llm is None:
        secrets = Secrets()
        _llm = LLM(api_key=secrets.get_api_key(), base_url=secrets.get_base_url())
    return _llm


def llm_infer(messages: list[dict], model: str = None) -> str:
    """Simple OpenAI-client inference - returns response content."""
    if model is None:
        model = Secrets().get_model()
    start = time.perf_counter()
    res = _get_llm().chat(model=model, messages=messages)
    dur = time.perf_counter() - start
    print(f"Inference duration: {dur:.4f}s")
    return res


def generate_answer_with_sources(
    question: str, chunks: list[dict]
) -> tuple[str, list[dict]]:
    """Returns (answer: str, sources: list[dict])."""
    if not chunks:
        return "I don't know based on the provided documents.", []

    from nexla_mcp.retriever import build_context

    context = build_context(chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    answer = llm_infer(messages)
    sources = [
        {
            "doc_filename": c["doc_filename"],
            "page_number": c["page_number"],
            "text": c["text"],
        }
        for c in chunks
    ]
    return answer, sources
