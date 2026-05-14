"""Nexla MCP LLM package.

Modules:
    prompts: Hardcoded system prompts.
    inference: LLM inference and answer generation with source citations.
"""

from nexla_mcp.llm.inference import generate_answer_with_sources, llm_infer
from nexla_mcp.llm.prompts import SYSTEM_PROMPT

__all__ = ["llm_infer", "SYSTEM_PROMPT", "generate_answer_with_sources"]
