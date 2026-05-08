"""Prompt construction for grounded RAG answers."""

from __future__ import annotations

from collections.abc import Sequence

from .models import RetrievedChunk


NO_EVIDENCE_MESSAGE = "找不到依据"


def build_rag_prompt(question: str, hits: Sequence[RetrievedChunk]) -> str:
    """Build the prompt sent to the LLM from retrieved chunks."""

    return f"""你是一个单主题资料问答助手。

请严格遵守规则：
1. 只能根据【资料】回答问题，不要编造资料外的信息。
2. 如果资料不足以回答，只回答“{NO_EVIDENCE_MESSAGE}”。
3. 回答中需要使用 [1]、[2] 这样的编号标注依据。
4. 如果问题要求列举，请尽量完整列出资料中的相关项目，不要用“等”等省略表达。
5. 答案要简洁、直接。

【资料】
{format_retrieved_context(hits)}

【问题】
{question}

【答案】"""


def format_retrieved_context(hits: Sequence[RetrievedChunk]) -> str:
    """Format retrieved chunks as numbered context blocks."""

    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        title = chunk.metadata.get("title")
        title_line = f"title={title}\n" if title else ""
        blocks.append(
            f"[{index}] source={chunk.source} chunk_id={chunk.id} score={hit.score:.4f}\n"
            f"{title_line}"
            f"{chunk.text.strip()}"
        )
    return "\n\n".join(blocks)
