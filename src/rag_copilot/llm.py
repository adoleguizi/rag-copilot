"""LLM clients used by the RAG chain."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .rag_chain import LLMClient, MockLLM


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DASHSCOPE_MODEL = "qwen3-omni-flash-2025-12-01"
DEFAULT_SYSTEM_PROMPT = "你是一个严谨的 RAG 问答助手。只能根据用户提供的资料回答，不要编造。"


class JsonTransport(Protocol):
    def post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
        """Post JSON and return the decoded JSON response."""


class UrllibJsonTransport:
    """Small HTTP transport implemented with the Python standard library."""

    def post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DashScope request failed with HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DashScope request failed: {exc.reason}") from exc


@dataclass
class DashScopeLLM:
    """Alibaba Cloud Model Studio/DashScope OpenAI-compatible chat client."""

    api_key: str | None = None
    model: str = DEFAULT_DASHSCOPE_MODEL
    base_url: str = DEFAULT_DASHSCOPE_BASE_URL
    temperature: float = 0.2
    max_tokens: int | None = 800
    timeout: float = 60.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    transport: JsonTransport | None = None

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = 800,
        timeout: float = 60.0,
    ) -> "DashScopeLLM":
        return cls(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=model or os.getenv("DASHSCOPE_MODEL", DEFAULT_DASHSCOPE_MODEL),
            base_url=base_url or os.getenv("DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY. Set it before using --llm dashscope.")

        transport = self.transport or UrllibJsonTransport()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        response = transport.post_json(
            self._chat_completions_url(),
            payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        return _extract_chat_content(response)

    def _chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


def create_model(
    provider: str,
    *,
    mock_response: str = "这是一个 MockLLM 测试答案。[1]",
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = 800,
) -> LLMClient:
    """Create an LLM client by provider name."""

    if provider == "mock":
        return MockLLM(mock_response)
    if provider == "dashscope":
        return DashScopeLLM.from_env(
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _extract_chat_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected DashScope response format: {response}") from exc

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    raise RuntimeError(f"Unexpected DashScope message content format: {content}")
