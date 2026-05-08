from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_copilot.llm import DEFAULT_DASHSCOPE_BASE_URL, DashScopeLLM, create_model
from rag_copilot.rag_chain import MockLLM


class FakeTransport:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any], dict[str, str], float]] = []

    def post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
        self.calls.append((url, payload, headers, timeout))
        return self.response


class DashScopeLLMTests(unittest.TestCase):
    def test_generate_posts_openai_compatible_chat_completion_request(self) -> None:
        transport = FakeTransport({"choices": [{"message": {"content": "基于资料回答。[1]"}}]})
        llm = DashScopeLLM(
            api_key="test-key",
            model="qwen3-omni-flash-2025-12-01",
            temperature=0.1,
            max_tokens=256,
            timeout=12,
            transport=transport,
        )

        answer = llm.generate("prompt text")

        self.assertEqual(answer, "基于资料回答。[1]")
        self.assertEqual(len(transport.calls), 1)
        url, payload, headers, timeout = transport.calls[0]
        self.assertEqual(url, f"{DEFAULT_DASHSCOPE_BASE_URL}/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(timeout, 12)
        self.assertEqual(payload["model"], "qwen3-omni-flash-2025-12-01")
        self.assertEqual(payload["temperature"], 0.1)
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1], {"role": "user", "content": "prompt text"})

    def test_generate_requires_api_key(self) -> None:
        llm = DashScopeLLM(api_key=None)

        with self.assertRaises(RuntimeError):
            llm.generate("prompt text")

    def test_create_model_returns_mock_model(self) -> None:
        llm = create_model("mock", mock_response="fixed answer")

        self.assertIsInstance(llm, MockLLM)
        self.assertEqual(llm.generate("prompt text"), "fixed answer")

    def test_create_model_returns_dashscope_model(self) -> None:
        llm = create_model(
            "dashscope",
            model="qwen3-omni-flash-2025-12-01",
            base_url="https://example.test/v1",
        )

        self.assertIsInstance(llm, DashScopeLLM)
        self.assertEqual(llm.model, "qwen3-omni-flash-2025-12-01")
        self.assertEqual(llm.base_url, "https://example.test/v1")


if __name__ == "__main__":
    unittest.main()
