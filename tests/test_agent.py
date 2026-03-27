"""Tests for the LangGraph agent construction.

These tests verify the agent graph builds correctly and tools bind properly.
LLM-dependent tests (actual inference) are in test_agent_integration.py
and require Ollama or Claude API access.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chem_copilot.agent.graph import build_agent
from chem_copilot.agent.prompt import SYSTEM_PROMPT


class TestAgentConstruction:
    def test_build_with_mock_llm(self) -> None:
        """Agent graph should build without errors."""
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        agent = build_agent(mock_llm)
        assert agent is not None

    def test_system_prompt_has_tool_descriptions(self) -> None:
        assert "generate_fhiaims_input" in SYSTEM_PROMPT
        assert "generate_psi4_input" in SYSTEM_PROMPT
        assert "parse_calculation_output" in SYSTEM_PROMPT

    def test_system_prompt_mentions_key_methods(self) -> None:
        assert "DFT" in SYSTEM_PROMPT
        assert "CCSD(T)" in SYSTEM_PROMPT
        assert "XDM" in SYSTEM_PROMPT


class TestLLMConfig:
    def test_ollama_config(self) -> None:
        from chem_copilot.llm.config import get_llm
        llm = get_llm(
            backend="ollama",
            model="qwen3:32b",
            base_url="http://localhost:11434",
        )
        assert llm is not None

    def test_invalid_backend_raises(self) -> None:
        from chem_copilot.llm.config import get_llm
        with pytest.raises(ValueError, match="Unknown backend"):
            get_llm(backend="gpt4")

    def test_claude_requires_api_key(self) -> None:
        import os
        from chem_copilot.llm.config import get_llm
        try:
            import langchain_anthropic  # noqa: F401
        except ImportError:
            pytest.skip("langchain-anthropic not installed (optional [claude] extra)")
        # Temporarily clear the env var if set
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                get_llm(backend="claude")
        finally:
            if old is not None:
                os.environ["ANTHROPIC_API_KEY"] = old
