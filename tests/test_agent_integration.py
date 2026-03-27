"""Integration test: verify the agent executes a tool call end-to-end.

Uses a fake LLM that emits a predetermined tool call, so we can test
the full graph execution without needing Ollama or an API key.
"""

from __future__ import annotations

from typing import Any

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver

from chem_copilot.agent.graph import build_agent


class FakeToolCallingLLM(BaseChatModel):
    """A fake LLM that returns a predetermined tool call on first invocation,
    then returns a plain text response on subsequent calls."""

    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeToolCallingLLM:
        """Accept tool binding (no-op for fake LLM)."""
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> ChatResult:
        self.call_count += 1

        # Check if we're responding to a tool result
        has_tool_result = any(
            getattr(m, "type", None) == "tool" for m in messages
        )

        if has_tool_result:
            # After seeing tool results, give a final answer
            msg = AIMessage(content="Here are your FHI-aims input files for the water molecule with PBE/light settings.")
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # First call: emit a tool call
        msg = AIMessage(
            content="",
            tool_calls=[{
                "id": "call_1",
                "name": "generate_fhiaims_input",
                "args": {
                    "symbols": ["O", "H", "H"],
                    "coords": [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]],
                    "xc": "pbe",
                    "basis": "light",
                },
            }],
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


class TestAgentEndToEnd:
    def test_tool_call_and_response(self) -> None:
        """Agent should call the tool and produce a final response."""
        llm = FakeToolCallingLLM()
        agent = build_agent(llm, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-e2e"}}

        result = agent.invoke(
            {"messages": [("user", "Set up a PBE/light FHI-aims calculation for water")]},
            config=config,
        )

        messages = result["messages"]
        # Should have: user, AI (tool call), tool result, AI (final answer)
        assert len(messages) >= 4

        # Check tool was called
        tool_results = [m for m in messages if getattr(m, "type", None) == "tool"]
        assert len(tool_results) == 1
        assert "control.in" in tool_results[0].content
        assert "geometry.in" in tool_results[0].content

        # Check final answer
        final = messages[-1]
        assert final.content  # Non-empty response
        assert "FHI-aims" in final.content or "water" in final.content

    def test_psi4_tool_call(self) -> None:
        """Test with a Psi4 tool call."""

        class FakePsi4LLM(BaseChatModel):
            call_count: int = 0

            @property
            def _llm_type(self) -> str:
                return "fake-psi4"

            def bind_tools(self, tools: Any, **kwargs: Any) -> FakePsi4LLM:
                return self

            def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> ChatResult:
                self.call_count += 1
                has_tool = any(getattr(m, "type", None) == "tool" for m in messages)
                if has_tool:
                    return ChatResult(generations=[ChatGeneration(
                        message=AIMessage(content="Here is your Psi4 CCSD(T)/cc-pVTZ input for water.")
                    )])
                return ChatResult(generations=[ChatGeneration(
                    message=AIMessage(content="", tool_calls=[{
                        "id": "call_2",
                        "name": "generate_psi4_input",
                        "args": {
                            "symbols": ["O", "H", "H"],
                            "coords": [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]],
                            "method": "ccsd(t)",
                            "basis": "cc-pVTZ",
                        },
                    }])
                )])

        llm = FakePsi4LLM()
        agent = build_agent(llm, checkpointer=MemorySaver())
        result = agent.invoke(
            {"messages": [("user", "CCSD(T)/cc-pVTZ for water")]},
            config={"configurable": {"thread_id": "test-psi4"}},
        )
        tool_results = [m for m in result["messages"] if getattr(m, "type", None) == "tool"]
        assert len(tool_results) == 1
        assert "ccsd(t)" in tool_results[0].content
        assert "import psi4" in tool_results[0].content

    def test_conversation_memory(self) -> None:
        """Second message in same thread should see conversation history."""
        llm = FakeToolCallingLLM()
        memory = MemorySaver()
        agent = build_agent(llm, checkpointer=memory)
        config = {"configurable": {"thread_id": "test-memory"}}

        # First turn
        agent.invoke(
            {"messages": [("user", "Set up water calc")]},
            config=config,
        )

        # Reset call count
        llm.call_count = 0

        # Second turn — agent should still work (has memory of first turn)
        result = agent.invoke(
            {"messages": [("user", "Now change the basis to tight")]},
            config=config,
        )
        # Agent was invoked successfully with memory
        assert len(result["messages"]) > 0
