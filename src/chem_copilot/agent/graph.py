"""LangGraph agent definition for the Computational Chemistry Copilot.

Implements a custom ReAct-style graph that handles both:
1. Native tool calls (via model.bind_tools — works with Claude, GPT, large Ollama models)
2. Text-based JSON tool calls (smaller Ollama models that output tool calls as JSON text)

This makes the agent robust across different model sizes and providers.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState

from chem_copilot.agent.prompt import SYSTEM_PROMPT
from chem_copilot.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# Build a tool lookup by name
_TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def build_agent(
    llm: BaseChatModel,
    checkpointer: Optional[MemorySaver] = None,
):
    """Build the Chem Copilot agent graph.

    Parameters
    ----------
    llm : BaseChatModel
        The language model to use for reasoning and tool selection.
    checkpointer : MemorySaver, optional
        Memory backend for conversation persistence. If None, creates
        an in-memory checkpointer.

    Returns
    -------
    CompiledGraph
        A LangGraph compiled graph ready to invoke.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    # Try to bind tools natively (works with Claude, GPT, large models)
    try:
        bound_llm = llm.bind_tools(ALL_TOOLS)
    except (NotImplementedError, Exception):
        logger.info("Model does not support native tool binding; using text-based parsing")
        bound_llm = llm

    # ---- Graph nodes ----

    def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
        """Call the LLM with the current message history."""
        messages = state["messages"]

        # Prepend system prompt if not already there
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        response = bound_llm.invoke(messages)

        # Check if the model emitted tool calls as text (common with smaller Ollama models)
        if not response.tool_calls and response.content:
            parsed = _parse_text_tool_call(response.content)
            if parsed is not None:
                tool_name, tool_args = parsed
                response = AIMessage(
                    content="",
                    tool_calls=[{
                        "id": f"text_call_{hash(tool_name) % 10000}",
                        "name": tool_name,
                        "args": tool_args,
                    }],
                )

        return {"messages": [response]}

    def call_tools(state: MessagesState) -> dict[str, list[BaseMessage]]:
        """Execute tool calls from the last AI message."""
        last_msg = state["messages"][-1]
        results: list[ToolMessage] = []

        for tc in last_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc.get("id", f"call_{tool_name}")

            tool = _TOOL_MAP.get(tool_name)
            if tool is None:
                results.append(ToolMessage(
                    content=f"Error: Unknown tool '{tool_name}'. Available: {list(_TOOL_MAP.keys())}",
                    tool_call_id=tool_id,
                ))
                continue

            try:
                result = tool.invoke(tool_args)
                results.append(ToolMessage(content=str(result), tool_call_id=tool_id))
            except Exception as exc:
                results.append(ToolMessage(
                    content=f"Error executing {tool_name}: {exc}",
                    tool_call_id=tool_id,
                ))

        return {"messages": results}

    def should_continue(state: MessagesState) -> Literal["tools", "end"]:
        """Route: if the last message has tool calls, go to tools; otherwise end."""
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "end"

    # ---- Build graph ----

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)


def _parse_text_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Try to extract a tool call from plain text JSON output.

    Handles patterns like:
      {"name": "generate_fhiaims_input", "arguments": {...}}
    or just raw JSON that matches a known tool name.
    """
    text = text.strip()

    # Try parsing as JSON directly
    for candidate in _extract_json_objects(text):
        if isinstance(candidate, dict):
            # Pattern: {"name": "tool_name", "arguments": {...}}
            name = candidate.get("name")
            args = candidate.get("arguments") or candidate.get("args") or candidate.get("parameters")
            if name and name in _TOOL_MAP and isinstance(args, dict):
                return name, args

    return None


def _extract_json_objects(text: str) -> list[Any]:
    """Extract JSON objects from text, handling markdown code blocks."""
    results = []

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)

    # Find JSON objects by matching braces
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start : i + 1])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None

    return results
