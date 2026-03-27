"""Demo: Interactive Computational Chemistry Copilot.

Usage:
    # Using Ollama (default, needs OLLAMA_BASE_URL set):
    OLLAMA_BASE_URL=http://your-ollama-server:11434 python examples/demo_agent.py

    # Using Claude API:
    ANTHROPIC_API_KEY=sk-... python examples/demo_agent.py --backend claude

    # Non-interactive with a single query:
    OLLAMA_BASE_URL=http://your-ollama-server:11434 python examples/demo_agent.py \
        --query "Set up a PBE0+XDM/tight single-point calculation for a water molecule"
"""

from __future__ import annotations

import argparse
import uuid

from chem_copilot.agent import build_agent
from chem_copilot.llm import get_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Chem Copilot Agent Demo")
    parser.add_argument(
        "--backend", choices=["ollama", "claude"], default="ollama",
        help="LLM backend (default: ollama)",
    )
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--query", default=None, help="Single query (non-interactive)")
    args = parser.parse_args()

    llm = get_llm(backend=args.backend, model=args.model)
    agent = build_agent(llm)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    if args.query:
        _run_query(agent, config, args.query)
        return

    print("Computational Chemistry Copilot")
    print("=" * 40)
    print("Type your request, or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        _run_query(agent, config, user_input)


def _run_query(agent, config: dict, query: str) -> None:
    """Send a query to the agent and print the response."""
    print()
    for event in agent.stream(
        {"messages": [("user", query)]},
        config=config,
        stream_mode="updates",
    ):
        for node_name, node_output in event.items():
            if node_name == "agent":
                msgs = node_output.get("messages", [])
                for msg in msgs:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"[Tool call: {tc['name']}]")
                    elif hasattr(msg, "content") and msg.content:
                        print(f"Assistant: {msg.content}")
            elif node_name == "tools":
                msgs = node_output.get("messages", [])
                for msg in msgs:
                    content = msg.content if hasattr(msg, "content") else str(msg)
                    # Truncate long tool outputs for display
                    if len(content) > 500:
                        content = content[:500] + "\n... (truncated)"
                    print(f"[Tool result: {content}]")
    print()


if __name__ == "__main__":
    main()
