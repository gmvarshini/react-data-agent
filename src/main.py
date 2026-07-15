"""Command line interface for the ReAct data analysis agent.

Example:
    uv run python -m src.main --question "Which product category has the highest total revenue?"

The CLI runs the agent on the supplied question and prints two things:

1. The full trace of the run: every reasoning turn, tool call (with its
   arguments) and tool result, in the order they occurred.
2. The final answer produced by the agent.

Printing the whole trace makes the ReAct loop transparent, which is useful for
learning and for debugging tool selection.
"""

from __future__ import annotations

import argparse

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError

from src.agent import build_agent, recursion_limit_for
from src.config import get_config


def format_message(message: BaseMessage, index: int) -> str:
    """Render a single message from the agent run as readable text.

    Args:
        message: A LangChain message from the final agent state.
        index: The 1-based position of this message in the trace.

    Returns:
        A formatted, multi-line string describing the message.
    """

    lines: list[str] = []

    if isinstance(message, HumanMessage):
        lines.append(f"[{index}] USER QUESTION")
        lines.append(f"    {message.content}")
    elif isinstance(message, AIMessage):
        tool_calls = message.tool_calls or []
        if tool_calls:
            lines.append(f"[{index}] AGENT REASONING -> tool call")
            if message.content:
                lines.append(f"    thought: {message.content}")
            for call in tool_calls:
                lines.append(f"    calls: {call['name']}({call['args']})")
        else:
            lines.append(f"[{index}] AGENT ANSWER")
            lines.append(f"    {message.content}")
    elif isinstance(message, ToolMessage):
        lines.append(f"[{index}] TOOL RESULT ({message.name})")
        lines.append(f"    {message.content}")
    else:
        lines.append(f"[{index}] {message.__class__.__name__}")
        lines.append(f"    {message.content}")

    return "\n".join(lines)


def run(question: str) -> None:
    """Execute the agent on a question and print the trace and final answer.

    Args:
        question: The natural language analytics question to answer.
    """

    config = get_config()
    agent = build_agent(config)

    inputs = {"messages": [HumanMessage(content=question)]}
    graph_config = {"recursion_limit": recursion_limit_for(config)}

    print("=" * 70)
    print(f"QUESTION: {question}")
    print(f"MODEL: {config.ollama_model}   MAX STEPS: {config.max_agent_steps}")
    print("=" * 70)

    try:
        result = agent.invoke(inputs, config=graph_config)
    except GraphRecursionError:
        print(
            "\nThe agent reached its maximum step limit "
            f"({config.max_agent_steps} steps) without finishing. "
            "Try a more specific question or raise MAX_AGENT_STEPS."
        )
        return

    messages = result["messages"]

    print("\nREASONING TRACE")
    print("-" * 70)
    for i, message in enumerate(messages, start=1):
        print(format_message(message, i))
        print()

    final = messages[-1]
    print("=" * 70)
    print("FINAL ANSWER")
    print("-" * 70)
    print(final.content)
    print("=" * 70)


def main() -> None:
    """Parse command line arguments and run the agent."""

    parser = argparse.ArgumentParser(
        description="Run the ReAct data analysis agent on a question."
    )
    parser.add_argument(
        "--question",
        required=True,
        help="The natural language question to ask the agent.",
    )
    args = parser.parse_args()
    run(args.question)


if __name__ == "__main__":
    main()
