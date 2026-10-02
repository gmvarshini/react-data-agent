"""ReAct agent construction for data analysis.

Graph structure
---------------
This module builds a ReAct (Reason + Act) agent using LangGraph's prebuilt
``create_react_agent`` helper. The resulting graph is a small state machine
with two nodes and a conditional edge between them:

* ``agent`` node: calls the language model. The model either produces a final
  answer or decides to call one of the available tools.
* ``tools`` node: executes whichever tool the model requested and appends the
  tool output back onto the message state.

Edges:

* START -> ``agent``
* ``agent`` -> ``tools`` when the model emitted a tool call (conditional edge)
* ``tools`` -> ``agent`` so the model can reason over the tool result
* ``agent`` -> END when the model returns a plain answer with no tool call

This loop is the essence of the ReAct pattern: the model alternates between
reasoning about the question and acting through tools until it has enough
information to answer.

Step limiting
-------------
Left unbounded, a ReAct loop can call tools indefinitely. To keep the agent
safe and cheap, the recursion limit is derived from ``MAX_AGENT_STEPS`` in the
application configuration and passed to the graph at invocation time.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from src.config import AppConfig, get_config
from src.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a data analysis assistant working with a sales \
dataset. The data contains one row per sale with these columns: date, \
product_category, region, units_sold and revenue.

You answer questions by reasoning step by step and calling the tools provided \
to you rather than guessing. Choose the single most appropriate tool for each \
step, read its structured output, and continue until you can give a precise, \
data backed answer.

Guidelines:
- Prefer calling a tool over estimating a number from memory.
- Use filter_by_region for questions about one region.
- Use aggregate_by_category to compare product categories.
- Use compute_summary_statistics for distribution questions across all data.
- Use find_top_performers to identify the highest ranked individual sales.
- When you have the figures you need, stop calling tools and state the answer \
clearly, citing the concrete numbers you used."""


def build_agent(config: AppConfig | None = None) -> CompiledStateGraph:
    """Construct and compile the ReAct data analysis agent.

    Args:
        config: Optional application configuration. When omitted, the
            configuration is loaded from the environment via
            :func:`src.config.get_config`.

    Returns:
        A compiled LangGraph state graph ready to be invoked with a list of
        messages. The returned graph exposes the standard LangGraph interface
        (``invoke``, ``stream`` and friends).
    """

    settings = config or get_config()

    # temperature=0 keeps tool selection and answers deterministic, which is
    # important for a reasoning agent that should not improvise numbers.
    model = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )

    return create_react_agent(
        model,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
    )


def recursion_limit_for(config: AppConfig) -> int:
    """Translate the configured step budget into a LangGraph recursion limit.

    LangGraph counts every node execution against the recursion limit. Each
    ReAct step is roughly one ``agent`` node execution plus one ``tools`` node
    execution, so the raw step count is doubled, with a small allowance for the
    final answer turn.

    Args:
        config: The active application configuration.

    Returns:
        A positive integer recursion limit for graph invocation.
    """

    return config.max_agent_steps * 2 + 1
