"""Multi-agent supervisor system for data analysis.

Instead of one agent that sees every tool, this module splits the work between
a supervisor and two specialist agents:

* ``segment_analyst``: answers questions about regions and product categories.
  Tools: ``filter_by_region`` and ``aggregate_by_category``.
* ``statistics_analyst``: answers questions about the overall distribution and
  the best individual sales. Tools: ``compute_summary_statistics`` and
  ``find_top_performers``.

Each specialist is itself a small ReAct agent (built with LangGraph's
``create_react_agent``) that only sees its own tools. Fewer tools per agent
means shorter prompts and fewer wrong tool choices.

Graph structure
---------------
The system is a LangGraph ``StateGraph`` with three nodes:

* ``supervisor``: looks at the conversation and decides who acts next. It
  returns ``segment_analyst``, ``statistics_analyst`` or ``FINISH``. On
  ``FINISH`` it writes the final answer from the specialists' reports.
* ``segment_analyst`` and ``statistics_analyst``: run their own ReAct loop and
  add a short report to the shared state.

Edges:

* START -> ``supervisor``
* ``supervisor`` -> a specialist, or END (conditional edge on ``state["next"]``)
* each specialist -> ``supervisor``, so the supervisor can decide again

Context engineering
-------------------
A specialist adds only its final report to the shared state, not every
internal tool call. This keeps the supervisor's context small and focused.

Testability
-----------
:func:`build_multi_agent` takes plain Python callables for the router, the
final answer writer and the specialists. Production code wires these to an
Ollama model (:func:`build_default_multi_agent`), while the tests pass simple
fakes, so the graph logic is tested without a language model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from src.config import AppConfig, get_config
from src.tools import (
    aggregate_by_category,
    compute_summary_statistics,
    filter_by_region,
    find_top_performers,
)

FINISH = "FINISH"

# Which tools each specialist may use. Every tool belongs to exactly one agent.
WORKER_TOOLS = {
    "segment_analyst": [filter_by_region, aggregate_by_category],
    "statistics_analyst": [compute_summary_statistics, find_top_performers],
}

WORKER_PROMPTS = {
    "segment_analyst": (
        "You are the segment analyst for a sales dataset with the columns date, "
        "product_category, region, units_sold and revenue. You answer the parts "
        "of a question that are about one region or about comparing product "
        "categories. Always call a tool instead of guessing, then reply with a "
        "short report that states the exact numbers you found."
    ),
    "statistics_analyst": (
        "You are the statistics analyst for a sales dataset with the columns "
        "date, product_category, region, units_sold and revenue. You answer the "
        "parts of a question about the overall distribution of a metric (mean, "
        "median, standard deviation) or about the single best sales. Always call "
        "a tool instead of guessing, then reply with a short report that states "
        "the exact numbers you found."
    ),
}

SUPERVISOR_PROMPT = """You are the supervisor of a small data analysis team. \
You never analyse data yourself. You decide which specialist should act next.

Team members:
- segment_analyst: questions about one region (North, South, East, West) or \
comparing product categories.
- statistics_analyst: questions about the overall distribution of revenue or \
units sold (mean, median, min, max, standard deviation) or the top individual \
sales.

Read the user question and the reports already in the conversation. If a part \
of the question is still unanswered, choose the specialist for that part. If \
every part has been answered by a report, choose FINISH. Never choose the same \
specialist twice for the same part of the question."""

FINAL_ANSWER_PROMPT = """You are the supervisor of a data analysis team. Using \
only the specialists' reports in the conversation, write one clear final answer \
to the user's question. Cite the exact numbers from the reports. Do not invent \
numbers."""


class Route(BaseModel):
    """The supervisor's routing decision."""

    next: Literal["segment_analyst", "statistics_analyst", "FINISH"] = Field(
        description=(
            "The specialist that should act next, or FINISH when every part of "
            "the question has been answered."
        )
    )
    reason: str = Field(description="One short sentence explaining the choice.")


class SupervisorState(MessagesState):
    """Shared state: the message list plus the supervisor's latest decision."""

    next: str


# Type aliases for the injectable building blocks.
Router = Callable[[list[BaseMessage]], Route]
Finisher = Callable[[list[BaseMessage]], str]
Worker = Callable[[list[BaseMessage]], str]


def build_multi_agent(
    router: Router,
    finisher: Finisher,
    workers: dict[str, Worker],
) -> CompiledStateGraph:
    """Assemble the supervisor graph from plain callables.

    Args:
        router: Takes the conversation and returns a :class:`Route` decision.
        finisher: Takes the conversation and returns the final answer text.
        workers: Maps each specialist name to a callable that takes the
            conversation and returns that specialist's short report.

    Returns:
        A compiled LangGraph graph. Invoke it with
        ``{"messages": [HumanMessage(...)]}`` and a ``recursion_limit``.
    """

    def supervisor_node(state: SupervisorState) -> dict:
        decision = router(state["messages"])
        if decision.next == FINISH:
            answer = finisher(state["messages"])
            return {
                "next": FINISH,
                "messages": [AIMessage(content=answer, name="supervisor")],
            }
        return {"next": decision.next}

    def make_worker_node(name: str, worker: Worker) -> Callable[[SupervisorState], dict]:
        def worker_node(state: SupervisorState) -> dict:
            report = worker(state["messages"])
            # Only the short report enters the shared state (context engineering).
            return {"messages": [AIMessage(content=report, name=name)]}

        return worker_node

    def route_after_supervisor(state: SupervisorState) -> str:
        return END if state["next"] == FINISH else state["next"]

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor_node)
    for name, worker in workers.items():
        graph.add_node(name, make_worker_node(name, worker))
        graph.add_edge(name, "supervisor")

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {**{name: name for name in workers}, END: END},
    )
    return graph.compile()


def build_default_multi_agent(config: AppConfig | None = None) -> CompiledStateGraph:
    """Build the supervisor system wired to a local Ollama model.

    Args:
        config: Optional application configuration. When omitted, it is loaded
            from the environment via :func:`src.config.get_config`.

    Returns:
        The compiled multi-agent graph.
    """

    settings = config or get_config()
    model = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )
    routing_model = model.with_structured_output(Route)

    def router(messages: list[BaseMessage]) -> Route:
        return routing_model.invoke([SystemMessage(SUPERVISOR_PROMPT), *messages])

    def finisher(messages: list[BaseMessage]) -> str:
        reply = model.invoke([SystemMessage(FINAL_ANSWER_PROMPT), *messages])
        return str(reply.content)

    def make_worker(name: str) -> Worker:
        agent = create_react_agent(
            model, tools=WORKER_TOOLS[name], prompt=WORKER_PROMPTS[name]
        )

        def run(messages: list[BaseMessage]) -> str:
            result = agent.invoke(
                {"messages": messages},
                config={"recursion_limit": settings.max_agent_steps * 2 + 1},
            )
            return str(result["messages"][-1].content)

        return run

    workers = {name: make_worker(name) for name in WORKER_TOOLS}
    return build_multi_agent(router, finisher, workers)
