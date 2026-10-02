"""Tests for the multi-agent supervisor graph.

The router, final answer writer and specialists are replaced by simple fakes,
so these tests check the graph logic (routing, looping back to the supervisor,
finishing, step limits) without needing Ollama or a language model.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from src.multi_agent import WORKER_TOOLS, Route, build_multi_agent
from src.tools import ALL_TOOLS


def scripted_router(decisions: list[str]):
    """Return a fake router that gives the decisions in order."""

    remaining = list(decisions)

    def router(messages: list[BaseMessage]) -> Route:
        return Route(next=remaining.pop(0), reason="scripted for the test")

    return router


def fake_worker(name: str, calls: list[str]):
    """Return a fake specialist that records that it was called."""

    def worker(messages: list[BaseMessage]) -> str:
        calls.append(name)
        return f"report from {name}"

    return worker


def build_with(decisions: list[str], calls: list[str]):
    return build_multi_agent(
        router=scripted_router(decisions),
        finisher=lambda messages: "final answer",
        workers={
            "segment_analyst": fake_worker("segment_analyst", calls),
            "statistics_analyst": fake_worker("statistics_analyst", calls),
        },
    )


def run(graph, limit: int = 20) -> list[BaseMessage]:
    result = graph.invoke(
        {"messages": [HumanMessage(content="test question")]},
        config={"recursion_limit": limit},
    )
    return result["messages"]


def test_supervisor_routes_to_specialists_in_order() -> None:
    """The supervisor sends work to each specialist it picks, then finishes."""

    calls: list[str] = []
    graph = build_with(["segment_analyst", "statistics_analyst", "FINISH"], calls)

    messages = run(graph)

    assert calls == ["segment_analyst", "statistics_analyst"]
    names = [m.name for m in messages if isinstance(m, AIMessage)]
    assert names == ["segment_analyst", "statistics_analyst", "supervisor"]
    assert messages[-1].content == "final answer"


def test_supervisor_can_finish_without_specialists() -> None:
    """If the router says FINISH at once, no specialist is called."""

    calls: list[str] = []
    messages = run(build_with(["FINISH"], calls))

    assert calls == []
    assert messages[-1].name == "supervisor"


def test_specialist_reports_are_added_to_state() -> None:
    """Each specialist adds exactly one short report message."""

    calls: list[str] = []
    messages = run(build_with(["statistics_analyst", "FINISH"], calls))

    reports = [m for m in messages if m.name == "statistics_analyst"]
    assert len(reports) == 1
    assert reports[0].content == "report from statistics_analyst"


def test_step_limit_stops_a_supervisor_that_never_finishes() -> None:
    """A router that never says FINISH is stopped by the recursion limit."""

    calls: list[str] = []
    graph = build_with(["segment_analyst"] * 50, calls)

    with pytest.raises(GraphRecursionError):
        run(graph, limit=7)


def test_every_tool_belongs_to_exactly_one_specialist() -> None:
    """The specialists split the four tools without overlap or gaps."""

    assigned = [t.name for tools in WORKER_TOOLS.values() for t in tools]
    assert sorted(assigned) == sorted(t.name for t in ALL_TOOLS)
    assert len(assigned) == len(set(assigned))
