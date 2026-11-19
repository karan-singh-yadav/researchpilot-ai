"""Assembles the LangGraph research workflow.

START -> validate_input -> plan -> [approval_gate (human pause)] -> dispatch
      -> research_task x N (parallel, per dependency wave) -> ingest -> (next wave | analyze)
      -> analyze -> (follow-up research | write) -> critic -> (revise -> critic)* -> finalize -> END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from app.graph.context import RunContext
from app.graph.nodes import make_nodes
from app.graph.state import ResearchState
from app.services.llm import LLMError


def _retry_on(exc: Exception) -> bool:
    return isinstance(exc, LLMError) and exc.retryable


def build_graph(ctx: RunContext | None, checkpointer=None):
    n = make_nodes(ctx)
    retry = RetryPolicy(max_attempts=2, initial_interval=0.5, jitter=False, retry_on=_retry_on)
    g = StateGraph(ResearchState)
    for name in (
        "validate_input",
        "approval_gate",
        "dispatch",
        "research_task",
        "ingest",
        "finalize",
    ):
        g.add_node(name, n[name])
    for name in ("plan", "analyze", "write", "critic", "revise"):
        g.add_node(name, n[name], retry_policy=retry)

    g.add_edge(START, "validate_input")
    g.add_conditional_edges(
        "validate_input",
        lambda s: END if s.get("status") == "failed" else "plan",
        ["plan", END],
    )

    def after_plan(s):
        if s.get("status") == "failed":
            return END
        return (
            "approval_gate" if s["require_approval"] and not s.get("plan_approved") else "dispatch"
        )

    g.add_conditional_edges("plan", after_plan, ["approval_gate", "dispatch", END])
    g.add_edge("approval_gate", "dispatch")
    g.add_conditional_edges("dispatch", n["next_wave"], ["research_task", "analyze", "write"])
    g.add_edge("research_task", "ingest")
    g.add_conditional_edges("ingest", n["next_wave"], ["research_task", "analyze", "write"])
    g.add_conditional_edges("analyze", n["after_analysis"], ["research_task", "write", END])
    g.add_conditional_edges("write", n["after_write"], ["critic", END])
    g.add_conditional_edges("critic", n["after_critic"], ["revise", "finalize"])
    g.add_conditional_edges("revise", n["after_revise"], ["critic", "finalize"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer, interrupt_before=["approval_gate"])


def graph_mermaid() -> str:
    return build_graph(None).get_graph().draw_mermaid()


DOC_HEADER = (
    "# Research workflow graph\n\n"
    "_Generated from the compiled LangGraph by `scripts/export_graph.py`; do not edit by hand._\n\n"
)


def render_graph_doc() -> str:
    return f"{DOC_HEADER}```mermaid\n{graph_mermaid().rstrip()}\n```\n"
