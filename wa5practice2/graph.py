"""Build the four-step LangGraph workflow."""

from langgraph.graph import END, START, StateGraph

from nodes import assign_engineers, extract_fixes, format_summary, load_incident
from state import IncidentState


def build_workflow():
    """Return the compiled workflow, ready to invoke with an initial state."""
    workflow = StateGraph(IncidentState)
    workflow.add_node("load_incident", load_incident)
    workflow.add_node("extract_fixes", extract_fixes)
    workflow.add_node("assign_engineers", assign_engineers)
    workflow.add_node("format_summary", format_summary)

    workflow.add_edge(START, "load_incident")
    workflow.add_edge("load_incident", "extract_fixes")
    workflow.add_edge("extract_fixes", "assign_engineers")
    workflow.add_edge("assign_engineers", "format_summary")
    workflow.add_edge("format_summary", END)
    return workflow.compile()
