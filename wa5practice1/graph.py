"""Build the four-step LangGraph workflow."""

from langgraph.graph import END, START, StateGraph

from nodes import assign_technicians, extract_repairs, format_summary, load_inspection
from state import InspectionState


def build_workflow():
    """Return the compiled workflow, ready to invoke with an initial state."""
    workflow = StateGraph(InspectionState)
    workflow.add_node("load_inspection", load_inspection)
    workflow.add_node("extract_repairs", extract_repairs)
    workflow.add_node("assign_technicians", assign_technicians)
    workflow.add_node("format_summary", format_summary)

    workflow.add_edge(START, "load_inspection")
    workflow.add_edge("load_inspection", "extract_repairs")
    workflow.add_edge("extract_repairs", "assign_technicians")
    workflow.add_edge("assign_technicians", "format_summary")
    workflow.add_edge("format_summary", END)
    return workflow.compile()
