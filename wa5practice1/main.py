"""Run the property inspection workflow once and print its summary."""

from graph import build_workflow
from state import InspectionState


if __name__ == "__main__":
    workflow = build_workflow()
    initial_state: InspectionState = {
        "messages": [],
        "inspection_id": "",
        "property": "",
        "crew": [],
        "report": "",
        "work_orders": [],
        "summary": "",
        "status": "input",
    }
    result = workflow.invoke(initial_state)
    print("\n" + result["summary"])
