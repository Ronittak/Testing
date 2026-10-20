"""Run the incident review workflow once and print its summary."""

from graph import build_workflow
from state import IncidentState


if __name__ == "__main__":
    workflow = build_workflow()
    initial_state: IncidentState = {
        "messages": [],
        "incident_id": "",
        "service": "",
        "responders": [],
        "review": "",
        "tickets": [],
        "summary": "",
        "status": "input",
    }
    result = workflow.invoke(initial_state)
    print("\n" + result["summary"])
