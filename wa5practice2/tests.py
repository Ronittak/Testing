import json
import os
import sys
import typing
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REVIEWS_FILE = "data/incident_reviews.json"
TEST_INCIDENT_ID = "INC-301"

STATE_FIELDS = ["messages", "incident_id", "service", "responders", "review",
                "tickets", "summary", "status"]
NODE_NAMES = ["load_incident", "extract_fixes", "assign_engineers", "format_summary"]
NODE_ORDER = [("__start__", "load_incident"),
              ("load_incident", "extract_fixes"),
              ("extract_fixes", "assign_engineers"),
              ("assign_engineers", "format_summary"),
              ("format_summary", "__end__")]
PRIORITIES = {"high", "medium", "low"}

SAMPLE_REVIEW = ("The nightly invoice batch failed on Tuesday because the queue consumer ran out "
                 "of memory, and invoices went out six hours late. The team agreed to raise the "
                 "consumer memory limit and to add an alert on queue depth before month end.")
SAMPLE_RESPONDERS = ["Hamid Karimi", "Sophie Laurent", "Diego Alvarez"]
SAMPLE_TASKS = ["Raise the invoice queue consumer memory limit.",
                "Add an alert on invoice queue depth before month end.",
                "Record the batch failure in the incident log."]
SAMPLE_TICKETS = [
    {"task": SAMPLE_TASKS[0], "engineer": "Hamid Karimi", "priority": "High"},
    {"task": SAMPLE_TASKS[1], "engineer": "Sophie Laurent", "priority": "Medium"},
    {"task": SAMPLE_TASKS[2], "engineer": "Diego Alvarez", "priority": "Low"},
]

EXTRACT_REPLY = "\n".join("FIX: " + t for t in SAMPLE_TASKS)
ASSIGN_REPLY = "  ENGINEER: Sophie Laurent  \n  PRIORITY: High  \n"


def call_arg(call, index, name):
    """Read a call argument whether it was passed positionally or by keyword."""
    if len(call.args) > index:
        return call.args[index]
    return call.kwargs.get(name)


def blank_state(**overrides):
    """A fresh state dict so every node can be exercised on its own."""
    state = {"messages": [], "incident_id": "", "service": "", "responders": [],
             "review": "", "tickets": [], "summary": "", "status": "input"}
    state.update(overrides)
    return state


# --------------------------------------------------------------------------------------
# Shared setup runs through cached(). Work happens once and any failure is raised inside
# the calling test, so every test reports PASSED or FAILED, never ERROR or SKIPPED.
# There are no skip, skipif or xfail markers in this file.
# --------------------------------------------------------------------------------------

_CACHE = {}


def cached(name, builder):
    if name not in _CACHE:
        try:
            _CACHE[name] = (True, builder())
        except Exception as exc:
            _CACHE[name] = (False, f"{type(exc).__name__}: {exc}")
    succeeded, payload = _CACHE[name]
    if not succeeded:
        pytest.fail(f"could not build '{name}' -- {payload}", pytrace=False)
    return payload


def build_review_file():
    """Independent read of the dataset. Ground truth, uses no student code."""
    with open(REVIEWS_FILE) as file:
        return json.load(file)["incidents"]


class _SpyResponse:
    def __init__(self, text):
        self.text = text


class _SpyModels:
    def __init__(self, owner, reply):
        self._owner = owner
        self._reply = reply

    def generate_content(self, model=None, contents=None, **kwargs):
        self._owner.calls.append({"model": model, "prompt": contents})
        return _SpyResponse(self._reply)


class SpyClient:
    """Stands in for genai.Client so an agent call can be inspected without the network."""

    def __init__(self, reply):
        self.calls = []
        self.models = _SpyModels(self, reply)


def build_extract_spy():
    import agents
    spy = SpyClient(EXTRACT_REPLY)
    with patch.object(agents.genai, "Client", return_value=spy):
        returned = agents.extract_remediation_items(SAMPLE_REVIEW)
    return spy, returned


def build_assign_spy():
    import agents
    spy = SpyClient(ASSIGN_REPLY)
    with patch.object(agents.genai, "Client", return_value=spy):
        returned = agents.assign_engineer_and_priority(SAMPLE_TASKS[0], SAMPLE_RESPONDERS)
    return spy, returned


def build_loaded_incident():
    import nodes
    with patch("builtins.input", return_value=TEST_INCIDENT_ID):
        return nodes.load_incident(blank_state())


def build_unknown_incident():
    import nodes
    with patch("builtins.input", return_value="INC-000"):
        return nodes.load_incident(blank_state())


def build_extracted():
    import nodes
    with patch.object(nodes, "extract_remediation_items", return_value=list(SAMPLE_TASKS)) as spy:
        update = nodes.extract_fixes(blank_state(review=SAMPLE_REVIEW))
    return update, spy


def build_assigned():
    import nodes
    tickets = [{"task": t, "engineer": "", "priority": ""} for t in SAMPLE_TASKS]
    replies = [("Hamid Karimi", "High"), ("Sophie Laurent", "Medium"), ("Diego Alvarez", "Low")]
    with patch.object(nodes, "assign_engineer_and_priority", side_effect=replies) as spy:
        update = nodes.assign_engineers(blank_state(responders=SAMPLE_RESPONDERS, tickets=tickets))
    return update, spy


def build_formatted():
    import nodes
    return nodes.format_summary(blank_state(incident_id="INC-900", service="Billing Batch Failure",
                                            responders=SAMPLE_RESPONDERS,
                                            tickets=[dict(t) for t in SAMPLE_TICKETS]))


def build_compiled_graph():
    from graph import build_workflow
    return build_workflow().get_graph()


def build_workflow_result():
    """The only test that reaches the live model. Runs the workflow once and is reused."""
    from graph import build_workflow
    with patch("builtins.input", return_value=TEST_INCIDENT_ID):
        return build_workflow().invoke(blank_state())


# --------------------------------------------------------------------------------------
# state.py  (2 tests)
# --------------------------------------------------------------------------------------

def test_state_declares_every_workflow_field():
    """The state schema must carry all eight fields the workflow reads and writes."""
    from state import IncidentState
    hints = typing.get_type_hints(IncidentState, include_extras=True)

    for field in STATE_FIELDS:
        assert field in hints, f"missing state field: {field}"
    assert set(IncidentState.__annotations__) >= set(STATE_FIELDS)


def test_messages_field_uses_the_add_messages_reducer():
    """The messages channel must be annotated with the add_messages reducer."""
    from langgraph.graph.message import add_messages
    from state import IncidentState

    hints = typing.get_type_hints(IncidentState, include_extras=True)
    metadata = typing.get_args(hints["messages"])[1:]

    assert metadata, "messages is not an Annotated field"
    assert add_messages in metadata


# --------------------------------------------------------------------------------------
# agents.py  (3 tests)
# --------------------------------------------------------------------------------------

def test_extract_agent_sends_the_review_and_returns_the_tasks_it_is_given():
    """The extraction agent must send the review and return exactly the fixes the model reported."""
    spy, returned = cached("extract_spy", build_extract_spy)

    assert len(spy.calls) == 1
    assert SAMPLE_REVIEW in spy.calls[0]["prompt"]
    assert isinstance(returned, list)
    assert returned == SAMPLE_TASKS


def test_assign_agent_sends_the_task_and_responders_and_returns_engineer_and_priority():
    """The assignment agent must supply the task and every responder, then parse both fields."""
    spy, returned = cached("assign_spy", build_assign_spy)
    prompt = spy.calls[0]["prompt"]

    assert len(spy.calls) == 1
    assert SAMPLE_TASKS[0] in prompt
    for member in SAMPLE_RESPONDERS:
        assert member in prompt

    assert isinstance(returned, tuple)
    assert returned == ("Sophie Laurent", "High")


def test_agents_use_the_model_name_from_the_environment():
    """Both agents must request the model named in the environment, not a hard coded one."""
    expected = os.getenv("GEMINI_MODEL")
    extract_spy, _ = cached("extract_spy", build_extract_spy)
    assign_spy, _ = cached("assign_spy", build_assign_spy)

    assert isinstance(expected, str)
    assert len(expected.strip()) > 0
    assert extract_spy.calls[0]["model"] == expected
    assert assign_spy.calls[0]["model"] == expected


# --------------------------------------------------------------------------------------
# nodes.py  (5 tests)
# --------------------------------------------------------------------------------------

def test_load_incident_reads_the_requested_incident_from_the_file():
    """The load node must return the stored service, responders and review for the entered ID."""
    update = cached("loaded", build_loaded_incident)
    incidents = cached("incidents", build_review_file)
    expected = [i for i in incidents if i["incident_id"] == TEST_INCIDENT_ID][0]

    assert isinstance(update, dict)
    assert update["incident_id"] == expected["incident_id"]
    assert update["service"] == expected["service"]
    assert update["responders"] == expected["responders"]
    assert update["review"] == expected["review"]
    assert update["status"] == "extracting"


def test_load_incident_reports_an_unknown_incident_id():
    """An ID that is not in the file must yield no review and no responders."""
    update = cached("unknown", build_unknown_incident)

    assert isinstance(update, dict)
    assert update["review"] == ""
    assert update["responders"] == []
    assert update["status"] == "completed"
    assert update["incident_id"] == "INC-000"


def test_extract_fixes_records_one_entry_per_task_returned():
    """The extract node must turn every fix from the agent into one unassigned ticket."""
    update, spy = cached("extracted", build_extracted)
    tickets = update["tickets"]

    assert spy.call_count == 1
    assert call_arg(spy.call_args, 0, "review") == SAMPLE_REVIEW
    assert len(tickets) == len(SAMPLE_TASKS)
    assert [t["task"] for t in tickets] == SAMPLE_TASKS
    for ticket in tickets:
        assert ticket["engineer"] == ""
        assert ticket["priority"] == ""
    assert update["status"] == "assigning"


def test_assign_engineers_calls_the_agent_once_for_each_ticket():
    """The assign node must ask the agent per ticket and keep the tasks aligned."""
    update, spy = cached("assigned", build_assigned)
    tickets = update["tickets"]

    assert spy.call_count == len(SAMPLE_TASKS)
    for call, task in zip(spy.call_args_list, SAMPLE_TASKS):
        assert call_arg(call, 0, "task") == task
        assert call_arg(call, 1, "responders") == SAMPLE_RESPONDERS

    assert [t["task"] for t in tickets] == SAMPLE_TASKS
    assert [t["engineer"] for t in tickets] == [t["engineer"] for t in SAMPLE_TICKETS]
    assert [t["priority"] for t in tickets] == [t["priority"] for t in SAMPLE_TICKETS]


def test_format_summary_renders_every_ticket():
    """The summary must report the incident details and every task, engineer and priority."""
    update = cached("formatted", build_formatted)
    summary = update["summary"]

    assert isinstance(summary, str)
    assert "INC-900" in summary
    assert "Billing Batch Failure" in summary
    assert str(len(SAMPLE_TICKETS)) in summary
    for ticket in SAMPLE_TICKETS:
        assert ticket["task"] in summary
        assert ticket["engineer"] in summary
        assert ticket["priority"] in summary
    assert update["status"] == "completed"


# --------------------------------------------------------------------------------------
# graph.py  (2 tests)
# --------------------------------------------------------------------------------------

def test_graph_registers_the_four_workflow_nodes():
    """The compiled graph must contain exactly the four documented nodes."""
    compiled = cached("compiled", build_compiled_graph)
    nodes = [n for n in compiled.nodes if not n.startswith("__")]

    assert sorted(nodes) == sorted(NODE_NAMES)


def test_graph_wires_the_nodes_in_the_documented_order():
    """Execution must run load, extract, assign and format in that order."""
    compiled = cached("compiled", build_compiled_graph)
    edges = {(e.source, e.target) for e in compiled.edges}

    for edge in NODE_ORDER:
        assert edge in edges, f"missing edge: {edge[0]} -> {edge[1]}"


# --------------------------------------------------------------------------------------
# end to end  (3 tests) -- the only tests that reach the live model, one shared run
# --------------------------------------------------------------------------------------

def test_workflow_completes_and_returns_every_state_field():
    """One full run must finish and return the whole state, matching the stored incident."""
    result = cached("workflow", build_workflow_result)
    incidents = cached("incidents", build_review_file)
    expected = [i for i in incidents if i["incident_id"] == TEST_INCIDENT_ID][0]

    assert isinstance(result, dict)
    for field in STATE_FIELDS:
        assert field in result

    assert result["incident_id"] == TEST_INCIDENT_ID
    assert result["service"] == expected["service"]
    assert result["responders"] == expected["responders"]
    assert result["review"] == expected["review"]
    assert result["status"] == "completed"
    assert len(result["messages"]) > 1


def test_workflow_tickets_are_owned_by_responders():
    """Every ticket must be a filled record owned by somebody who really responded."""
    result = cached("workflow", build_workflow_result)
    incidents = cached("incidents", build_review_file)
    expected = [i for i in incidents if i["incident_id"] == TEST_INCIDENT_ID][0]
    tickets = result["tickets"]
    responders = result["responders"]

    assert responders == expected["responders"]
    assert isinstance(tickets, list)
    assert len(tickets) > 0
    for ticket in tickets:
        assert sorted(ticket) == ["engineer", "priority", "task"]
        assert len(ticket["task"].strip()) > 0
        assert len(ticket["engineer"].strip()) > 0
        assert ticket["priority"].strip().lower() in PRIORITIES
        assert any(ticket["engineer"] in r or r in ticket["engineer"] for r in responders), ticket["engineer"]


def test_workflow_summary_reports_the_tickets_it_produced():
    """The summary must describe the same incident and the same tickets held in state."""
    result = cached("workflow", build_workflow_result)
    summary = result["summary"]

    assert isinstance(summary, str)
    assert result["service"] in summary
    assert str(len(result["tickets"])) in summary
    for ticket in result["tickets"]:
        assert ticket["task"] in summary
        assert ticket["engineer"] in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
