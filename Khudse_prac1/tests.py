import json
import os
import sys
import typing
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPORTS_FILE = "data/inspection_reports.json"
TEST_INSPECTION_ID = "INS-201"

STATE_FIELDS = ["messages", "inspection_id", "property", "crew", "report",
                "work_orders", "summary", "status"]
NODE_NAMES = ["load_inspection", "extract_repairs", "assign_technicians", "format_summary"]
NODE_ORDER = [("__start__", "load_inspection"),
              ("load_inspection", "extract_repairs"),
              ("extract_repairs", "assign_technicians"),
              ("assign_technicians", "format_summary"),
              ("format_summary", "__end__")]
URGENCIES = {"high", "medium", "low"}

SAMPLE_REPORT = ("The car park barrier at the staff entrance stuck in the raised position on "
                 "Monday and stayed open overnight. The crew agreed to replace the barrier motor "
                 "and to check the access card readers before the weekend.")
SAMPLE_CREW = ["Hamid Karimi", "Sophie Laurent", "Diego Alvarez"]
SAMPLE_TASKS = ["Replace the staff car park barrier motor.",
                "Check the access card readers before the weekend.",
                "Record the barrier fault in the maintenance log."]
SAMPLE_ORDERS = [
    {"task": SAMPLE_TASKS[0], "technician": "Hamid Karimi", "urgency": "High"},
    {"task": SAMPLE_TASKS[1], "technician": "Sophie Laurent", "urgency": "Medium"},
    {"task": SAMPLE_TASKS[2], "technician": "Diego Alvarez", "urgency": "Low"},
]

EXTRACT_REPLY = "\n".join("REPAIR: " + t for t in SAMPLE_TASKS)
ASSIGN_REPLY = "  TECHNICIAN: Sophie Laurent  \n  URGENCY: High  \n"


def call_arg(call, index, name):
    """Read a call argument whether it was passed positionally or by keyword."""
    if len(call.args) > index:
        return call.args[index]
    return call.kwargs.get(name)


def blank_state(**overrides):
    """A fresh state dict so every node can be exercised on its own."""
    state = {"messages": [], "inspection_id": "", "property": "", "crew": [],
             "report": "", "work_orders": [], "summary": "", "status": "input"}
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


def build_report_file():
    """Independent read of the dataset. Ground truth, uses no student code."""
    with open(REPORTS_FILE) as file:
        return json.load(file)["inspections"]


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
        returned = agents.extract_repair_tasks(SAMPLE_REPORT)
    return spy, returned


def build_assign_spy():
    import agents
    spy = SpyClient(ASSIGN_REPLY)
    with patch.object(agents.genai, "Client", return_value=spy):
        returned = agents.assign_technician_and_urgency(SAMPLE_TASKS[0], SAMPLE_CREW)
    return spy, returned


def build_loaded_inspection():
    import nodes
    with patch("builtins.input", return_value=TEST_INSPECTION_ID):
        return nodes.load_inspection(blank_state())


def build_unknown_inspection():
    import nodes
    with patch("builtins.input", return_value="INS-000"):
        return nodes.load_inspection(blank_state())


def build_extracted():
    import nodes
    with patch.object(nodes, "extract_repair_tasks", return_value=list(SAMPLE_TASKS)) as spy:
        update = nodes.extract_repairs(blank_state(report=SAMPLE_REPORT))
    return update, spy


def build_assigned():
    import nodes
    orders = [{"task": t, "technician": "", "urgency": ""} for t in SAMPLE_TASKS]
    replies = [("Hamid Karimi", "High"), ("Sophie Laurent", "Medium"), ("Diego Alvarez", "Low")]
    with patch.object(nodes, "assign_technician_and_urgency", side_effect=replies) as spy:
        update = nodes.assign_technicians(blank_state(crew=SAMPLE_CREW, work_orders=orders))
    return update, spy


def build_formatted():
    import nodes
    return nodes.format_summary(blank_state(inspection_id="INS-900", property="Depot Annual Survey",
                                            crew=SAMPLE_CREW,
                                            work_orders=[dict(o) for o in SAMPLE_ORDERS]))


def build_compiled_graph():
    from graph import build_workflow
    return build_workflow().get_graph()


def build_workflow_result():
    """The only test that reaches the live model. Runs the workflow once and is reused."""
    from graph import build_workflow
    with patch("builtins.input", return_value=TEST_INSPECTION_ID):
        return build_workflow().invoke(blank_state())


# --------------------------------------------------------------------------------------
# state.py  (2 tests)
# --------------------------------------------------------------------------------------

def test_state_declares_every_workflow_field():
    """The state schema must carry all eight fields the workflow reads and writes."""
    from state import InspectionState
    hints = typing.get_type_hints(InspectionState, include_extras=True)

    for field in STATE_FIELDS:
        assert field in hints, f"missing state field: {field}"
    assert set(InspectionState.__annotations__) >= set(STATE_FIELDS)


def test_messages_field_uses_the_add_messages_reducer():
    """The messages channel must be annotated with the add_messages reducer."""
    from langgraph.graph.message import add_messages
    from state import InspectionState

    hints = typing.get_type_hints(InspectionState, include_extras=True)
    metadata = typing.get_args(hints["messages"])[1:]

    assert metadata, "messages is not an Annotated field"
    assert add_messages in metadata


# --------------------------------------------------------------------------------------
# agents.py  (3 tests)
# --------------------------------------------------------------------------------------

def test_extract_agent_sends_the_report_and_returns_the_tasks_it_is_given():
    """The extraction agent must send the report and return exactly the repairs the model reported."""
    spy, returned = cached("extract_spy", build_extract_spy)

    assert len(spy.calls) == 1
    assert SAMPLE_REPORT in spy.calls[0]["prompt"]
    assert isinstance(returned, list)
    assert returned == SAMPLE_TASKS


def test_assign_agent_sends_the_task_and_crew_and_returns_technician_and_urgency():
    """The assignment agent must supply the task and every crew member, then parse both fields."""
    spy, returned = cached("assign_spy", build_assign_spy)
    prompt = spy.calls[0]["prompt"]

    assert len(spy.calls) == 1
    assert SAMPLE_TASKS[0] in prompt
    for member in SAMPLE_CREW:
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

def test_load_inspection_reads_the_requested_inspection_from_the_file():
    """The load node must return the stored property, crew and report for the entered ID."""
    update = cached("loaded", build_loaded_inspection)
    inspections = cached("inspections", build_report_file)
    expected = [i for i in inspections if i["inspection_id"] == TEST_INSPECTION_ID][0]

    assert isinstance(update, dict)
    assert update["inspection_id"] == expected["inspection_id"]
    assert update["property"] == expected["property"]
    assert update["crew"] == expected["crew"]
    assert update["report"] == expected["report"]
    assert update["status"] == "extracting"


def test_load_inspection_reports_an_unknown_inspection_id():
    """An ID that is not in the file must yield no report and no crew."""
    update = cached("unknown", build_unknown_inspection)

    assert isinstance(update, dict)
    assert update["report"] == ""
    assert update["crew"] == []
    assert update["status"] == "completed"
    assert update["inspection_id"] == "INS-000"


def test_extract_repairs_records_one_entry_per_task_returned():
    """The extract node must turn every repair from the agent into one unassigned work order."""
    update, spy = cached("extracted", build_extracted)
    orders = update["work_orders"]

    assert spy.call_count == 1
    assert call_arg(spy.call_args, 0, "report") == SAMPLE_REPORT
    assert len(orders) == len(SAMPLE_TASKS)
    assert [o["task"] for o in orders] == SAMPLE_TASKS
    for order in orders:
        assert order["technician"] == ""
        assert order["urgency"] == ""
    assert update["status"] == "assigning"


def test_assign_technicians_calls_the_agent_once_for_each_work_order():
    """The assign node must ask the agent per work order and keep the tasks aligned."""
    update, spy = cached("assigned", build_assigned)
    orders = update["work_orders"]

    assert spy.call_count == len(SAMPLE_TASKS)
    for call, task in zip(spy.call_args_list, SAMPLE_TASKS):
        assert call_arg(call, 0, "task") == task
        assert call_arg(call, 1, "crew") == SAMPLE_CREW

    assert [o["task"] for o in orders] == SAMPLE_TASKS
    assert [o["technician"] for o in orders] == [o["technician"] for o in SAMPLE_ORDERS]
    assert [o["urgency"] for o in orders] == [o["urgency"] for o in SAMPLE_ORDERS]


def test_format_summary_renders_every_work_order():
    """The summary must report the inspection details and every task, technician and urgency."""
    update = cached("formatted", build_formatted)
    summary = update["summary"]

    assert isinstance(summary, str)
    assert "INS-900" in summary
    assert "Depot Annual Survey" in summary
    assert str(len(SAMPLE_ORDERS)) in summary
    for order in SAMPLE_ORDERS:
        assert order["task"] in summary
        assert order["technician"] in summary
        assert order["urgency"] in summary
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
    """One full run must finish and return the whole state, matching the stored inspection."""
    result = cached("workflow", build_workflow_result)
    inspections = cached("inspections", build_report_file)
    expected = [i for i in inspections if i["inspection_id"] == TEST_INSPECTION_ID][0]

    assert isinstance(result, dict)
    for field in STATE_FIELDS:
        assert field in result

    assert result["inspection_id"] == TEST_INSPECTION_ID
    assert result["property"] == expected["property"]
    assert result["crew"] == expected["crew"]
    assert result["report"] == expected["report"]
    assert result["status"] == "completed"
    assert len(result["messages"]) > 1


def test_workflow_work_orders_are_owned_by_crew_members():
    """Every work order must be a filled record owned by somebody who is really on the crew."""
    result = cached("workflow", build_workflow_result)
    inspections = cached("inspections", build_report_file)
    expected = [i for i in inspections if i["inspection_id"] == TEST_INSPECTION_ID][0]
    orders = result["work_orders"]
    crew = result["crew"]

    assert crew == expected["crew"]
    assert isinstance(orders, list)
    assert len(orders) > 0
    for order in orders:
        assert sorted(order) == ["task", "technician", "urgency"]
        assert len(order["task"].strip()) > 0
        assert len(order["technician"].strip()) > 0
        assert order["urgency"].strip().lower() in URGENCIES
        assert any(order["technician"] in c or c in order["technician"] for c in crew), order["technician"]


def test_workflow_summary_reports_the_work_orders_it_produced():
    """The summary must describe the same inspection and the same orders held in state."""
    result = cached("workflow", build_workflow_result)
    summary = result["summary"]

    assert isinstance(summary, str)
    assert result["property"] in summary
    assert str(len(result["work_orders"])) in summary
    for order in result["work_orders"]:
        assert order["task"] in summary
        assert order["technician"] in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
