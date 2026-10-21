import json
from pathlib import Path
from langchain_core.messages import AIMessage
from agents import extract_repair_tasks,assign_technician_and_urgency
from state import InspectionState



REPORT_FILE = Path("data/inspection_reports.json")

def load_inspection(state : InspectionState) -> dict:

    inspection_id = input("Enter inspection id: ").strip()
    with REPORT_FILE.open(encoding="utf-8") as file:
        inspections = json.load(file)["inspections"]

    for inspection in inspections:
        if inspection["inspection_id"] == inspection_id:
            return{
                "inspection_id":inspection_id,
                "property":inspection["property"],
                "crew":inspection['crew'],
                "report":inspection["report"],
                "status":"extracting",
                "messages":[AIMessage(content=f"Inspection loaded with id {inspection_id}")]
            }

    return {
        "inspection_id":inspection_id,
        "property":"",
        "crew":[],
        "report":"",
        "status":"completed",
        "messages":[AIMessage(content=f"No {inspection_id} inspection found")]
    }

def extract_repairs(state:InspectionState):

    if not state['report']:
        return{
            "work_orders":[],
            "status":"completed",
            "messages":[AIMessage(content="Nothing to extract repairs")]
        }

    tasks = extract_repair_tasks(state['report'])
    return{
        "work_orders":[{"task":task,"technician":"","urgency":""} for task in tasks],
        "status":"assigning",
        "messages":[AIMessage(content=f"Extracted {len(tasks)} work orders.")]
    }

def assign_technicians(state:InspectionState):

    orders = []
    for order in state["work_orders"]:
        technician,urgency = assign_technician_and_urgency(order['task'],state['crew'])
        orders.append({"task":order['task'],"technician":technician,"urgency":urgency})

    return{
        "work_orders":orders,
        "messages":[AIMessage(content=f"Assigned technician")]
    }

def format_summary(state: InspectionState):
    lines =[
        "Property report summary",
        "",
        f"Inspection ID: {state['inspection_id']}",
        f"Property     : {state['property']}",
        f"crew         : {state['crew']}",
        "",
        f"work_orders ({len(state['work_orders'])})"
    ]
    for index, order in enumerate(state['work_orders'],start=1):
        lines.append(f"  {index}. [{order['urgency']}] {order['task']}")
        lines.append(f"     Technician: {order['technician']}")

    return{
        "summary":"\n".join(lines),
        "status":"completed",
        "messages":[AIMessage(content="property Repair summary")]
    }