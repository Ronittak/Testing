"""The state shared by the incident review workflow nodes."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class IncidentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    incident_id: str
    service: str
    responders: list[str]
    review: str
    tickets: list[dict[str, str]]
    summary: str
    status: Literal["input", "extracting", "assigning", "completed"]
