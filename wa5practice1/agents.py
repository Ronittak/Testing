"""Gemini agents for extracting repairs and assigning technicians."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


load_dotenv(Path(__file__).resolve().parent / ".env")


def extract_repair_tasks(report: str) -> list[str]:
    """Return the repairs reported by the model, without rewriting them."""
    prompt = (
        "Identify every repair or follow-up job arising from this property "
        "inspection report. Preserve any named location, deadline, and urgency "
        "in each job. Do not invent jobs. Reply with one job per line, each "
        "beginning with REPAIR: followed by the job. Do not include commentary. "
        "If there are no jobs, return no REPAIR: lines.\n\n"
        f"Inspection report:\n{report}"
    )
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=os.environ["GEMINI_MODEL"], contents=prompt
    )
    if response.text is None:
        raise ValueError("The extraction model returned no text.")
    return [
        line.strip().removeprefix("REPAIR:").strip()
        for line in response.text.splitlines()
        if line.strip().startswith("REPAIR:")
    ]


def assign_technician_and_urgency(task: str, crew: list[str]) -> tuple[str, str]:
    """Ask the model for one crew member and urgency, then read its labelled reply."""
    prompt = (
        "Assign this repair to exactly one technician from the complete crew list "
        "below. Copy the technician's name exactly as written. Choose the urgency "
        "as exactly High, Medium, or Low. Reply with exactly two lines:\n"
        "TECHNICIAN: <crew member name>\nURGENCY: <High, Medium, or Low>\n\n"
        f"Repair: {task}\nCrew:\n" + "\n".join(crew)
    )
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=os.environ["GEMINI_MODEL"], contents=prompt
    )
    if response.text is None:
        raise ValueError("The assignment model returned no text.")

    fields = {}
    for line in response.text.splitlines():
        label, separator, value = line.strip().partition(":")
        if separator and label in {"TECHNICIAN", "URGENCY"}:
            fields[label] = value.strip()

    if "TECHNICIAN" not in fields or "URGENCY" not in fields:
        raise ValueError("The model reply must include TECHNICIAN: and URGENCY: lines.")
    technician, urgency = fields["TECHNICIAN"], fields["URGENCY"]
    if technician not in crew or urgency not in {"High", "Medium", "Low"}:
        raise ValueError("The model returned an invalid technician or urgency.")
    return technician, urgency
