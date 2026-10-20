"""Gemini agents for extracting remediation fixes and assigning engineers."""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


load_dotenv(Path(__file__).resolve().parent / ".env")


def extract_remediation_items(review: str) -> list[str]:
    """Return the fixes reported by the model, without rewriting them."""
    prompt = (
        "Identify every remediation or follow-up action arising from this incident "
        "review. Preserve any named system, deadline, and urgency in each action. "
        "Do not invent actions. Reply with one action per line, each beginning with "
        "FIX: followed by the action. Do not include commentary. If there are no "
        "actions, return no FIX: lines.\n\n"
        f"Incident review:\n{review}"
    )
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=os.environ["GEMINI_MODEL"], contents=prompt
    )
    if response.text is None:
        raise ValueError("The extraction model returned no text.")
    return [
        line.strip().removeprefix("FIX:").strip()
        for line in response.text.splitlines()
        if line.strip().startswith("FIX:")
    ]


def assign_engineer_and_priority(task: str, responders: list[str]) -> tuple[str, str]:
    """Ask the model for one responder and priority, then read its labelled reply."""
    prompt = (
        "Assign this remediation action to exactly one engineer from the complete "
        "responder list below. Copy the engineer's name exactly as written. Choose "
        "the priority as exactly High, Medium, or Low. Reply with exactly two lines:\n"
        "ENGINEER: <responder name>\nPRIORITY: <High, Medium, or Low>\n\n"
        f"Action: {task}\nResponders:\n" + "\n".join(responders)
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
        if separator and label in {"ENGINEER", "PRIORITY"}:
            fields[label] = value.strip()

    if "ENGINEER" not in fields or "PRIORITY" not in fields:
        raise ValueError("The model reply must include ENGINEER: and PRIORITY: lines.")
    engineer, priority = fields["ENGINEER"], fields["PRIORITY"]
    if engineer not in responders or priority not in {"High", "Medium", "Low"}:
        raise ValueError("The model returned an invalid engineer or priority.")
    return engineer, priority
