import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


def ask_gemini(prompt):
    client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(model="gemini-3.1-flash-lite",contents=prompt)
    return response.text.strip()

def extract_repair_tasks(report:str):
    prompt = (
        ""
        ""
        ""
        ""
        f"Insepection report: {report}"
    )
    response = ask_gemini(prompt)

    if response is None:
        raise ValueError("There is no response")

    return [
        line.strip().removeprefix("REPAIR:").strip()
        for line in response.splitlines()
        if line.strip().startswith('REPAIR:')
    ]

def assign_technician_and_urgency(task:str,crew:list[str]) -> tuple[str,str]:
    prompt = (
        ""
        ""
        ""
        "TECHNICIAN: <crew member name>"
        "URGECNY: <High,Medium,Low>"
        f"Report: {task}\nCrew: "+"\n".join(crew)
    )
    response = ask_gemini(prompt)
    if response is None:
        raise ValueError("There is non reponse generated")

    fields = {}
    for line in response.splitlines():
        key,sep,val = line.strip().partition(":")
        if sep and key in {'TECNICIAN','URGENCY'}:
            fields[key] = val

    technician,urgency = fields['TEHCNICIAN'],fields['URGENCY']
    return technician,urgency


        
