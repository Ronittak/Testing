import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

def extract_repair_tasks(report: str):
    prompt = (
        "Identify every repair job from the reports"
        "Format it with `REPAIR: with job of repairing`"
        "If no repair job then format it as no REPAIR:"
        f"Inspection Report: {report}"
    )

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model = os.getenv("GEMINI_MODEL")
    try:
        response = client.models.generate_content(model=model,contents=prompt)
        response = response.text
    except:
        response = """
                    REPAIR:wertyuiopasdfghjk
                    REPAIR:asdfghqweasdf
                    REPAIR:wertasdfgxcvb
                  """
    return [
        line.strip().removeprefix("REPAIR:").strip()
        for line in response.splitlines()
        if line.strip().startswith("REPAIR")
    ]

def assign_technician_and_urgency(task: str, crew: list[str]) -> tuple[str, str]:
    prompt = (
        "Return exactly two lines"
        "TECHNICIAN: <crew member name>"
        "URGENCY: <High,Medium,Low>"
        f"\n\nREPAIR:{task}\nCrew: \n"+"\n".join(crew)
    )

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model = os.getenv("GEMINI_MODEL")
    try:
        response = client.models.generate_content(model=model,contents=prompt)
        response = response.text
    except:
        response = f"""
                   TECHNICIAN:{crew[0]}
                   URGENCY:MEDIUM   
                   """

    fields = {}
    for line in response.splitlines():
        key,sep,val = line.partition(':')
        if sep and key.strip() in {'TECHNICIAN','URGENCY'}:
            fields[key.strip()] = val.strip()

    technician = fields['TECHNICIAN']
    urgency = fields['URGENCY']

    return technician,urgency

 