import re

ABBREVIATIONS = {
    "PO": "by mouth", "WBC": "white blood cell count", "BID": "twice a day",
    "PRN": "as needed", "IV": "intravenous", "NAD": "no acute distress",
    "ED": "emergency department", "BP": "blood pressure", "CT": "computed tomography",
    "HR": "heart rate", "HTN": "hypertension", "RR": "respiratory rate",
    "TID": "three times a day", "CXR": "chest x-ray", "SOB": "shortness of breath",
    "DM": "diabetes mellitus", "PMH": "past medical history", "MI": "myocardial infarction",
    "HPI": "history of present illness", "QID": "four times a day",
    "CBC": "complete blood count", "CHF": "congestive heart failure",
    "ICU": "intensive care unit", "UTI": "urinary tract infection",
    "COPD": "chronic obstructive pulmonary disease", "BUN": "blood urea nitrogen",
    "CAP": "community-acquired pneumonia", "PNA": "pneumonia", "IM": "intramuscular",
    "PICC": "peripherally inserted central catheter", "DNR": "do not resuscitate",
    "WNL": "within normal limits", "URI": "upper respiratory infection",
    "ABG": "arterial blood gas", "CXR PA": "chest x-ray posteroanterior",
    "O2 SAT": "oxygen saturation", "ARDS": "acute respiratory distress syndrome",
    "HAP": "hospital-acquired pneumonia", "SPO2": "oxygen saturation",
    "VAP": "ventilator-associated pneumonia", "TEMP": "temperature",
    "FIO2": "fraction of inspired oxygen", "NG TUBE": "nasogastric tube",
    "RA": "room air",       # dominant-sense decision, see thesis note
    "AC": "before meals",   # dominant-sense decision, see thesis note
}

# Left unexpanded on purpose — no dominant sense in this data
LEFT_UNEXPANDED = {
    "MS": ["mental status", "multiple sclerosis", "morphine sulfate", "musculoskeletal"],
    "DC": ["discharge", "discontinue"],
    "OD": ["overdose", "once daily", "right eye"],
}

def expand_abbreviations(text):
    if not isinstance(text, str):
        return text
    result = text
    # CVA context rule — check first, before the general pass
    result = re.sub(r"\bCVA\b(?=\s+tenderness)", "costovertebral angle", result)
    result = re.sub(r"\bCVA\b", "cerebrovascular accident (stroke)", result)
    for abbr, full in ABBREVIATIONS.items():
        result = re.sub(rf"\b{re.escape(abbr)}\b", full, result)
    return result