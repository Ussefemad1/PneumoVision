import pandas as pd

discharge_raw = pd.read_csv("data/mimic-iv-note/discharge_clean.csv")
discharge_split = pd.read_csv("data/mimic-iv-note/discharge_sections.csv")

dn_sample = discharge_raw.sample(n=50, random_state=7)

with open("hand_check_discharge_v2.txt", "w", encoding="utf-8") as f:
    for _, row in dn_sample.iterrows():
        note_id = row["note_id"]
        text = row["text"]
        split_row = discharge_split[discharge_split["note_id"] == note_id]
        f.write(f"{'='*70}\nNOTE ID: {note_id}\n{'='*70}\n")
        # Show a window AROUND "Discharge Diagnosis" instead of the start of the note
        idx = text.find("Discharge Diagnosis")
        if idx == -1:
            f.write("--- 'Discharge Diagnosis' NOT FOUND IN ORIGINAL TEXT ---\n")
            snippet = text[-800:]  # show the tail instead
        else:
            snippet = text[max(0, idx-100): idx+800]
        f.write("--- ORIGINAL TEXT (around Discharge Diagnosis) ---\n")
        f.write(snippet + "\n\n")
        f.write("--- EXTRACTED 'Discharge Diagnosis' ---\n")
        val = split_row["Discharge Diagnosis"].values[0] if len(split_row) and "Discharge Diagnosis" in split_row.columns else "(not found)"
        f.write(f"{val}\n\n")
        f.write("CORRECT? (yes/no/partial): \n\n\n")

print("Wrote hand_check_discharge_v2.txt")