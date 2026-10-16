"""Fill the notes of Norhan's demo_cases folders from MIMIC-IV-Note.

Run this yourself, on the credentialed machine. For every case folder that
has an ``ids.txt`` with a ``hadm_id``, it writes next to the X-ray and EHR:

- ``radiology.txt``           all radiology reports of the admission, joined
                              with a space (as DataFusion builds RR for
                              phenotyping) -> use for PNEUMONIA;
- ``radiology_first48h.txt``  only reports from the first 48 h of the ICU
                              stay (as for in-hospital mortality) -> use for
                              MORTALITY;
- ``discharge.txt``           the admission's discharge note -> PNEUMONIA only,
                              never mortality (it describes the outcome);

then removes ``notes_TODO.txt``. A file is not written when the admission has
no such note, and the summary says so.

PhysioNet data use agreement: this is real patient data. Keep it on this
machine and on localhost. Never commit it, upload it to the Railway site,
paste it or screenshot it to an online tool or AI assistant, or show it to
anyone without PhysioNet credentialed access. The script prints folder names
and counts only, never note text or identifiers.

Usage (repo root, the repo .venv):
    python tools/demo/fill_demo_notes.py \\
        --cases docs/demo-examples/demo_cases \\
        --notes-dir D:/PneumoVision_data --stays data/ehr_full/root/all_stays.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

NOTE_COLUMNS = ["subject_id", "hadm_id", "charttime", "text"]
HADM = re.compile(r"hadm_id\D{0,5}?(\d{6,})", re.IGNORECASE)


def read_hadm_id(ids_file: Path) -> int | None:
    """``hadm_id: 123``-style text, or a header row with a value row under it."""
    text = ids_file.read_text(encoding="utf-8", errors="replace")
    match = HADM.search(text)
    if match:
        return int(match.group(1))
    lines = [re.split(r"[,\t;|]", line.strip()) for line in text.splitlines() if line.strip()]
    for header, values in zip(lines, lines[1:]):
        names = [h.strip().lower() for h in header]
        if "hadm_id" in names:
            value = values[names.index("hadm_id")].strip() if len(values) > names.index("hadm_id") else ""
            if value.isdigit():
                return int(value)
    return None


def read_notes(path: Path, hadm_ids: set[int]) -> pd.DataFrame:
    """The rows of a multi-GB note file whose admission is in ``hadm_ids``."""
    kept = []
    for chunk in pd.read_csv(path, usecols=NOTE_COLUMNS, chunksize=200_000):
        chunk = chunk.dropna(subset=["hadm_id", "text"])
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        kept.append(chunk[chunk["hadm_id"].isin(hadm_ids)])
    notes = pd.concat(kept, ignore_index=True)
    notes["charttime"] = pd.to_datetime(notes["charttime"])
    return notes.sort_values("charttime")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, required=True, help="the demo_cases folder")
    parser.add_argument("--notes-dir", type=Path, required=True,
                        help="MIMIC-IV-Note folder with radiology.csv and discharge.csv")
    parser.add_argument("--stays", type=Path, required=True,
                        help="all_stays.csv, for the ICU admission time (48 h window)")
    args = parser.parse_args()

    cases: dict[str, int] = {}
    for folder in sorted(p for p in args.cases.iterdir() if p.is_dir()):
        ids = folder / "ids.txt"
        hadm = read_hadm_id(ids) if ids.is_file() else None
        if hadm is None:
            print(f"  {folder.name}: no hadm_id found in ids.txt, skipped")
            continue
        cases[folder.name] = hadm
    if not cases:
        raise SystemExit("No case folders with a hadm_id in ids.txt.")

    stays = pd.read_csv(args.stays, usecols=["hadm_id", "intime"])
    stays["intime"] = pd.to_datetime(stays["intime"])
    # The first ICU stay of the admission (one per admission after extraction).
    intime = stays.sort_values("intime").groupby("hadm_id")["intime"].first()

    wanted = set(cases.values())
    print(f"{len(cases)} cases; reading notes (a few minutes for the multi-GB files)...")
    radiology = read_notes(args.notes_dir / "radiology.csv", wanted)
    discharge = read_notes(args.notes_dir / "discharge.csv", wanted)

    for name, hadm in cases.items():
        folder = args.cases / name
        rr = radiology[radiology["hadm_id"] == hadm]
        dn = discharge[discharge["hadm_id"] == hadm]
        written: list[str] = []

        if not rr.empty:
            (folder / "radiology.txt").write_text(" ".join(rr["text"].astype(str)), encoding="utf-8")
            written.append(f"radiology ({len(rr)} reports)")
            start = intime.get(hadm)
            if start is not None:
                early = rr[(rr["charttime"] >= start) & (rr["charttime"] <= start + pd.Timedelta(hours=48))]
                if not early.empty:
                    (folder / "radiology_first48h.txt").write_text(
                        " ".join(early["text"].astype(str)), encoding="utf-8"
                    )
                    written.append(f"radiology_first48h ({len(early)})")
                else:
                    written.append("no report in the first 48 h")
        else:
            written.append("NO radiology report")

        if not dn.empty:
            (folder / "discharge.txt").write_text(str(dn["text"].iloc[-1]), encoding="utf-8")
            written.append("discharge")
        else:
            written.append("NO discharge note")

        todo = folder / "notes_TODO.txt"
        if todo.exists() and not rr.empty:
            todo.unlink()
        print(f"  {name}: " + ", ".join(written))

    print("Done. Keep these files local (PhysioNet agreement).")


if __name__ == "__main__":
    main()
