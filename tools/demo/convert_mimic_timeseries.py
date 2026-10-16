"""Convert MIMIC ``*_timeseries.csv`` files into the Analyze page's EHR CSV.

The extracted timeseries count ``Hours`` forward from ICU admission, can be
fractional, can hold several rows per hour and run past 48 h. The Analyze page
takes one row per whole hour with ``hour`` offsets (0 = now, -47 = 47 hours
earlier). This binning is the training Discretizer's (timestep 1.0, start time
zero): a reading at time t falls in bin ``int(t - 1e-6)``; bins 0..47 (the
in-hospital-mortality window) are kept; within a bin the last reading of each
variable wins, as the Discretizer overwrites in row order. Bin b is written as
``hour = b - 47``, so the window ends at "now".

For every case folder holding an ``ehr.csv`` in the MIMIC layout it writes
``ehr_site.csv`` alongside. Run it yourself on the credentialed machine; it
prints row counts only. Keep the output local (PhysioNet agreement).

Usage:
    python tools/demo/convert_mimic_timeseries.py --cases docs/demo-examples/demo_cases
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

WINDOW_HOURS = 48
EPS = 1e-6


def convert(src: Path, dst: Path) -> tuple[int, int]:
    """Returns (rows read, hourly rows written)."""
    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header or header[0] != "Hours":
            raise ValueError(f"{src} is not a MIMIC timeseries (first column is not Hours)")
        variables = header[1:]
        bins: dict[int, dict[str, str]] = {}
        rows_read = 0
        for row in reader:
            if not row or not row[0].strip():
                continue
            rows_read += 1
            t = float(row[0])
            if t > WINDOW_HOURS + EPS:
                continue
            b = max(0, int(t - EPS))
            cells = bins.setdefault(b, {})
            for name, value in zip(variables, row[1:]):
                if value.strip() != "":
                    cells[name] = value.strip()

    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hour", *variables])
        for b in sorted(bins):
            if b >= WINDOW_HOURS or not bins[b]:
                continue
            writer.writerow([b - (WINDOW_HOURS - 1), *(bins[b].get(v, "") for v in variables)])
    return rows_read, sum(1 for b in bins if b < WINDOW_HOURS and bins[b])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()
    for folder in sorted(p for p in args.cases.iterdir() if p.is_dir()):
        src = folder / "ehr.csv"
        if not src.is_file():
            continue
        try:
            read, written = convert(src, folder / "ehr_site.csv")
        except ValueError as exc:
            print(f"  {folder.name}: skipped ({exc.__class__.__name__}: not a MIMIC timeseries)")
            continue
        print(f"  {folder.name}: {read} readings -> {written} hourly rows in ehr_site.csv")
    print("Done. Upload ehr_site.csv (not ehr.csv) on the Analyze page. Keep it local.")


if __name__ == "__main__":
    main()
