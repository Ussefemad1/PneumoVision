"""Pre-flight for Round 3 with --data_pairs partial: counts only, never rows.

    python -m tools.pv.r3_preflight --task phenotyping --data real {DATA}
    python -m tools.pv.r3_preflight --task mortality   --data real {DATA} --missing-image-as-absent

Same options as ``python -m tools.pv.run r3`` (it plans the same run). Prints:

1. the r2b parents found in the manifest, each sha256 checked;
2. per split: stays, and how many have a selected X-ray, a radiology report and
   (phenotyping) a discharge note -- after the partial-mode selection in
   datasets/DataFusion.py; selected X-rays whose file is not on disk; and
   whether the dataset length matches its metadata (partial mode indexes the
   metadata by position);
3. the F2 load: every reader tensor bit-identical to its parent file;
4. on a validation sample: the share of confidence values >= 0.75 per modality
   (EHR counts include padded hours, which upstream pools too);
5. the estimated cache size versus the free space where the cache will go.

Exits 1 if any selected X-ray is missing on disk and --missing-image-as-absent
was not given (it is never turned on silently), or if a count check fails.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from tools.pv import r3_cache, round3  # noqa: E402


def note_present(value) -> bool:
    """What the collate + detect_missingness_batch see: NaN -> "" -> missing."""
    return isinstance(value, str) and bool(value.strip())


def split_counts(dataset, task: str) -> dict:
    meta = dataset.metadata_with_labels
    n = len(dataset)
    rows = meta.iloc[:n]
    cxr = getattr(dataset, "cxr_ds", None)
    dicoms = (
        [d for d in rows["dicom_id"].values if isinstance(d, str) and d]
        if "dicom_id" in rows
        else []
    )
    counts = {
        "stays": n,
        "metadata_rows": len(meta),
        "cxr_selected": len(dicoms),
        "cxr_missing_on_disk": sum(
            1 for d in dicoms if cxr is not None and d not in cxr.filenames_to_path
        ),
        "cxr_without_chexpert_label": sum(
            1 for d in dicoms if cxr is not None and d not in cxr.filesnames_to_labels
        ),
        "rr": int(sum(note_present(t) for t in rows["radiology_text"].values)),
    }
    if task == "phenotyping":
        counts["dn"] = int(sum(note_present(t) for t in rows["discharge_text"].values))
    return counts


def per_stay_bytes(captured: dict[str, torch.Tensor]) -> float:
    n = captured["y"].shape[0]
    return sum(v.numel() * 4 for v in captured.values()) / n


def main(argv: list[str] | None = None) -> int:
    parser = round3.build_parser()
    parser.prog = "python -m tools.pv.r3_preflight"
    parser.add_argument("--sample", type=int, default=256, help="validation stays for step 4")
    opts = parser.parse_args(argv)
    run = round3.plan(opts)
    task = run["task"]
    ok = True

    print(f"[preflight] Round 3 {task}, {run['data']} data, fusion {run['fusion_type']}")
    print("1. parents (sha256 checked against the manifest)")
    for reader, p in run["parents"].items():
        print(f"   {reader:<4} {p['id']}  sha256 OK")

    args = round3.trainer_args(run)
    loaders = round3.live_loaders(args)
    print("2. stays per split (partial mode)")
    total_stays = 0
    for split, dl in zip(r3_cache.SPLITS, loaders, strict=True):
        c = split_counts(dl.dataset, task)
        total_stays += c["stays"]
        print(f"   {split:<5} " + "  ".join(f"{k} {v}" for k, v in c.items()))
        if c["stays"] != c["metadata_rows"]:
            ok = False
            print(f"   STOP {split}: dataset length {c['stays']} != metadata rows "
                  f"{c['metadata_rows']} (partial mode reads metadata by position)")  # fmt: skip
        if c["cxr_missing_on_disk"]:
            if opts.missing_image_as_absent:
                print(f"   {split}: {c['cxr_missing_on_disk']} X-rays missing on disk will be "
                      "treated as absent (--missing-image-as-absent; recorded in the "
                      "manifest)")  # fmt: skip
            else:
                ok = False
                print(f"   STOP {split}: {c['cxr_missing_on_disk']} selected X-rays are not on "
                      "disk. Download them, or rerun with --missing-image-as-absent to treat "
                      "them as a missing modality.")  # fmt: skip
        if c["cxr_without_chexpert_label"] and not opts.missing_image_as_absent:
            ok = False
            print(f"   STOP {split}: {c['cxr_without_chexpert_label']} selected X-rays have no "
                  "CheXpert row (cxr_dataset.py would raise KeyError); "
                  "--missing-image-as-absent treats them as absent too")  # fmt: skip

    print("3. F2: strict parent load")
    with round3.logged(run, "preflight") as out:
        trainer = round3.build_trainer(args, loaders)
    lines = round3.f2_lines(out.getvalue())
    for line in lines:
        print(f"   {line}")
    if len(lines) != len(run["parents"]):
        ok = False
        print("   STOP: not every reader reported bit-identical tensors")

    print(f"4. share of confidence values >= 0.75 on {opts.sample} validation stays")
    captured, seen = [], 0
    for batch in loaders[1]:
        if seen >= opts.sample:
            break
        captured.append(r3_cache.capture_batch(trainer.model, batch, trainer.device))
        seen += captured[-1]["y"].shape[0]
    sample = r3_cache.concat(captured)
    for m in args.modalities.split("-"):
        name = m.lower()
        high = float(sample[f"{name}_n_high"].sum())
        total = high + float(sample[f"{name}_n_low"].sum())
        present = int(sample["missingness"][:, args.modalities.split("-").index(m)].sum())
        print(f"   {m:<4} {high / total if total else float('nan'):.3f}  "
              f"(present in {present} of {seen} stays)")  # fmt: skip

    print("5. cache size")
    bytes_per_stay = per_stay_bytes(sample)
    estimate = bytes_per_stay * total_stays
    cache_dir = Path(run["cache_dir"])
    anchor = next((p for p in [cache_dir, *cache_dir.parents] if p.exists()), Path.cwd())
    free = shutil.disk_usage(anchor).free
    print(f"   ~{bytes_per_stay / 1024:.0f} KB per stay x {total_stays} stays = "
          f"{estimate / 2**30:.1f} GB (fp32); free at {anchor}: {free / 2**30:.1f} GB")  # fmt: skip
    if estimate > 0.9 * free:
        ok = False
        print("   STOP: not enough free space for the cache")
    print("PREFLIGHT " + ("OK" if ok else "FAILED (see STOP lines)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
