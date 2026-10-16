"""Round 3 (MedPatch, CMSMAFusion c-msma) on top of the frozen Round 2b readers.

    python -m tools.pv.run r3 --task phenotyping --data real {DATA} --dry-run
    python -m tools.pv.run r3 --task phenotyping --data real {DATA}                  # all steps
    python -m tools.pv.run r3 --task mortality   --data real {DATA} --step cache      # one step
    python -m tools.pv.run r3 --task mortality   --data real {DATA} \
        --run-id r3-fusion-001 --record-only

Steps (``--step``; ``all`` runs them in this order):

- ``cache``        run the frozen readers once per stay (tools/pv/r3_cache.py),
                   resumable, on the Colab disk;
- ``equivalence``  cached vs live forward on real validation batches (max abs
                   differences of high_conf / low / late, gradients, one SGD step);
- ``train``        MSMA_Trainer on the cache: the main run, then the same with
                   ``--ablation without_missingness_module`` (and, with
                   ``--extra-ablations``, without_joint_module / without_late_module);
- ``record``       score the head chosen on validation (F3) on val and test and
                   write the manifest row (stage r3, reader fusion) and r3_results.csv;
- ``ablations``    (not part of ``all``) train and score further variants from
                   the same cache after the row exists (``--variants``; default
                   without_joint_module, without_late_module,
                   late_weights_per_class, mask_missing_tokens) into
                   r3_ablations.csv; the manifest row is left untouched.

- ``sweep``        (not part of ``all``; needs ``--sweep-thresholds``) train and
                   score one variant per swept threshold from the binned cache
                   into r3_threshold_sweep.csv. The run's own threshold (0.75)
                   is the recorded main run; the swept ones are ablations only and
                   never become the manifest row, the reported result or the
                   checkpoint anyone loads.

``--sweep-thresholds`` makes the cache step capture, besides the 0.75 groups
(stored exactly as without it), one group per other threshold
(tools/pv/r3_cache.at_threshold). Such a cache lives in its own directory
(<task>-sweep-<thresholds>), so a plain cache is never overwritten.

Exit code 0 when the steps succeed, non-zero on an error.

The command is Confidence-Patching.sh with: --fusion_type c-msma (c-e-msma
only with --allow-entropy, and then live, without the cache), --data_pairs
partial, every --*_confidence_threshold 0.75, --bert_model_name from the
manifest, --load_* = the r2b parents (latest real row per reader unless
--parents), --epochs (default 100), --save_dir <RUNS_ROOT>/<task>/r3/<id>.
Metrics: AUROC / AUPRC only.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.pv import manifest, r3_cache  # noqa: E402
from tools.pv.medpatch_bridge import (  # noqa: E402
    MEDPATCH,
    REPO_ROOT,
    build_loaders,
    medpatch_cwd,
    parse_args,
)
from tools.pv.paper_scripts import (  # noqa: E402
    ENTROPY_FUSION,
    LOAD_FLAG,
    READERS_FOR_TASK,
    ROUND3_FUSION,
    TASK_ALIASES,
    canonical_task,
    round3_argv,
    round3_script,
)

STAGE, READER = "r3", "fusion"
THRESHOLD = "0.75"
TARGET = {"phenotyping": 21, "in-hospital-mortality": 0}
TARGET_NAME = {"phenotyping": "pneumonia", "in-hospital-mortality": "mortality"}
#: Virtual smoke settings (real runs use the paper's: 100 epochs, batch 16).
VIRTUAL_DEFAULTS = {"epochs": 3, "batch_size": 7, "bootstrap_iters": 20}
STEPS = ("cache", "equivalence", "train", "record")
EXTRA_ABLATIONS = ("without_joint_module", "without_late_module")
MAIN_ABLATION = "without_missingness_module"
#: Variants trained from the same cache: name -> args overrides on the main run.
VARIANTS = {
    MAIN_ABLATION: {"ablation": MAIN_ABLATION},
    "without_joint_module": {"ablation": "without_joint_module"},
    "without_late_module": {"ablation": "without_late_module"},
    "late_weights_per_class": {"late_weights_per_class": True},
    "mask_missing_tokens": {"mask_missing_tokens": True},
}
DEFAULT_ABLATION_STEP = (
    "without_joint_module",
    "without_late_module",
    "late_weights_per_class",
    "mask_missing_tokens",
)


# ── plan ─────────────────────────────────────────────────────────────────────


def parent_rows(args, task: str) -> dict[str, dict[str, str]]:
    readers = READERS_FOR_TASK[task]
    if args.parents:
        rows = {}
        for row_id in args.parents:
            row = manifest.find(row_id)
            if row is None:
                raise manifest.ManifestError(
                    f"--parents {row_id} is not in {manifest.manifest_path()}"
                )
            if row["reader"] in rows:
                raise manifest.ManifestError(f"two --parents for reader {row['reader']}")
            rows[row["reader"]] = row
        if "dn" in rows and task == "in-hospital-mortality":
            raise manifest.ManifestError(
                "dn is never a mortality parent: discharge notes leak the outcome"
            )
        missing = [r for r in readers if r not in rows]
        if missing or set(rows) - set(readers):
            raise manifest.ManifestError(f"--parents must give one r2b row per reader {readers}")
        return rows
    return {r: manifest.latest("r2b", r, args.data, task=task) for r in readers}


def data_root(args, task: str) -> Path:
    if args.data_root:
        return Path(args.data_root)
    suffix = "" if task == "phenotyping" else "-mortality"
    return REPO_ROOT / "data" / "virtual" / f"{args.preset}-r3{suffix}"


def data_dirs(args, task: str) -> dict[str, str]:
    from tools.pv.run import VIRTUAL_NORMALIZER  # noqa: PLC0415

    if args.data == "virtual":
        root = data_root(args, task).resolve()
        if not (root / "README.md").is_file():
            flag = "" if task == "phenotyping" else " --task mortality"
            raise SystemExit(
                f"No virtual Round 3 dataset at {root}. Generate it with:\n"
                f"  python -m tools.synthetic.make_virtual_dataset --preset {args.preset}"
                f"{flag} --round3 --out {root}"
            )
        return {
            "--ehr_data_dir": str(root / "ehr"),
            "--cxr_data_dir": str(root / "cxr"),
            "--notes_data_dir": str(root / "notes"),
            "--normalizer_state": str(root / "ehr" / VIRTUAL_NORMALIZER[task]),
        }
    missing = [
        n for n in ("ehr_data_dir", "cxr_data_dir", "notes_data_dir") if not getattr(args, n)
    ]
    if missing:
        raise SystemExit(
            "--data real needs " + ", ".join(f"--{m.replace('_', '-')}" for m in missing)
        )
    dirs = {
        "--ehr_data_dir": args.ehr_data_dir,
        "--cxr_data_dir": args.cxr_data_dir,
        "--notes_data_dir": args.notes_data_dir,
    }
    if args.normalizer_state:
        dirs["--normalizer_state"] = args.normalizer_state
    return dirs


def default_cache_dir(task: str, data: str, sweep=()) -> Path:
    """On the Colab VM disk (/content), never Drive by default; else under runs/ (gitignored).

    A sweep cache gets its own directory (<task>-sweep-065-075-085), so it never
    overwrites the plain cache a recorded run was trained from.
    """
    base = (
        Path("/content/r3_cache") if Path("/content").is_dir() else REPO_ROOT / "runs" / "r3_cache"
    )
    if sweep:
        name = f"{task}-sweep-" + "-".join(f"{round(t * 100):03d}" for t in sweep)
        return base / data / name
    return base / data / task


def plan(args) -> dict:
    task = canonical_task(args.task)
    fusion = args.fusion_type or ROUND3_FUSION
    if fusion == ENTROPY_FUSION and not args.allow_entropy:
        raise manifest.ManifestError(
            "Round 3 runs c-msma. c-e-msma marks no token high at threshold 0.75; pass "
            "--allow-entropy to run it deliberately (live, without the cache)."
        )
    rows = parent_rows(args, task)
    files = {}
    for reader, row in rows.items():
        files[reader] = manifest.verify_parent(
            row, reader, args.data, expected_stage="r2b", task=task
        )
    bert = {manifest.bert_model_name_of(rows[r]) for r in ("rr", "dn") if r in rows}
    if len(bert) != 1 or "" in bert:
        raise manifest.ManifestError(f"text parents must record one bert_model_name, got {bert}")
    bert_name = bert.pop()

    sweep = sorted({round(float(t), 4) for t in (args.sweep_thresholds or ())})
    if any(not 0.5 < t <= 1.0 for t in sweep):
        raise SystemExit(f"--sweep-thresholds must lie in (0.5, 1]: {sweep}")
    if sweep and fusion != ROUND3_FUSION:
        raise SystemExit("--sweep-thresholds needs the cache (c-msma)")
    run_id = args.run_id or manifest.next_id(STAGE, READER)
    save_dir = (manifest.runs_root() / task / STAGE / run_id).resolve()
    defaults = VIRTUAL_DEFAULTS if args.data == "virtual" else {"epochs": 100}
    overrides: dict[str, str | None] = {
        "--fusion_type": fusion,
        "--data_pairs": "partial",
        **{f"--{m}_confidence_threshold": THRESHOLD for m in ("ehr", "cxr", "rr", "dn")},
        "--bert_model_name": bert_name,
        **{LOAD_FLAG[r]: str(f) for r, f in files.items()},
        **data_dirs(args, task),
        "--save_dir": str(save_dir),
        "--resume": None,
    }
    if task == "in-hospital-mortality":
        del overrides["--dn_confidence_threshold"]  # no DN reader for mortality
    deviations = [
        "F5: phenotyping EHR confidence uses the temperature-scaled logits "
        "(released code: unscaled)",
    ]
    for name in ("epochs", "batch_size", "bootstrap_iters", "num_workers"):
        value = getattr(args, name, None) or defaults.get(name)
        if value is not None:
            overrides[f"--{name}"] = str(value)
            if (
                name in ("epochs", "batch_size")
                and str(value) != {"epochs": "100", "batch_size": "16"}[name]
            ):
                deviations.append(f"{name}={value} (paper {'100' if name == 'epochs' else '16'})")
    for flag, note in (
        ("late_weights_per_class", "late weights normalised per class (--late_weights_per_class)"),
        (
            "mask_missing_tokens",
            "absent modalities' tokens masked from the pools (--mask_missing_tokens)",
        ),
        (
            "missing_image_as_absent",
            "X-rays missing on disk treated as absent (--missing_image_as_absent)",
        ),
    ):
        if getattr(args, flag):
            overrides[f"--{flag}"] = None
            deviations.append(note)
    cached = fusion == ROUND3_FUSION
    if cached:
        deviations += [
            "cached Round 3: frozen readers run once per stay in eval mode (BERT dropout off; "
            "released loop: every epoch, train mode)",
            "cache: train X-rays without augmentation (RandomHorizontalFlip/RandomAffine)",
            "cache: train EHR padding fixed by the capture batches "
            "(EHR pools include padded hours)",
        ]
    argv = round3_argv(task, overrides, allow_entropy=args.allow_entropy)
    return {
        "id": run_id,
        "stage": STAGE,
        "reader": READER,
        "task": task,
        "data": args.data,
        "fusion_type": fusion,
        "cached": cached,
        "parents": {
            r: {"id": rows[r]["id"], "sha256": rows[r]["sha256"], "file": str(files[r])}
            for r in rows
        },  # fmt: skip
        "parent_id": ";".join(rows[r]["id"] for r in READERS_FOR_TASK[task]),
        "bert_model_name": bert_name,
        "save_dir": str(save_dir),
        "cache_dir": str(
            Path(args.cache_dir) if args.cache_dir else default_cache_dir(task, args.data, sweep)
        ),
        "sweep_thresholds": sweep,
        "script": str(round3_script(task).relative_to(REPO_ROOT)).replace("\\", "/"),
        "argv": argv,
        "deviations": deviations,
    }


def describe(run: dict) -> str:
    lines = [
        f"[dry-run] {run['id']}  stage r3  task {run['task']}  data {run['data']}  "
        f"fusion {run['fusion_type']} ({'cached' if run['cached'] else 'live'})  "
        "(nothing will run)",
        f"  script      : {run['script']}",
    ]
    for reader, p in run["parents"].items():
        lines.append(
            f"  parent {reader:<4}: {p['id']}  sha256 {p['sha256'][:16]}...  -- matches: OK"
        )
    lines += [
        f"  BERT        : {run['bert_model_name']} (from the text parents)",
        f"  save_dir    : {run['save_dir']}",
        f"  cache_dir   : {run['cache_dir']}",
        f"  deviations  : {'; '.join(run['deviations'])}",
        "  argv        : python fusion_main.py " + " ".join(run["argv"]),
    ]
    return "\n".join(lines)


# ── helpers ──────────────────────────────────────────────────────────────────


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
        return len(text)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextlib.contextmanager
def logged(run: dict, step: str):
    """Tee stdout to <save_dir>/<step>.log; collect medpatch/ strays into the run folder."""
    from tools.pv.run import _collect_stray_outputs  # noqa: PLC0415

    save_dir = Path(run["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in MEDPATCH.iterdir()}
    buffer = io.StringIO()
    with open(save_dir / f"{step}.log", "a", encoding="utf-8") as log:
        tee = _Tee(sys.stdout, log, buffer)
        try:
            with contextlib.redirect_stdout(tee):
                yield buffer
        finally:
            try:
                _collect_stray_outputs(before, save_dir / "medpatch_outputs")
            except Exception as exc:  # noqa: BLE001 -- housekeeping must not mask the step
                print(f"  WARNING could not collect files written into medpatch/: {exc!r}")


def save_run(run: dict) -> None:
    path = Path(run["save_dir"]) / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")


def trainer_args(run: dict, ablation: str | None = None, save_dir: str | None = None):
    args = parse_args(run["argv"])
    if ablation:
        args.ablation = ablation
    if save_dir:
        args.save_dir = save_dir
    return args


def build_trainer(args, loaders):
    """MSMA_Trainer (strict F2 parent load, equalize) on the given loaders."""
    from trainers.MSMA_trainer import MSMA_Trainer  # noqa: PLC0415

    with medpatch_cwd():
        return MSMA_Trainer(*loaders[:2], args, loaders[2])


def live_loaders(args):
    with medpatch_cwd():
        loaders = build_loaders(args)
    # The cache stores one view per stay: train X-rays without augmentation.
    train_ds, val_ds = loaders[0].dataset, loaders[1].dataset
    if hasattr(train_ds, "cxr_ds") and hasattr(val_ds, "cxr_ds"):
        train_ds.cxr_ds.transform = val_ds.cxr_ds.transform
    return loaders


def f2_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if "tensors bit-identical to" in line]


# ── steps ────────────────────────────────────────────────────────────────────


def step_cache(run: dict, opts) -> None:
    cache_dir = Path(run["cache_dir"])
    args = trainer_args(run)
    with logged(run, "cache") as out:
        loaders = live_loaders(args)
        trainer = build_trainer(args, loaders)
        run["f2"] = f2_lines(out.getvalue())
        sweep = run.get("sweep_thresholds") or []
        current = r3_cache.fingerprint(args, run["parents"], sweep)
        meta = r3_cache.read_meta(cache_dir)
        if meta is not None:
            problems = r3_cache.fingerprint_mismatch(meta["fingerprint"], current)
            if problems:
                raise SystemExit(
                    f"[r3 cache] {cache_dir} holds a different cache; use another --cache-dir "
                    "or delete it:\n  " + "\n  ".join(problems)
                )
        else:
            meta = {
                "fingerprint": current,
                "splits": {},
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            r3_cache.write_meta(cache_dir, meta)
        modalities = args.modalities.split("-")
        check_disk(trainer, loaders, cache_dir, opts, sweep)
        for split, dl in zip(r3_cache.SPLITS, loaders, strict=True):
            info = r3_cache.build_split(
                trainer.model, dl.dataset, split, cache_dir, batch_size=args.batch_size,
                collate_fn=dl.collate_fn, device=trainer.device, shard_size=opts.shard_size,
                num_workers=dl.num_workers, thresholds=sweep,
            )  # fmt: skip
            info.update(r3_cache.split_summary(cache_dir, split, modalities))
            info["images_missing_on_disk"] = missing_images(dl.dataset)
            meta["splits"][split] = info
            r3_cache.write_meta(cache_dir, meta)
            print(f"[r3 cache] {split}: {json.dumps(info)}")
        size = r3_cache.dir_bytes(cache_dir)
        print(f"[r3 cache] done: {size / 2**30:.2f} GB at {cache_dir}, "
              f"free {shutil.disk_usage(cache_dir).free / 2**30:.1f} GB")  # fmt: skip
        info = {"dir": str(cache_dir), "bytes": size, "splits": meta["splits"]}
        # A run trained from a plain cache keeps it as "cache"; its later sweep
        # cache is recorded beside it, never in its place.
        previous = run.get("cache")
        if previous and previous.get("dir", "") != str(cache_dir):
            run["sweep_cache"] = info
        else:
            run["cache"] = info
        if opts.copy_cache_to:
            copy_cache(cache_dir, Path(opts.copy_cache_to) / cache_dir.name)
    save_run(run)


def check_disk(trainer, loaders, cache_dir: Path, opts, sweep) -> None:
    """Refuse to start when free disk < the estimated remaining cache + --disk-margin-gb."""
    batch = next(iter(loaders[1]))
    per_stay = r3_cache.stays_bytes(
        r3_cache.capture_batch(trainer.model, batch, trainer.device, sweep)
    )
    total = sum(len(dl.dataset) for dl in loaders)
    built = r3_cache.dir_bytes(cache_dir) if cache_dir.is_dir() else 0
    remaining = max(0.0, per_stay * total * 1.02 - built)  # +2%: index keys, file overhead
    anchor = next((p for p in [cache_dir, *cache_dir.parents] if p.exists()), Path.cwd())
    free = shutil.disk_usage(anchor).free
    margin = opts.disk_margin_gb * 2**30
    print(f"[r3 cache] estimate: {per_stay / 1024:.0f} KB per stay x {total} stays = "
          f"{per_stay * total / 2**30:.1f} GB; {built / 2**30:.1f} GB already built; "
          f"free {free / 2**30:.1f} GB at {anchor}; margin {opts.disk_margin_gb} GB")  # fmt: skip
    if free < remaining + margin:
        raise SystemExit(
            f"[r3 cache] not enough disk: {remaining / 2**30:.1f} GB still to write + "
            f"{opts.disk_margin_gb} GB margin > {free / 2**30:.1f} GB free"
        )


def _files(root: Path) -> dict[str, int]:
    return {str(f.relative_to(root)): f.stat().st_size for f in root.rglob("*") if f.is_file()}


def copy_cache(source: Path, target: Path) -> bool:
    """Copy the cache file by file with shutil.copyfile; never fails the step.

    Drive's FUSE mount rejects copystat (what copytree / copy2 do after each
    file), so only the bytes are copied. Files already there with the same size
    are skipped, so a rerun resumes. Prints the file counts on both sides.
    """
    print(f"[r3 cache] copying to {target} (--copy-cache-to)")
    src = _files(source)
    copied = failed = 0
    for rel, size in sorted(src.items()):
        dest = target / rel
        try:
            if dest.is_file() and dest.stat().st_size == size:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / rel, dest)
            copied += 1
        except Exception as exc:  # noqa: BLE001 -- report, never fail the cache step
            failed += 1
            print(f"[r3 cache] WARNING could not copy {rel}: {exc!r}")
    try:
        dst = _files(target)
    except Exception as exc:  # noqa: BLE001
        print(f"[r3 cache] WARNING could not list {target}: {exc!r}")
        dst = {}
    same = sum(1 for rel, size in src.items() if dst.get(rel) == size)
    print(f"[r3 cache] copy: source {len(src)} files, target {len(dst)} files, "
          f"{same} identical in size; {copied} copied now, {failed} failed")  # fmt: skip
    return failed == 0 and same == len(src)


def missing_images(dataset) -> int:
    """Selected X-rays (dicom_id set) whose file is not on disk or has no CheXpert row."""
    meta = getattr(dataset, "metadata_with_labels", None)
    cxr = getattr(dataset, "cxr_ds", None)
    if meta is None or cxr is None or "dicom_id" not in meta:
        return 0
    ids = [d for d in meta["dicom_id"].values[: len(dataset)] if isinstance(d, str) and d]
    return sum(
        1 for d in ids if d not in cxr.filenames_to_path or d not in cxr.filesnames_to_labels
    )


def step_equivalence(run: dict, opts, tol: float = 1e-5) -> dict:
    args = trainer_args(run)
    with logged(run, "equivalence"):
        loaders = live_loaders(args)
        trainer = build_trainer(args, loaders)
        sweep = run.get("sweep_thresholds") or []
        main = r3_cache.main_threshold(args)
        checks = [("equivalence", None)]
        if sweep:
            other = opts.equivalence_threshold or max(
                (t for t in sweep if abs(t - main) > 1e-9), key=lambda t: abs(t - main)
            )
            checks.append(("equivalence_sweep", other))
        failed = {}
        for key, threshold in checks:
            reports = []
            for i, batch in enumerate(loaders[1]):
                if i >= opts.equivalence_batches:
                    break
                reports.append(
                    r3_cache.compare_live_cached(
                        trainer.model, batch, trainer.device, threshold=threshold, sweep=sweep
                    )
                )
            worst = {k: max(r[k] for r in reports) for k in reports[0] if k.startswith("max_abs")}
            stays = int(sum(r["stays"] for r in reports))
            at = main if threshold is None else threshold
            print(
                f"[r3 equivalence] threshold {at:.2f}: {len(reports)} val batch(es), "
                f"{stays} stays: " + ", ".join(f"{k} {v:.2e}" for k, v in worst.items())
            )
            run[key] = {"threshold": at, "batches": len(reports), "stays": stays, **worst,
                        "tolerance": tol}  # fmt: skip
            failed.update({f"{at:.2f} {k}": v for k, v in worst.items() if v > tol})
        save_run(run)
        if failed:
            raise SystemExit(f"[r3 equivalence] FAILED (> {tol}): {failed}")
        print("[r3 equivalence] PASS")
    return run["equivalence"]


def cache_loaders(run: dict, args, preload: bool, threshold: float | None = None):
    """Train / val / test loaders on the run's cache; `threshold`: split at a swept one."""
    cache_dir = Path(run["cache_dir"])
    meta = r3_cache.read_meta(cache_dir)
    if meta is None:
        raise SystemExit(f"[r3] no cache at {cache_dir}; run --step cache first")
    main_args = trainer_args(run)  # a sweep variant changes the thresholds, not the cache
    problems = r3_cache.fingerprint_mismatch(
        meta["fingerprint"],
        r3_cache.fingerprint(main_args, run["parents"], run.get("sweep_thresholds") or []),
    )
    if problems:
        raise SystemExit("[r3] the cache does not match this run:\n  " + "\n  ".join(problems))
    incomplete = [s for s in r3_cache.SPLITS if not r3_cache.split_complete(cache_dir, s, meta)]
    if incomplete:
        raise SystemExit(f"[r3] cache incomplete for {incomplete}; rerun --step cache (resumes)")
    from datasets.DataFusion import CLASSES  # noqa: PLC0415

    classes = CLASSES[: args.num_classes] if args.num_classes > 1 else ["mortality"]
    kw = {"classes": classes, "preload": preload, "threshold": threshold,
          "main": r3_cache.main_threshold(main_args)}  # fmt: skip
    return (
        r3_cache.CacheLoader(
            cache_dir, "train", args.batch_size, shuffle=True, drop_last=True, **kw
        ),
        r3_cache.CacheLoader(
            cache_dir, "val", args.batch_size, shuffle=False, drop_last=False, **kw
        ),
        r3_cache.CacheLoader(
            cache_dir, "test", args.batch_size, shuffle=False, drop_last=False, **kw
        ),
    )


def variant_names(opts) -> list[str]:
    """The variants the train / record steps cover: main, then the ablations asked for."""
    names = ["main"]
    if not opts.no_ablation:
        names.append(MAIN_ABLATION)
    if opts.extra_ablations:
        names += list(EXTRA_ABLATIONS)
    return names


def variant_args(run: dict, name: str):
    """The run's args for one variant, saved under <save_dir>/<name> (main: save_dir)."""
    if name == "main":
        return trainer_args(run)
    args = trainer_args(run, save_dir=str(Path(run["save_dir"]) / name))
    for key, value in VARIANTS[name].items():
        setattr(args, key, value)
    return args


def variants(run: dict, opts) -> list[tuple[str, object]]:
    return [(name, variant_args(run, name)) for name in variant_names(opts)]


def train_variant(run: dict, name: str, args, opts, threshold: float | None = None) -> str:
    """Train one variant on the cache (skipped when already finished); its best checkpoint."""
    if finished(args):
        print(f"[r3] {name}: already trained ({args.epochs} epochs or early-stopped); skipped")
        return str(_best(args))
    with logged(run, f"train_{name}") as out:
        loaders = cache_loaders(run, args, opts.preload, threshold)
        trainer = build_trainer(args, loaders)
        run.setdefault("f2", f2_lines(out.getvalue()))
        for module in (trainer.model.fusion_model.text_model, trainer.model.fusion_model.cxr_model):
            if module is not None:
                module.to("cpu")  # never used on the cache: free the GPU
        trainer.train()
        trainer.eval()
    return str(_best(args))


def step_train(run: dict, opts) -> None:
    run["requested_variants"] = variant_names(opts)  # what the checks expect, exactly
    save_run(run)
    if not run["cached"]:
        return live_train(run)
    for name, args in variants(run, opts):
        run.setdefault("trained", {})[name] = train_variant(run, name, args, opts)
        save_run(run)


def step_ablations(run: dict, opts) -> Path:
    """Train and score further variants after the row exists; manifest untouched."""
    if not run["cached"]:
        raise SystemExit("[r3] --step ablations needs the cache (c-msma)")
    names = opts.variants or list(DEFAULT_ABLATION_STEP)
    run["requested_ablations"] = sorted(set(run.get("requested_ablations", [])) | set(names))
    save_run(run)
    for name in names:
        args = variant_args(run, name)
        run.setdefault("trained", {})[name] = train_variant(run, name, args, opts)
        save_run(run)
        with logged(run, f"score_{name}"):
            run.setdefault("ablation_scores", {})[name] = score_variant(run, args, opts)
        save_run(run)
    return write_ablations(run)


def step_share(run: dict, opts) -> Path:
    """Copy the main run (θ = the run's own threshold) to --share-to; never a variant.

    The final model is always the recorded main run. Sweep and ablation
    variants live in subfolders of the run folder and are never copied: the
    shared folder holds the main best checkpoint (the manifest row's file),
    run.json, the result tables (r3_results, r3_ablations, r3_threshold_sweep:
    tables, labelled as ablations) and the top-level logs.
    """
    if not opts.share_to:
        raise SystemExit("--step share needs --share-to <dir>")
    row = manifest.find(run["id"])
    if row is None:
        raise SystemExit(f"[r3] {run['id']} is not recorded; run --step record first")
    best = Path(row["file_path"])
    if manifest.sha256_file(best) != row["sha256"]:
        raise SystemExit(f"[r3] {best} no longer matches the manifest sha256")
    save_dir, target = Path(run["save_dir"]), Path(opts.share_to)
    files = [best] + sorted(
        f for f in save_dir.iterdir()
        if f.is_file() and f.suffix in (".json", ".csv", ".log", ".txt")
    )  # fmt: skip
    target.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copyfile(f, target / f.name)  # bytes only: Drive's FUSE rejects copystat
    print(f"[r3 share] {len(files)} files of the main run -> {target} "
          f"(checkpoint {best.name}; no sweep or ablation variant)")  # fmt: skip
    return target


def sweep_name(t: float) -> str:
    return f"theta_{t:.2f}"


def step_sweep(run: dict, opts) -> Path:
    """One variant per swept threshold from the sweep cache -> r3_threshold_sweep.csv.

    The run's own threshold is the recorded main run (its scores, not a retrain);
    every other threshold is an ablation and never touches the manifest.
    """
    sweep = run.get("sweep_thresholds") or []
    if not sweep:
        raise SystemExit("[r3] --step sweep needs --sweep-thresholds (and their cache)")
    main = r3_cache.main_threshold(trainer_args(run))
    if any(abs(t - main) < 1e-9 for t in sweep) and "main" not in run.get("scores", {}):
        raise SystemExit("[r3] the main run is not recorded yet; run --step record first")
    for t in sweep:
        if abs(t - main) < 1e-9:
            continue
        name = sweep_name(t)
        args = trainer_args(run, save_dir=str(Path(run["save_dir"]) / name))
        r3_cache.set_threshold(args, t)
        run.setdefault("trained", {})[name] = train_variant(run, name, args, opts, threshold=t)
        save_run(run)
        with logged(run, f"score_{name}"):
            run.setdefault("sweep_scores", {})[name] = score_variant(run, args, opts, threshold=t)
        save_run(run)
    return write_sweep(run, main)


def finished(args) -> bool:
    """A variant whose 'last' checkpoint covers every epoch (or ran out of patience)."""
    import torch  # noqa: PLC0415

    from tools.pv.evaluate import checkpoint_path  # noqa: PLC0415

    last = checkpoint_path(args, "last")
    if not last.is_file() or not _best(args).is_file():
        return False
    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    return ckpt["epoch"] >= args.epochs - 1 or ckpt.get("patience", 0) >= args.patience


def live_train(run: dict) -> None:
    """c-e-msma (--allow-entropy): fusion_main.py itself, no cache."""
    from tools.pv.run import execute  # noqa: PLC0415

    run["trained"] = {"main": str(execute(run))}
    save_run(run)


def _best(args) -> Path:
    from tools.pv.evaluate import checkpoint_path  # noqa: PLC0415

    return checkpoint_path(args)


def score_variant(run: dict, args, opts, threshold: float | None = None) -> dict:
    """The chosen head's val and test scores, from the best checkpoint and the cache."""
    import torch  # noqa: PLC0415

    best = _best(args)
    if not best.is_file():
        raise SystemExit(f"[r3] no best checkpoint at {best}")
    # c-e-msma (--allow-entropy) has no cache: score it on the live loaders.
    loaders = (
        cache_loaders(run, args, opts.preload, threshold) if run["cached"] else live_loaders(args)
    )
    args.resume = False
    trainer = build_trainer(args, loaders)
    trainer.load_state(str(best))
    if getattr(trainer, "best_head", None) is None:
        trainer.best_head = trainer.choose_head_on_validation()
    trainer.model.eval()
    with torch.no_grad():
        val = dict(zip(trainer.HEADS, trainer.validate(trainer.val_dl, False)[:5], strict=True))
        test = dict(zip(trainer.HEADS, trainer.validate(trainer.test_dl, True)[:5], strict=True))
    head, target = trainer.best_head, TARGET[run["task"]]
    return {
        "checkpoint": str(best),
        "head": head,
        "val": metrics(val[head], target),
        "test": metrics(test[head], target),
    }


def metrics(ret: dict, target: int) -> dict:
    import numpy as np  # noqa: PLC0415

    auc, auprc = np.ravel(ret["auc_scores"]), np.ravel(ret["auprc_scores"])
    ci_auc, ci_prc = ret["ci_auroc"][target], ret["ci_auprc"][target]
    return {
        "macro_auroc": float(ret["auroc_mean"]),
        "macro_auprc": float(ret["auprc_mean"]),
        "target_auroc": float(auc[target]),
        "target_auprc": float(auprc[target]),
        "target_auroc_ci": [float(min(ci_auc)), float(max(ci_auc))],
        "target_auprc_ci": [float(min(ci_prc)), float(max(ci_prc))],
    }


def step_record(run: dict, opts, who: str) -> dict[str, str]:
    if manifest.find(run["id"]) is not None:
        raise SystemExit(f"[r3] {run['id']} is already in {manifest.manifest_path()}")
    with logged(run, "record"):
        scores = {}
        for name, args in variants(run, opts):
            if name != "main" and not _best(args).is_file():
                print(f"[r3] {name}: no checkpoint, skipped")
                continue
            scores[name] = score_variant(run, args, opts)
    run["scores"] = scores
    write_results(run)
    main = scores["main"]
    best = Path(main["checkpoint"])
    row = manifest.append_row(
        {
            "id": run["id"],
            "who": who,
            "stage": STAGE,
            "reader": READER,
            "task": run["task"],
            "file_path": best.as_posix(),
            "sha256": manifest.sha256_file(best),
            "parent_id": run["parent_id"],
            "data": run["data"],
            "val_auroc": f"{main['val']['target_auroc']:.4f}",
            "val_auprc": f"{main['val']['target_auprc']:.4f}",
            "notes": "; ".join(notes(run)),
            "bert_model_name": run["bert_model_name"],
        }
    )
    run["manifest_row"] = row
    save_run(run)
    return row


def _fmt_ci(m: dict, kind: str) -> str:
    lo, hi = m[f"target_{kind}_ci"]
    return f"{m[f'target_{kind}']:.3f} [{lo:.3f}, {hi:.3f}]"


def notes(run: dict) -> list[str]:
    task, target = run["task"], TARGET_NAME[run["task"]]
    out = []
    if run["data"] == "virtual":
        out.append("VIRTUAL, SYNTHETIC - not a scientific result")
    out.append(
        f"script={run['script']}; fusion={run['fusion_type']}; metric={target} val, chosen head"
    )
    for name, s in run.get("scores", {}).items():
        label = "main" if name == "main" else f"ablation {name}"
        v, t = s["val"], s["test"]
        out.append(
            f"{label}: head={s['head']}; val macro AUROC {v['macro_auroc']:.3f} AUPRC "
            f"{v['macro_auprc']:.3f}; test macro AUROC {t['macro_auroc']:.3f} AUPRC "
            f"{t['macro_auprc']:.3f}; test {target} AUROC {_fmt_ci(t, 'auroc')} "
            f"AUPRC {_fmt_ci(t, 'auprc')} (95% bootstrap CI)"
        )
    for split, info in (run.get("cache") or {}).get("splits", {}).items():
        present = ", ".join(f"{m} {n}" for m, n in info.get("present", {}).items())
        share = ", ".join(f"{m} {v:.3f}" for m, v in info.get("share_high", {}).items())
        out.append(f"{split}: {info.get('stays')} stays; present {present}; share>=0.75 {share}")
        if info.get("images_missing_on_disk"):
            out.append(f"{split}: {info['images_missing_on_disk']} X-rays missing on disk "
                       "treated as absent")  # fmt: skip
    if run.get("equivalence"):
        e = run["equivalence"]
        late = e.get("max_abs_diff_late", float("nan"))
        out.append(f"cached vs live: max abs diff late {late:.1e} "
                   f"({e['stays']} val stays)")  # fmt: skip
    out += run["deviations"]
    if task == "in-hospital-mortality":
        out.append("no discharge notes (they leak the outcome)")
    return out


RESULT_COLUMNS = ["variant", "chosen_head", "split", "macro_auroc", "macro_auprc",
                  "target_auroc", "target_auroc_lo", "target_auroc_hi", "target_auprc",
                  "target_auprc_lo", "target_auprc_hi"]  # fmt: skip


def write_results(run: dict) -> Path:
    return _write_table(run, "r3_results.csv", run.get("scores", {}), "")


def write_ablations(run: dict) -> Path:
    """r3_ablations.csv: the recorded main run for reference, then every ablation scored."""
    scores = {}
    if "main" in run.get("scores", {}):
        scores["main (recorded)"] = run["scores"]["main"]
    scores.update(run.get("ablation_scores", {}))
    return _write_table(run, "r3_ablations.csv", scores,
                        " -- ablations from the same cache; "
                        "the manifest row is the main run")  # fmt: skip


def write_sweep(run: dict, main: float) -> Path:
    """r3_threshold_sweep.csv: one row pair (val, test) per swept threshold."""
    scores = {}
    for t in run["sweep_thresholds"]:
        if abs(t - main) < 1e-9:
            scores[f"{sweep_name(t)} main run (recorded)"] = run["scores"]["main"]
        elif sweep_name(t) in run.get("sweep_scores", {}):
            scores[sweep_name(t)] = run["sweep_scores"][sweep_name(t)]
    return _write_table(
        run,
        "r3_threshold_sweep.csv",
        scores,
        f" -- threshold sweep (ablations); the result is the main run at {main:.2f}",
    )


def _write_table(run: dict, filename: str, scores: dict, title: str) -> Path:
    path = Path(run["save_dir"]) / filename
    cols = RESULT_COLUMNS
    lines = [f"# Round 3 {run['id']} ({run['task']}, {run['data']}); target = "
             f"{TARGET_NAME[run['task']]}; AUROC/AUPRC only{title}", ",".join(cols)]  # fmt: skip
    for name, s in scores.items():
        for split in ("val", "test"):
            m = s[split]
            lines.append(",".join(str(x) for x in (
                name, s["head"], split, f"{m['macro_auroc']:.4f}", f"{m['macro_auprc']:.4f}",
                f"{m['target_auroc']:.4f}", f"{m['target_auroc_ci'][0]:.4f}",
                f"{m['target_auroc_ci'][1]:.4f}", f"{m['target_auprc']:.4f}",
                f"{m['target_auprc_ci'][0]:.4f}", f"{m['target_auprc_ci'][1]:.4f}",
            )))  # fmt: skip
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.pv.run r3", description=__doc__.splitlines()[0]
    )
    p.add_argument("--task", choices=sorted(TASK_ALIASES), required=True)
    p.add_argument("--data", choices=manifest.DATA_KINDS, required=True)
    p.add_argument("--step", choices=("all", *STEPS, "ablations", "sweep", "share"),
                   default="all")  # fmt: skip
    p.add_argument("--share-to", dest="share_to",
                   help="--step share: copy the main run (never a variant) here")  # fmt: skip
    p.add_argument(
        "--sweep-thresholds",
        dest="sweep_thresholds",
        type=float,
        nargs="+",
        help="capture these thresholds too (own cache dir) for --step sweep",
    )
    p.add_argument("--equivalence-threshold", dest="equivalence_threshold", type=float,
                   help="the swept threshold --step equivalence also checks "
                        "(default: the one farthest from 0.75)")  # fmt: skip
    p.add_argument(
        "--disk-margin-gb",
        dest="disk_margin_gb",
        type=float,
        default=20.0,
        help="--step cache refuses to start below estimate + this much free disk",
    )
    p.add_argument("--variants", nargs="+", choices=sorted(VARIANTS),
                   help="--step ablations: which variants "
                        f"(default {' '.join(DEFAULT_ABLATION_STEP)})")  # fmt: skip
    p.add_argument("--parents", nargs="+", help="r2b manifest ids, one per reader")
    p.add_argument("--run-id", dest="run_id")
    p.add_argument("--who", default=os.environ.get("PV_WHO", os.environ.get("USERNAME", "unknown")))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--record-only", dest="record_only", action="store_true",
                   help="only the record step, for a run that trained but crashed "
                        "before the manifest")  # fmt: skip
    p.add_argument("--preset", default="smoke")
    p.add_argument("--data-root", dest="data_root")
    p.add_argument("--ehr-data-dir", dest="ehr_data_dir")
    p.add_argument("--cxr-data-dir", dest="cxr_data_dir")
    p.add_argument("--notes-data-dir", dest="notes_data_dir")
    p.add_argument("--normalizer-state", dest="normalizer_state")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--bootstrap-iters", dest="bootstrap_iters", type=int)
    p.add_argument("--num-workers", dest="num_workers", type=int)
    p.add_argument("--cache-dir", dest="cache_dir", help="default /content/r3_cache/<data>/<task>")
    p.add_argument("--copy-cache-to", dest="copy_cache_to",
                   help="also copy the finished cache here (e.g. Drive); "
                        "never by default")  # fmt: skip
    p.add_argument("--shard-size", dest="shard_size", type=int, default=1024,
                   help="stays per shard; a multiple of the batch size")  # fmt: skip
    p.add_argument("--preload", action="store_true", help="keep the cache in RAM while training")
    p.add_argument("--equivalence-batches", dest="equivalence_batches", type=int, default=1)
    p.add_argument("--no-ablation", dest="no_ablation", action="store_true",
                   help=f"skip the {MAIN_ABLATION} run")  # fmt: skip
    p.add_argument("--extra-ablations", dest="extra_ablations", action="store_true",
                   help="also train without_joint_module and without_late_module")  # fmt: skip
    p.add_argument("--late-weights-per-class", dest="late_weights_per_class", action="store_true")
    p.add_argument("--mask-missing-tokens", dest="mask_missing_tokens", action="store_true")
    p.add_argument("--missing-image-as-absent", dest="missing_image_as_absent", action="store_true")
    p.add_argument("--fusion-type", dest="fusion_type", choices=(ROUND3_FUSION, ENTROPY_FUSION))
    p.add_argument("--allow-entropy", dest="allow_entropy", action="store_true")
    return p


def main(argv: list[str] | None = None) -> dict | None:
    opts = build_parser().parse_args(argv)
    if opts.record_only and not opts.run_id:
        raise SystemExit("--record-only needs --run-id")
    run = plan(opts)
    if opts.dry_run:
        print(describe(run))
        return None
    saved = Path(run["save_dir"]) / "run.json"
    if saved.is_file():  # keep what earlier steps recorded (cache counts, F2, equivalence)
        previous = json.loads(saved.read_text(encoding="utf-8"))
        for key in ("cache", "f2", "equivalence", "trained", "requested_variants", "scores",
                    "requested_ablations", "ablation_scores", "manifest_row", "sweep_cache",
                    "equivalence_sweep", "sweep_scores"):  # fmt: skip
            if key in previous:
                run.setdefault(key, previous[key])
    save_run(run)
    steps = ("record",) if opts.record_only else STEPS if opts.step == "all" else (opts.step,)
    if not run["cached"]:
        steps = tuple(s for s in steps if s in ("train", "record"))
    row = None
    for step in steps:
        print(f"\n=== r3 {run['id']}: {step} ===", flush=True)
        if step == "cache":
            step_cache(run, opts)
        elif step == "equivalence":
            step_equivalence(run, opts)
        elif step == "train":
            step_train(run, opts)
        elif step == "ablations":
            step_ablations(run, opts)
        elif step == "sweep":
            step_sweep(run, opts)
        elif step == "share":
            step_share(run, opts)
        else:
            row = step_record(run, opts, opts.who)
            print(f"[pv] {row['id']} recorded: val {TARGET_NAME[run['task']]} AUROC "
                  f"{row['val_auroc']} AUPRC {row['val_auprc']} -> "
                  f"{manifest.manifest_path()}")  # fmt: skip
    return row


def cli(argv: list[str] | None = None) -> int:
    """Command-line entry: 0 when every step succeeded (errors raise SystemExit != 0)."""
    main(argv)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
