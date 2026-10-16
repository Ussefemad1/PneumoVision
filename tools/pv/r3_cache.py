"""Round 3 cache: the frozen readers' outputs, computed once per stay.

In Round 3 (CMSMAFusion, c-msma) every reader, confidence head and temperature
is frozen, yet the released loop runs the ViT and BioBERT (each text encoder
twice: ``text_model`` and ``text_model_fixed`` are the same module) on every
stay in every epoch -- several hours per phenotyping epoch on an L4. Only the
unimodal classifiers, the high/low projections, the fusers, the missingness
head and the weights train, and none has dropout.

So the readers run once, in eval mode, under ``torch.no_grad()``, and what the
trainable part needs is stored per stay (``CMSMAFusion._capture``):

- classifier inputs: ``<m>_feats``;
- EHR: the pooled high/low vectors (``ehr_high`` / ``ehr_low``; no projection);
- CXR / RR / DN: the masked means of the unprojected token features per class
  and group (``<m>_high_mean`` / ``<m>_low_mean``) and the token counts
  (``<m>_n_high`` / ``<m>_n_low``). The projections are linear, so
  ``masked_mean(W x + b) = W masked_mean(x) + b`` for a non-empty group, and
  forward() gives 0 for an empty one; ``forward_cached`` rebuilds both;
- ``<m>_conf_high`` / ``<m>_conf_low`` (only the lstm/transformer fusers use them);
- ``missingness`` (detect_missingness_batch), ``y`` and the stay ids.

Shards of ``shard_size`` stays per split are written atomically and skipped
when present, so a disconnect loses at most one shard. ``meta.json`` holds the
fingerprint (parents' sha256, thresholds, task, lists' md5, capture code
commit, ...) and training refuses a cache whose fingerprint differs.

SYNTHETIC data only in tests; on real data this runs on Colab.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterator

import torch

from tools.pv.medpatch_bridge import REPO_ROOT

CACHE_FORMAT = 1
#: Bump when the values stored under an existing key change. The capture-code
#: commit is recorded for provenance only: adding keys (the threshold sweep)
#: must not invalidate a cache whose stored values the code still reproduces.
CAPTURE_SEMANTICS = 1
SPLITS = ("train", "val", "test")
#: Code that decides what is captured: a cache built from other code is refused.
CAPTURE_CODE = (
    "medpatch/models",
    "medpatch/datasets",
    "medpatch/ehr_utils",
    "medpatch/trainers/trainer.py",
    "tools/pv/r3_cache.py",
)
#: Captured keys that are not per-stay float tensors.
INDEX_KEYS = ("stay_id", "index")


# ── capture ──────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def reuse_reader_calls(fusion_model) -> Iterator[None]:
    """Run each frozen reader once per batch.

    forward() calls ehr_model and ehr_model_fixed (and the CXR / text pairs)
    with the same input objects, and the pairs are the same modules, so the
    second call repeats the first exactly. Within this context each reader
    remembers its outputs per call signature (input object identities) and
    returns them on the repeat. Use only under no_grad with the readers in eval.
    """
    readers = {
        id(m): m
        for m in (fusion_model.ehr_model, fusion_model.cxr_model, fusion_model.text_model)
        if m is not None
    }
    memos: list[dict] = []
    for module in readers.values():
        memo: dict = {}
        memos.append(memo)
        original = module.forward

        def remembered(*args, _original=original, _memo=memo, **kwargs):
            key = tuple(id(a) for a in args) + tuple((k, id(v)) for k, v in sorted(kwargs.items()))
            if key not in _memo:
                # keep the inputs alive so their ids cannot be reused this batch
                _memo[key] = (args, kwargs, _original(*args, **kwargs))
            return _memo[key][2]

        module.forward = remembered
    try:
        yield
    finally:
        for module in readers.values():
            del module.forward  # back to the class method
        for memo in memos:
            memo.clear()


def live_inputs(batch, device):
    """(x, seq_lengths, img, pairs, rr, dn, y) from a DataFusion collate batch."""
    x, img, dn, rr, y_ehr, _y_cxr, seq_lengths, pairs = batch[:8]
    x = torch.as_tensor(x).float().to(device)
    y = torch.as_tensor(y_ehr).float().to(device)
    return x, seq_lengths, img.to(device), pairs, rr, dn, y


@torch.no_grad()
def capture_batch(model, batch, device, thresholds=()) -> dict[str, torch.Tensor]:
    """The cached tensors for one collate batch (CPU, fp32; counts as fp32 too).

    `thresholds`: the sweep. Besides the run's own (main) high/low groups, one
    group per other threshold is captured (CMSMAFusion._capture_sweep), so any
    of them can be rebuilt later with at_threshold().
    """
    fm = model.fusion_model
    x, seq_lengths, img, pairs, rr, dn, y = live_inputs(batch, device)
    fm._capture = {}
    fm._capture_thresholds = tuple(thresholds) or None
    try:
        with reuse_reader_calls(fm):
            model(x, seq_lengths, img, pairs, rr, dn)
        captured = fm._capture
    finally:
        fm._capture = None
        fm._capture_thresholds = None
    out = {k: v.detach().to("cpu", torch.float32) for k, v in captured.items()}
    out["y"] = y.detach().cpu()
    return out


# ── threshold sweep: any captured threshold from the main groups ─────────────


def sweep_key(name: str, t: float) -> str:
    return f"{name}_t{round(t * 100):03d}"


def _group_keys(name: str) -> tuple[str, str]:
    return ("ehr_high", "ehr_low") if name == "ehr" else (f"{name}_high_mean", f"{name}_low_mean")


def modality_names(batch: dict) -> list[str]:
    return sorted(k[: -len("_n_high")] for k in batch if k.endswith("_n_high"))


def at_threshold(batch: dict, t: float, main: float) -> dict:
    """The batch with every modality's high/low groups split at `t` instead of `main`.

    t == main returns the batch itself: the stored main groups, bit for bit.
    Otherwise t's captured group -- per stay and class, whichever side of t has
    fewer tokens (``_is_high`` says which) -- is used as stored, and the other
    side is the total minus it, in float64: total = main high sum + main low sum
    (mean x count). That side holds at least half the tokens, so the rounding
    of the stored fp32 means costs ~2 * eps * max|x|, whatever the stay length.
    Counts are exact; an empty side is exactly 0, as forward() gives.
    """
    if abs(t - main) < 1e-9:
        return batch
    out = dict(batch)
    for name in modality_names(batch):
        high_key, low_key = _group_keys(name)
        key = sweep_key(name, t)
        if f"{key}_mean" not in batch:
            raise KeyError(f"threshold {t} was not captured for {name} (no {key}_mean)")
        hi, lo = batch[high_key].double(), batch[low_key].double()
        nh, nl = batch[f"{name}_n_high"].double(), batch[f"{name}_n_low"].double()
        ch, cl = batch[f"{name}_conf_high"].double(), batch[f"{name}_conf_low"].double()
        s_mean, s_n = batch[f"{key}_mean"].double(), batch[f"{key}_n"].double()
        s_conf = batch[f"{key}_conf"].double()
        is_high = batch[f"{key}_is_high"] > 0.5  # shaped like the counts
        o_n = nh + nl - s_n
        o_mean = torch.where(o_n > 0, (hi * nh + lo * nl - s_mean * s_n) / o_n.clamp_min(1), 0.0)
        c_n, c_s, c_o = nh.squeeze(-1), s_n.squeeze(-1), o_n.squeeze(-1)
        conf_total = ch * c_n + cl * nl.squeeze(-1)
        o_conf = torch.where(c_o > 0, (conf_total - s_conf * c_s) / c_o.clamp_min(1), 0.0)
        hi_c = is_high.squeeze(-1)  # the confidences have no trailing axis
        groups = (
            torch.where(is_high, s_mean, o_mean),
            torch.where(is_high, o_mean, s_mean),
            torch.where(is_high, s_n, o_n),
            torch.where(is_high, o_n, s_n),
            torch.where(hi_c, s_conf, o_conf),
            torch.where(hi_c, o_conf, s_conf),
        )
        for k, v in zip(
            (high_key, low_key, f"{name}_n_high", f"{name}_n_low",
             f"{name}_conf_high", f"{name}_conf_low"),
            groups, strict=True,
        ):  # fmt: skip
            out[k] = v.to(batch[k].dtype)
    return out


def main_threshold(args) -> float:
    """The run's threshold; the sweep needs one shared by every modality."""
    mods = [m.lower() for m in args.modalities.split("-")]
    values = {getattr(args, f"{m}_confidence_threshold") for m in mods}
    if len(values) != 1:
        raise ValueError(f"the threshold sweep needs one threshold for all modalities: {values}")
    return values.pop()


def set_threshold(args, t: float) -> None:
    for m in ("ehr", "cxr", "rr", "dn"):
        setattr(args, f"{m}_confidence_threshold", t)


# ── live vs cached ───────────────────────────────────────────────────────────

HEAD_KEYS = ("high_conf", "low", "late")


def _loss(trainer_args, out, y, num_classes):
    """The trainer's Round 3 loss with equal head weights (gradient comparison only)."""
    import torch.nn.functional as F  # noqa: PLC0415

    total = 0.0
    for key in HEAD_KEYS:
        pred = out[key].reshape(out[key].shape[0], -1)
        target = y.reshape(pred.shape)
        total = total + F.binary_cross_entropy_with_logits(pred, target)
    return total


def compare_live_cached(
    model, batch, device, lr: float = 1e-3, threshold: float | None = None, sweep=()
) -> dict[str, float]:
    """Max |live - cached| for high_conf / low / late, their gradients, and one SGD step.

    `threshold` (with `sweep`, the captured thresholds): compare at that
    threshold instead of the run's own -- the live model splits at it, the
    cached batch is rebuilt with at_threshold().

    The step is plain SGD, linear in the gradient. Adam's first step is about
    lr * sign(g), so a near-zero gradient that differs only by float round-off
    (~1e-8) can move a weight by a sizeable fraction of lr -- noise, not a cache
    difference. The gradients themselves are compared directly.

    Readers in eval mode for both. Starts from the same trainable weights
    (deep copies); returns differences only -- no data.
    """
    import copy  # noqa: PLC0415

    model.eval()
    captured = capture_batch(model, batch, device, thresholds=sweep)
    main = main_threshold(model.args)
    if threshold is not None:
        captured = at_threshold(captured, threshold, main)
    cached_batch = {k: v.to(device) for k, v in captured.items()}
    live_model, cached_model = copy.deepcopy(model), copy.deepcopy(model)
    if threshold is not None:
        set_threshold(live_model.fusion_model.args, threshold)
    x, seq_lengths, img, pairs, rr, dn, y = live_inputs(batch, device)
    report: dict[str, float] = {}

    out_live = live_model(x, seq_lengths, img, pairs, rr, dn)
    out_cached = cached_model.fusion_model.forward_cached(cached_batch)
    for key in HEAD_KEYS:
        report[f"max_abs_diff_{key}"] = float((out_live[key] - out_cached[key]).abs().max())

    num_classes = model.args.num_classes
    trainable = [n for n, p in live_model.named_parameters() if p.requires_grad]
    opt_live = torch.optim.SGD([p for p in live_model.parameters() if p.requires_grad], lr)
    opt_cached = torch.optim.SGD([p for p in cached_model.parameters() if p.requires_grad], lr)
    _loss(None, out_live, y, num_classes).backward()
    _loss(None, out_cached, y, num_classes).backward()
    live_params = dict(live_model.named_parameters())
    cached_params = dict(cached_model.named_parameters())
    grad_diff = 0.0
    for name in trainable:
        g_live, g_cached = live_params[name].grad, cached_params[name].grad
        if g_live is None and g_cached is None:
            continue
        g_live = torch.zeros_like(live_params[name]) if g_live is None else g_live
        g_cached = torch.zeros_like(cached_params[name]) if g_cached is None else g_cached
        grad_diff = max(grad_diff, float((g_live - g_cached).abs().max()))
    report["max_abs_diff_grad"] = grad_diff
    opt_live.step()
    opt_cached.step()
    step_diff = max(
        float((live_params[n].detach() - cached_params[n].detach()).abs().max()) for n in trainable
    )
    report["max_abs_diff_after_step"] = step_diff
    report["trainable_tensors"] = float(len(trainable))
    report["stays"] = float(y.shape[0])
    return report


# ── fingerprint ──────────────────────────────────────────────────────────────


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_code_commit() -> str:
    """Last commit touching the capture code, '+dirty' if it has local changes."""
    try:
        commit = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", *CAPTURE_CODE],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", *CAPTURE_CODE],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("+dirty" if dirty else "")


def fingerprint(args, parents: dict[str, dict[str, str]], sweep=()) -> dict:
    """What a cache depends on. `parents`: reader -> {"id", "sha256"}; `sweep`: thresholds."""
    lists = {}
    for split in SPLITS:
        path = Path(args.ehr_data_dir) / args.task / f"{split}_listfile.csv"
        lists[split] = md5_file(path) if path.is_file() else "missing"
    return {
        "cache_format": CACHE_FORMAT,
        "task": args.task,
        "modalities": args.modalities,
        "num_classes": args.num_classes,
        "data_pairs": args.data_pairs,
        "thresholds": {
            m: getattr(args, f"{m}_confidence_threshold") for m in ("ehr", "cxr", "rr", "dn")
        },
        "cxr_token_confidence": bool(getattr(args, "cxr_token_confidence", False)),
        "bert_model_name": getattr(args, "bert_model_name", ""),
        "missing_image_as_absent": bool(getattr(args, "missing_image_as_absent", False)),
        "timestep": args.timestep,
        "parents": {r: parents[r]["sha256"] for r in sorted(parents)},
        "lists_md5": lists,
        "sweep_thresholds": sorted(float(t) for t in sweep),
        "capture_semantics": CAPTURE_SEMANTICS,
        "capture_code_commit": capture_code_commit(),
    }


#: Fingerprint keys a cache written before they existed is taken to have.
FINGERPRINT_DEFAULTS = {"sweep_thresholds": [], "capture_semantics": 1}
#: Recorded, not compared (see CAPTURE_SEMANTICS).
PROVENANCE_ONLY = ("capture_code_commit",)


def fingerprint_mismatch(saved: dict, current: dict) -> list[str]:
    saved = {**FINGERPRINT_DEFAULTS, **saved}
    keys = sorted((set(saved) | set(current)) - set(PROVENANCE_ONLY))
    return [f"{k}: cache {saved.get(k)!r}, now {current.get(k)!r}" for k in keys
            if saved.get(k) != current.get(k)]  # fmt: skip


def stays_bytes(sample: dict[str, torch.Tensor]) -> float:
    """Bytes per stay of a captured batch (what a shard stores per stay, fp32)."""
    n = sample["y"].shape[0]
    return sum(v.numel() * v.element_size() for v in sample.values()) / n


# ── shards ───────────────────────────────────────────────────────────────────


def shard_path(cache_dir: Path, split: str, k: int) -> Path:
    return Path(cache_dir) / split / f"shard_{k:05d}.pt"


def write_meta(cache_dir: Path, meta: dict) -> None:
    path = Path(cache_dir) / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_meta(cache_dir: Path) -> dict | None:
    path = Path(cache_dir) / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def concat(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def build_split(model, dataset, split, cache_dir, *, batch_size, collate_fn, device,
                shard_size=1024, num_workers=0, log=print, thresholds=()) -> dict:  # fmt: skip
    """Capture one split into shards; skip the ones already written. Returns counts."""
    from torch.utils.data import DataLoader, Subset  # noqa: PLC0415

    if shard_size % batch_size:
        # EHR pooling includes padded hours (upstream) and collate pads to the
        # longest stay in the batch, so a stay's cached EHR pool depends on its
        # batch-mates. Shards made of whole batches keep val/test batches
        # exactly those of the live (unshuffled) loader.
        raise ValueError(f"shard_size {shard_size} must be a multiple of batch_size {batch_size}")
    n = len(dataset)
    stay_ids = _stay_ids(dataset, n)
    shards = (n + shard_size - 1) // shard_size
    model.eval()
    done = skipped = 0
    started = time.time()
    for k in range(shards):
        path = shard_path(cache_dir, split, k)
        if path.is_file():
            skipped += 1
            continue
        idx = list(range(k * shard_size, min(n, (k + 1) * shard_size)))
        dl = DataLoader(
            Subset(dataset, idx), batch_size, shuffle=False, collate_fn=collate_fn,
            num_workers=num_workers, drop_last=False,
        )  # fmt: skip
        parts = [capture_batch(model, batch, device, thresholds) for batch in dl]
        shard = concat(parts)
        shard["stay_id"] = torch.as_tensor(stay_ids[idx[0] : idx[-1] + 1], dtype=torch.int64)
        shard["index"] = torch.as_tensor(idx, dtype=torch.int64)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(shard, tmp)
        os.replace(tmp, path)  # same folder: atomic, never a half-written shard
        done += 1
        free = shutil.disk_usage(path.parent).free
        log(
            f"[r3 cache] {split} shard {k + 1}/{shards}: {len(idx)} stays, "
            f"{path.stat().st_size / 2**20:.1f} MB; cache {dir_bytes(cache_dir) / 2**30:.2f} GB, "
            f"free {free / 2**30:.1f} GB, {time.time() - started:.0f}s"
        )
    size = dir_bytes(Path(cache_dir) / split) if (Path(cache_dir) / split).is_dir() else 0
    log(f"[r3 cache] {split}: {n} stays, {shards} shards, measured {size / 2**30:.2f} GB "
        f"({size / max(n, 1) / 1024:.0f} KB per stay)")  # fmt: skip
    return {"stays": n, "shards": shards, "written": done, "skipped": skipped, "bytes": size}


def _stay_ids(dataset, n):
    """Stay id per dataset index (DataFusion's metadata row order), or -1."""
    meta = getattr(dataset, "metadata_with_labels", None)
    if meta is not None and "stay_id" in meta:
        values = meta["stay_id"].values
        return [int(values[i]) if i < len(values) else -1 for i in range(n)]
    return [-1] * n


def split_complete(cache_dir: Path, split: str, meta: dict) -> bool:
    info = (meta.get("splits") or {}).get(split)
    if not info:
        return False
    return all(shard_path(cache_dir, split, k).is_file() for k in range(info["shards"]))


def split_summary(cache_dir: Path, split: str, modalities: list[str]) -> dict:
    """Presence counts and the share of tokens in the high group, per modality."""
    present = None
    n_high = {m: 0.0 for m in modalities}
    n_all = {m: 0.0 for m in modalities}
    stays = 0
    for path in sorted((Path(cache_dir) / split).glob("shard_*.pt")):
        shard = torch.load(path, map_location="cpu", weights_only=True)
        miss = shard["missingness"]
        present = miss.sum(0) if present is None else present + miss.sum(0)
        stays += miss.shape[0]
        for m in modalities:
            name = m.lower()
            n_high[m] += float(shard[f"{name}_n_high"].sum())
            n_all[m] += float(shard[f"{name}_n_high"].sum() + shard[f"{name}_n_low"].sum())
    return {
        "stays": stays,
        "present": {m: int(present[i]) for i, m in enumerate(modalities)} if stays else {},
        "share_high": {m: (n_high[m] / n_all[m] if n_all[m] else float("nan")) for m in modalities},
    }


class CacheLoader:
    """Batches of cached stays (dicts of tensors) for MSMA_Trainer.batch_outputs.

    With `threshold`, every batch is split at that swept threshold
    (at_threshold); without, at the run's own.

    Shuffled: a fresh permutation of shards and of the stays inside each shard
    every epoch (torch.Generator seeded by seed + epoch), batches carried across
    shard boundaries, last partial batch dropped -- like the train DataLoader
    (shuffle=True, drop_last=True). Unshuffled: dataset order, last partial
    batch kept -- like val/test. `preload` keeps every shard in memory.
    """

    def __init__(self, cache_dir, split, batch_size, *, shuffle, drop_last, classes,
                 seed=379647, preload=False, threshold=None, main=None):  # fmt: skip
        self.paths = sorted((Path(cache_dir) / split).glob("shard_*.pt"))
        if not self.paths:
            raise SystemExit(f"[r3 cache] no shards for {split} in {cache_dir}")
        self.batch_size, self.shuffle, self.drop_last = batch_size, shuffle, drop_last
        self.seed, self.epoch = seed, 0
        # Threshold sweep: batches split at `threshold` (from main's groups).
        self.threshold, self.main = threshold, main
        self.dataset = _Classes(classes)
        self._kept = [self._load(p) for p in self.paths] if preload else None
        sizes = [int(self._shard(i)["y"].shape[0]) for i in range(len(self.paths))]
        self.stays = sum(sizes)

    @staticmethod
    def _load(path):
        return torch.load(path, map_location="cpu", weights_only=True)

    def _shard(self, i):
        return self._kept[i] if self._kept is not None else self._load(self.paths[i])

    def __len__(self):
        full, rest = divmod(self.stays, self.batch_size)
        return full if self.drop_last or not rest else full + 1

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        order = (
            torch.randperm(len(self.paths), generator=g).tolist()
            if self.shuffle
            else range(len(self.paths))
        )
        buffer: list[dict[str, torch.Tensor]] = []
        held = 0
        for i in order:
            shard = {k: v for k, v in self._shard(i).items() if k not in INDEX_KEYS}
            n = shard["y"].shape[0]
            rows = torch.randperm(n, generator=g) if self.shuffle else torch.arange(n)
            start = 0
            while start < n:
                take = min(self.batch_size - held, n - start)
                sel = rows[start : start + take]
                buffer.append({k: v[sel] for k, v in shard.items()})
                held += take
                start += take
                if held == self.batch_size:
                    yield self._at(concat(buffer))
                    buffer, held = [], 0
        if buffer and not self.drop_last:
            yield self._at(concat(buffer))

    def _at(self, batch):
        if self.threshold is None:
            return batch
        return at_threshold(batch, self.threshold, self.main)


class _Classes:
    def __init__(self, classes):
        self.CLASSES = list(classes)
