"""Round 3 cache (tools/pv/r3_cache.py): cached forward == live forward, and training on it.

Real CMSMAFusion around the stub readers in r3_stubs.py, real DataFusion
collate (missing X-ray -> zero image, missing note -> ""), real MSMA_Trainer.
CPU, seconds; every number is SYNTHETIC.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from r3_stubs import r3_model
from torch import nn

from tools.pv import r3_cache

TOL = 1e-5
ABLATIONS = [None, "without_joint_module", "without_missingness_module", "without_late_module"]


class StayDataset(torch.utils.data.Dataset):
    """DataFusion __getitem__ tuples for synthetic stays, some modalities missing."""

    def __init__(self, n, task, seed=0, missing_every=3):
        g = np.random.default_rng(seed)
        self.task = task
        self.items = []
        for i in range(n):
            pixels = torch.Generator().manual_seed(1000 * seed + i)
            img = (
                None if i % missing_every == 0 else torch.rand(3, 384, 384, generator=pixels) + 0.1
            )
            rr = "" if i % missing_every == 1 else f"report {seed}-{i}"
            dn = "" if i % missing_every == 2 else f"discharge {seed}-{i}"
            y = g.integers(0, 2, 25) if task == "phenotyping" else int(i % 2)
            self.items.append((g.normal(size=(5 + i % 3, 76)).astype(np.float32), img,
                               dn if task == "phenotyping" else None, rr, y, None,
                               2, 1, 0, 20_000_000 + i))  # fmt: skip
        self.metadata_with_labels = {"stay_id": SimpleNamespace(values=np.arange(n) + 30_000_000)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def setup(task, seed=0, **extra):
    from datasets.DataFusion import my_collate  # noqa: PLC0415

    args, model = r3_model(task, seed=seed, **extra)
    fm = model.fusion_model
    with torch.no_grad():  # confident heads, so both high and low groups are populated
        for m in args.modalities.split("-"):
            getattr(fm, f"{m.lower()}_confidence_predictor").confidence_layer.weight.mul_(10)
    model.equalize()
    collate = partial(my_collate, args=args)
    return args, model, collate


def batch_of(task, collate, n=6, seed=0):
    ds = StayDataset(n, task, seed)
    return collate([ds[i] for i in range(n)])


@pytest.mark.parametrize("task", ["phenotyping", "in-hospital-mortality"])
@pytest.mark.parametrize("ablation", ABLATIONS)
def test_cached_equals_live(task, ablation):
    extra = {"ablation": ablation} if ablation else {}
    args, model, collate = setup(task, **extra)
    report = r3_cache.compare_live_cached(model, batch_of(task, collate), torch.device("cpu"))
    for key in ("high_conf", "low", "late", "grad", "after_step"):
        assert report[f"max_abs_diff_{key}"] <= TOL, report


@pytest.mark.parametrize(
    "extra",
    [
        {"late_weights_per_class": None},
        {"mask_missing_tokens": None},
        {"fuser": "lstm"},
        {"fuser": "transformer"},
    ],
)
def test_cached_equals_live_with_options(extra):
    args, model, collate = setup("phenotyping", **extra)
    report = r3_cache.compare_live_cached(
        model, batch_of("phenotyping", collate), torch.device("cpu")
    )
    assert max(v for k, v in report.items() if k.startswith("max_abs_diff")) <= TOL, report


def test_capture_runs_each_reader_once(monkeypatch):
    args, model, collate = setup("phenotyping")
    fm = model.fusion_model
    calls = {"ehr": 0, "cxr": 0, "text": 0}
    for name, module in (("ehr", fm.ehr_model), ("cxr", fm.cxr_model), ("text", fm.text_model)):
        cls = type(module)

        def counted(self, *a, _n=name, _f=cls.forward, **k):  # real computations only
            calls[_n] += 1
            return _f(self, *a, **k)

        monkeypatch.setattr(cls, "forward", counted)
    r3_cache.capture_batch(model, batch_of("phenotyping", collate), torch.device("cpu"))
    # once each (twice without the reuse: *_model and *_model_fixed); text: dn + rr
    assert calls == {"ehr": 1, "cxr": 1, "text": 2}


# ── decision 1: --late_weights_per_class ─────────────────────────────────────


def _late(model, batch):
    x, img, dn, rr, *_rest = batch
    seq_lengths, pairs = batch[6], batch[7]
    model.eval()
    with torch.no_grad():
        out = model(torch.as_tensor(x).float(), seq_lengths, img, pairs, rr, dn)
    return out["late"], model.fusion_model.last_late_weights


def test_per_class_weights_identical_for_one_class():
    _, plain, collate = setup("in-hospital-mortality")
    _, flagged, _ = setup("in-hospital-mortality", late_weights_per_class=None)
    flagged.load_state_dict(plain.state_dict())
    batch = batch_of("in-hospital-mortality", collate)
    late_plain, w_plain = _late(plain, batch)
    late_flag, w_flag = _late(flagged, batch)
    assert torch.equal(late_plain, late_flag) and torch.equal(w_plain, w_flag)


# ── missingness module: an absent modality gets exactly zero weight ───────────


@pytest.mark.parametrize("task", ["phenotyping", "in-hospital-mortality"])
@pytest.mark.parametrize("per_class", [False, True])
def test_missing_modality_weight_is_zero(task, per_class):
    extra = {"late_weights_per_class": None} if per_class else {}
    args, model, collate = setup(task, **extra)
    batch = batch_of(task, collate)
    _, weights = _late(model, batch)  # [B, columns, classes]
    missingness = model.fusion_model.detect_missingness_batch(
        len(batch[2] or batch[3]), cxr=batch[1], dn=batch[2], rr=batch[3]
    )
    absent = missingness == 0
    assert absent.any(), "the batch must miss some modality"
    columns = missingness.shape[1]
    assert torch.all(weights[:, :columns][absent] == 0)  # exactly zero
    if per_class:
        torch.testing.assert_close(
            weights.sum(dim=1), torch.ones(weights.shape[0], weights.shape[2])
        )
    else:  # upstream: one total over columns x classes
        torch.testing.assert_close(weights.sum(dim=(1, 2)), torch.ones(weights.shape[0]))


# ── decision 3: --mask_missing_tokens ────────────────────────────────────────


def _with_empty_note_tokens(model, value):
    text = model.fusion_model.text_model
    original = text.tokens

    def tokens(notes):
        out = original(notes)
        for i, note in enumerate(notes):
            if not note.strip():
                out[i] = value
        return out

    text.tokens = tokens


@pytest.mark.parametrize("masked", [False, True])
def test_mask_missing_tokens_ignores_phantom_tokens(masked):
    extra = {"mask_missing_tokens": None} if masked else {}
    outs = []
    for value in (0.0, 3.0):  # what BERT makes of an empty note
        args, model, collate = setup("phenotyping", **extra)
        _with_empty_note_tokens(model, value)
        batch = batch_of("phenotyping", collate)
        x, img, dn, rr = batch[0], batch[1], batch[2], batch[3]
        model.eval()
        with torch.no_grad():
            out = model(torch.as_tensor(x).float(), batch[6], img, batch[7], rr, dn)
        outs.append(out["high_conf"] + out["low"])
    assert torch.allclose(outs[0], outs[1]) is masked


# ── shards, loader, training on the cache ────────────────────────────────────


def build(tmp_path, task, n=13, shard_size=8, **extra):
    args, model, collate = setup(task, **extra)
    splits = {"train": StayDataset(n, task, 1), "val": StayDataset(n, task, 2),
              "test": StayDataset(n, task, 3)}  # fmt: skip
    for split, ds in splits.items():
        r3_cache.build_split(model, ds, split, tmp_path, batch_size=4, collate_fn=collate,
                             device=torch.device("cpu"), shard_size=shard_size)  # fmt: skip
    return args, model, collate


def test_shards_resume_and_loader_order(tmp_path, capsys):
    args, model, collate = build(tmp_path, "phenotyping")
    assert sorted(p.name for p in (tmp_path / "val").iterdir()) == [
        "shard_00000.pt", "shard_00001.pt"
    ]  # fmt: skip
    (tmp_path / "val" / "shard_00001.pt").unlink()  # a disconnect mid-build
    capsys.readouterr()
    info = r3_cache.build_split(model, StayDataset(13, "phenotyping", 2), "val", tmp_path,
                                batch_size=4, collate_fn=collate, device=torch.device("cpu"),
                                shard_size=8)  # fmt: skip
    assert (info["written"], info["skipped"]) == (1, 1)

    val = r3_cache.CacheLoader(tmp_path, "val", 4, shuffle=False, drop_last=False, classes="c" * 25)
    sizes = [b["y"].shape[0] for b in val]
    assert sizes == [4, 4, 4, 1] and len(val) == 4  # last batch of one kept
    stays = torch.cat([torch.load(p)["stay_id"] for p in sorted((tmp_path / "val").glob("*.pt"))])
    assert stays.tolist() == list(range(30_000_000, 30_000_013))

    train = r3_cache.CacheLoader(tmp_path, "train", 4, shuffle=True, drop_last=True,
                                 classes="c" * 25)  # fmt: skip
    first, second = [torch.cat([b["y"] for b in train]) for _ in range(2)]
    assert len(train) == 3 and first.shape[0] == 12 and not torch.equal(first, second)


@pytest.mark.parametrize("task", ["phenotyping", "in-hospital-mortality"])
def test_cached_batches_equal_live_batches(task, tmp_path):
    """The stored shards reproduce the live val/test loader's outputs, batch by batch."""
    args, model, collate = build(tmp_path, task, n=13, shard_size=8)
    loader = r3_cache.CacheLoader(tmp_path, "test", 4, shuffle=False, drop_last=False,
                                  classes=["c"] * args.num_classes)  # fmt: skip
    ds = StayDataset(13, task, 3)
    model.eval()
    for b, cached in enumerate(loader):
        live = collate([ds[i] for i in range(4 * b, min(13, 4 * b + 4))])
        with torch.no_grad():
            out_cached = model.fusion_model.forward_cached(cached)
            x = torch.as_tensor(live[0]).float()
            out_live = model(x, live[6], live[1], live[7], live[3], live[2])
        for key in r3_cache.HEAD_KEYS:
            assert (out_cached[key] - out_live[key]).abs().max() <= TOL, (b, key)


def test_shard_size_must_be_whole_batches(tmp_path):
    args, model, collate = setup("in-hospital-mortality")
    with pytest.raises(ValueError, match="multiple of batch_size"):
        r3_cache.build_split(model, StayDataset(5, "in-hospital-mortality"), "val", tmp_path,
                             batch_size=4, collate_fn=collate, device=torch.device("cpu"),
                             shard_size=6)  # fmt: skip


def _cached_trainer(tmp_path, task, **extra):
    import trainers.MSMA_trainer as msma  # noqa: PLC0415
    from models.loss_set import Loss  # noqa: PLC0415

    args, model, _ = build(tmp_path / "cache", task, **extra)
    args.save_dir, args.epochs, args.bootstrap_iters, args.lr = str(tmp_path / "run"), 2, 3, 0.01
    classes = [f"class {i}" for i in range(args.num_classes)]
    trainer = msma.MSMA_Trainer.__new__(msma.MSMA_Trainer)
    msma.Trainer.__init__(trainer, args)
    trainer.args, trainer.model, trainer.device = args, model, torch.device("cpu")
    trainer.epoch, trainer.start_epoch, trainer.patience = 0, 0, 0
    trainer.loss, trainer.best_auroc, trainer.best_stats = Loss(args), 0, None
    trainer.class_names = None
    trainer.train_dl = r3_cache.CacheLoader(tmp_path / "cache", "train", 4, shuffle=True,
                                            drop_last=True, classes=classes)  # fmt: skip
    trainer.val_dl = r3_cache.CacheLoader(tmp_path / "cache", "val", 4, shuffle=False,
                                          drop_last=False, classes=classes)  # fmt: skip
    trainer.test_dl = r3_cache.CacheLoader(tmp_path / "cache", "test", 4, shuffle=False,
                                           drop_last=False, classes=classes)  # fmt: skip
    trainer.weights = nn.Parameter(torch.ones(3))
    trainer.optimizer = torch.optim.Adam(list(model.parameters()) + [trainer.weights], args.lr)
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer)
    return trainer


@pytest.mark.parametrize("task", ["phenotyping", "in-hospital-mortality"])
@pytest.mark.parametrize("ablation", ABLATIONS)
def test_msma_trains_and_evaluates_on_the_cache(task, ablation, tmp_path, monkeypatch):
    import trainers.MSMA_trainer as msma  # noqa: PLC0415

    monkeypatch.setattr(msma.wandb, "log", lambda *a, **k: None)
    extra = {"ablation": ablation} if ablation else {}
    trainer = _cached_trainer(tmp_path, task, **extra)
    readers = [trainer.model.fusion_model.text_model, trainer.model.fusion_model.cxr_model]
    for module in readers:
        module.forward = lambda *a, **k: pytest.fail("a reader ran during cached training")
    trainer.train()
    trainer.eval()
    best = torch.load(trainer.checkpoint_path(), weights_only=False)
    assert best["best_head"] in trainer.HEADS
    assert (tmp_path / "run" / "test_all_heads_0.01.csv").is_file()


def test_resumed_finished_run_does_not_crash(tmp_path, monkeypatch):
    """--resume restores no best_stats; a run with nothing left to train used to crash."""
    import trainers.MSMA_trainer as msma  # noqa: PLC0415

    monkeypatch.setattr(msma.wandb, "log", lambda *a, **k: None)
    trainer = _cached_trainer(tmp_path, "in-hospital-mortality")
    trainer.start_epoch = trainer.args.epochs  # as after resuming a finished run
    trainer.train()  # must not raise on best_stats None
