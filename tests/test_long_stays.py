"""EHR stays longer than the 2,646 learned temperature positions (phenotyping).

Phenotyping collate pads EHR to at least 2,646 hours but never truncates, and
MIMIC-IV has longer stays: ehr_temperature[:seq_len] was then shorter than the
sequence and the division crashed (Colab, r3-fusion-002, a 3,069-hour stay).
Positions past 2,646 now use temperature 1.0. Real LSTM, stub text/CXR readers,
SYNTHETIC tensors.
"""

from __future__ import annotations

import importlib
import json
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_round3_tools import lineage, opts  # noqa: F401 -- pytest fixture
from test_threshold_sweep import setup

from tools.pv import r3_cache, round3

# After tools.pv, which puts medpatch/ on sys.path.
fusion = importlib.import_module("models.fusion")

CPU = torch.device("cpu")
LONG = 3069


def old_slice(temperature, seq_len):  # what the code did before the fix
    return temperature[:seq_len]


class Stays(torch.utils.data.Dataset):
    """DataFusion items; `hours` per stay (one may be longer than 2,646)."""

    def __init__(self, hours, seed=0):
        g = np.random.default_rng(seed)
        self.items = []
        for i, h in enumerate(hours):
            img = torch.rand(3, 384, 384, generator=torch.Generator().manual_seed(seed * 100 + i))
            self.items.append((g.normal(size=(h, 76)).astype(np.float32), img + 0.1,
                               f"discharge {i}", f"report {i}", g.integers(0, 2, 25), None,
                               2, 1, 0, 20_000_000 + i))  # fmt: skip
        self.metadata_with_labels = {"stay_id": SimpleNamespace(values=np.arange(len(hours)))}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_for(args):
    from datasets.DataFusion import my_collate  # noqa: PLC0415

    return partial(my_collate, args=args)


# ── the helper ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", [(2646, 25), (48,)])
def test_temperature_for(shape):
    t = torch.rand(shape) + 0.5
    for n in (1, shape[0] - 1, shape[0]):
        assert torch.equal(fusion.temperature_for(t, n), t[:n])  # unchanged within range
    longer = fusion.temperature_for(t, shape[0] + 423)
    assert torch.equal(longer[: shape[0]], t)
    assert torch.equal(longer[shape[0] :], torch.ones((423,) + shape[1:]))


# ── a 3,069-hour stay: live and cached ───────────────────────────────────────


def test_round3_runs_a_stay_longer_than_2646_hours_live_and_cached():
    args, model, _ = setup("phenotyping", real_ehr=True)
    collate = collate_for(args)
    batch = collate([Stays([LONG, 40, 6])[i] for i in range(3)])
    assert batch[0].shape[1] == LONG and max(batch[6]) == LONG  # padded to the long stay
    report = r3_cache.compare_live_cached(model, batch, CPU)
    assert max(v for k, v in report.items() if k.startswith("max_abs")) <= 1e-5, report
    sweep = r3_cache.compare_live_cached(model, batch, CPU, threshold=0.65,
                                         sweep=[0.65, 0.75, 0.85])  # fmt: skip
    assert max(v for k, v in sweep.items() if k.startswith("max_abs")) <= 1e-5, sweep


@pytest.mark.parametrize("cls", ["c-msma", "c-e-msma"])
def test_bit_identical_for_stays_of_2646_hours_or_fewer(cls, monkeypatch):
    args, model, _ = setup("phenotyping", real_ehr=True, fusion_type=cls)
    batch = collate_for(args)([Stays([2646, 30, 7])[i] for i in range(3)])
    x, seq, img, pairs, rr, dn, _ = r3_cache.live_inputs(batch, CPU)
    model.eval()
    with torch.no_grad():
        new = model(x, seq, img, pairs, rr, dn)
        new_capture = r3_cache.capture_batch(model, batch, CPU) if cls == "c-msma" else None
        monkeypatch.setattr(fusion, "temperature_for", old_slice)
        old = model(x, seq, img, pairs, rr, dn)
        old_capture = r3_cache.capture_batch(model, batch, CPU) if cls == "c-msma" else None
    for key in ("high_conf", "low", "late"):
        assert torch.equal(new[key], old[key]), key
    if new_capture is not None:
        assert all(torch.equal(new_capture[k], old_capture[k]) for k in old_capture)


def test_round2b_ehr_model_takes_a_long_stay_and_is_unchanged_otherwise(monkeypatch):
    from models.ehr_models import EHR_encoder  # noqa: PLC0415
    from models.fusion import Fusion  # noqa: PLC0415

    from tools.pv.medpatch_bridge import parse_args  # noqa: PLC0415

    args = parse_args(["--fusion_type", "temp_c-unimodal_ehr", "--modalities", "EHR", "--task",
                       "phenotyping", "--num_classes", "25", "--ehr_encoder", "lstm",
                       "--classifier", "mlp"])  # fmt: skip
    torch.manual_seed(0)
    model = Fusion(args, EHR_encoder(args), None, None).eval()
    with torch.no_grad():
        out = model(torch.randn(1, LONG, 76), [LONG])["temp_c-unimodal_ehr"]
        assert out.shape[:2] == (1, LONG)
        x = torch.randn(2, 100, 76)
        new = model(x, [100, 60])["temp_c-unimodal_ehr"]
        monkeypatch.setattr(fusion, "temperature_for", old_slice)
        old = model(x, [100, 60])["temp_c-unimodal_ehr"]
    assert torch.equal(new, old)


# ── counts per split, kept across a resumed cache ────────────────────────────


def test_build_split_counts_long_stays_per_shard(tmp_path):
    args, model, _ = setup("phenotyping", real_ehr=True)
    info = r3_cache.build_split(model, Stays([10, LONG, 12, 9, 11, 2700, 8, 7]), "val", tmp_path,
                                batch_size=2, collate_fn=collate_for(args), device=CPU,
                                shard_size=4, long_after=2646)  # fmt: skip
    assert info["long_stays_new"] == {"0": 1, "1": 1}


class _Loader(list):
    def __init__(self, ds, collate, batch=2):
        super().__init__(collate([ds[j] for j in range(i, min(i + batch, len(ds)))])
                         for i in range(0, len(ds), batch))  # fmt: skip
        self.dataset, self.collate_fn, self.num_workers = ds, collate, 0


def test_cache_step_records_and_keeps_the_counts(lineage, monkeypatch, capsys):  # noqa: F811
    tmp_path, _ = lineage
    o = opts(tmp_path, "phenotyping", "--shard-size", "2", "--disk-margin-gb", "0")
    run = round3.plan(o)
    args, model, _ = setup("phenotyping", real_ehr=True, batch_size=2)
    args.ehr_data_dir = str(tmp_path)  # no listfiles here: their md5 reads "missing"
    collate = collate_for(args)
    splits = [Stays([10, LONG, 12, 9], 1), Stays([7, 8], 2), Stays([2900, 6], 3)]
    loaders = tuple(_Loader(ds, collate) for ds in splits)
    monkeypatch.setattr(round3, "trainer_args", lambda run_, *a, **k: args)
    monkeypatch.setattr(round3, "live_loaders", lambda a: loaders)
    monkeypatch.setattr(round3, "build_trainer",
                        lambda a, loaders_: SimpleNamespace(model=model, device=CPU))  # fmt: skip
    round3.step_cache(run, o)
    assert "train: 1 stays longer than 2646 h" in capsys.readouterr().out
    meta = r3_cache.read_meta(run["cache_dir"])
    assert [meta["splits"][s]["long_stays"] for s in ("train", "val", "test")] == [1, 0, 1]
    # resume: shard 1 of train kept (its count kept), shard 0 (the long stay) rebuilt
    (r3_cache.shard_path(run["cache_dir"], "train", 0)).unlink()
    round3.step_cache(run, o)
    meta = r3_cache.read_meta(run["cache_dir"])
    assert meta["splits"]["train"]["long_stays_by_shard"] == {"0": 1, "1": 0}
    assert meta["splits"]["train"]["long_stays"] == 1
    notes = "; ".join(round3.notes(run))
    assert "train: 1 stays longer than 2646 h (EHR temperature 1.0 beyond)" in notes
    assert any("2,646 learned temperature positions" in d for d in run["deviations"])
    assert json.loads(json.dumps(meta))  # meta.json stays plain JSON


def test_the_old_slice_crashes_on_the_long_stay(monkeypatch):
    """The Colab failure, reproduced: without the fix a 3,069-hour stay cannot run."""
    args, model, _ = setup("phenotyping", real_ehr=True)
    batch = collate_for(args)([Stays([LONG, 40])[i] for i in range(2)])
    x, seq, img, pairs, rr, dn, _ = r3_cache.live_inputs(batch, CPU)
    monkeypatch.setattr(fusion, "temperature_for", old_slice)
    with pytest.raises(RuntimeError, match="3069"), torch.no_grad():
        model.eval()(x, seq, img, pairs, rr, dn)
