"""Round 3 fixes F1-F7 (docs/model_track_notes.md, "Round 3"), on stub readers.

Builds real CMSMAFusion / EMSMAFusion models around the tiny readers in
r3_stubs.py and drives medpatch's own trainer code. CPU, seconds, nothing
downloaded; every number is SYNTHETIC.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from r3_stubs import D, r3_args, r3_batch, r3_model
from torch import nn

FUSIONS = ["c-msma", "c-e-msma"]


def run(model, batch):
    x, seq_lengths, img, pairs, rr, dn, _ = batch
    return model(x, seq_lengths, img, pairs, rr, dn)


# ── F1: Round 3 builds with a CXR encoder that has no projection_layer ────────


@pytest.mark.parametrize("fusion_type", FUSIONS)
def test_f1_builds_without_projection_layer(fusion_type, capsys):
    args, model = r3_model(fusion_type=fusion_type)
    fm = model.fusion_model
    assert not hasattr(fm.cxr_model, "cxr_encoder")
    assert "[F1]" in capsys.readouterr().out
    for proj in (fm.cxr_high_proj, fm.cxr_low_proj):
        assert (proj.in_features, proj.out_features) == (fm.cxr_model.full_feats_dim, D)
    before = {k: v.clone() for k, v in fm.cxr_high_proj.state_dict().items()}
    model.equalize()  # must not touch the CXR projections, must still copy fc_rr
    assert all(torch.equal(v, fm.cxr_high_proj.state_dict()[k]) for k, v in before.items())
    assert torch.equal(fm.rr_high_proj.weight, fm.text_model.fc_rr.weight)
    out = run(model, r3_batch(3))
    assert {k: tuple(v.shape) for k, v in out.items()} == {
        "high_conf": (3, 25),
        "low": (3, 25),
        "late": (3, 25),
    }


def test_f1_still_copies_a_real_projection_layer():
    from models.fusion import Fusion  # noqa: PLC0415
    from r3_stubs import StubCXR, StubEHR, StubText  # noqa: PLC0415

    args = r3_args()
    cxr = StubCXR()
    cxr.cxr_encoder = nn.Module()
    cxr.cxr_encoder.projection_layer = nn.Linear(cxr.full_feats_dim, D)
    model = Fusion(args, StubEHR(), cxr, StubText(args.modalities))
    model.equalize()
    assert torch.equal(
        model.fusion_model.cxr_low_proj.weight, cxr.cxr_encoder.projection_layer.weight
    )


# ── F4: late fusion masks each note with its own column ──────────────────────


def _constant(classifier: nn.Module, value: float):
    with torch.no_grad():
        classifier.classifier.fc.weight.zero_()
        classifier.classifier.fc.bias.fill_(value)


@pytest.mark.parametrize("fusion_type", FUSIONS)
def test_f4_rr_missing_dn_present_keeps_the_dn_weight(fusion_type):
    _, model = r3_model(fusion_type=fusion_type)
    fm = model.fusion_model
    for name in ("ehr_classifier", "cxr_classifier", "missingness_classifier",
                 "high_fuser_classifier", "low_fuser_classifier"):  # fmt: skip
        _constant(getattr(fm, name), 0.0)
    _constant(fm.dn_classifier, 5.0)
    _constant(fm.rr_classifier, -5.0)
    out = run(model, r3_batch(2, missing=[(0, "RR"), (1, "RR")]))
    # Uniform weights; present: EHR, CXR, DN + missingness, low, high = 6 columns.
    # Only DN is non-zero. Upstream normalises the weights over all
    # 6 x 25 (column, class) entries together, not per class, so
    # late = 5 / (6 * 25) (kept; see model_track_notes, "late weights").
    # With the RR/DN swap the RR prediction took DN's place: -5 / (6 * 25).
    torch.testing.assert_close(out["late"], torch.full((2, 25), 5 / (6 * 25)))


def test_f4_reorder_helper():
    from models.fusion import reorder_by_modalities  # noqa: PLC0415

    order = ["EHR", "CXR", "DN", "RR"]
    (got,) = reorder_by_modalities(["EHR", "CXR", "RR", "DN"], order, order)
    assert got == ["EHR", "CXR", "RR", "DN"]


# ── F5: phenotyping EHR confidence uses the temperature ──────────────────────


def test_f5_ehr_temperature_changes_the_pools():
    _, model = r3_model()
    fm = model.fusion_model
    with torch.no_grad():
        fm.ehr_confidence_predictor.confidence_layer.weight.mul_(20)
    batch = r3_batch(3)
    model.eval()
    with torch.no_grad():
        cold = run(model, batch)["high_conf"]
        fm.ehr_temperature.fill_(1e4)  # every EHR confidence -> 0.5 < 0.75: no high tokens
        hot = run(model, batch)["high_conf"]
    assert not torch.allclose(cold, hot), "EHR temperature has no effect (F5)"


@pytest.mark.parametrize("cls", ["CMSMAFusion", "EMSMAFusion"])
def test_f5_no_unscaled_ehr_sigmoid_left(cls):
    import models.fusion as fusion  # noqa: PLC0415

    code = "\n".join(
        line.split("#")[0] for line in inspect.getsource(getattr(fusion, cls)).splitlines()
    )
    assert "torch.sigmoid(ehr_conf_logits)" not in code


# ── F2: strict per-reader parent loading ─────────────────────────────────────


def _trainer(model, args):
    from trainers.trainer import Trainer  # noqa: PLC0415

    trainer = Trainer.__new__(Trainer)
    trainer.model, trainer.args, trainer.device = model, args, torch.device("cpu")
    return trainer


def _parents(model, tmp_path, task="phenotyping", seed=1):
    """One fake Round 2b file per reader: random values for exactly its keys."""
    from trainers.trainer import Trainer  # noqa: PLC0415

    own = model.state_dict()
    g = torch.Generator().manual_seed(seed)
    paths = {}
    readers = ["ehr", "cxr", "rr"] + (["dn"] if task == "phenotyping" else [])
    bert = {}
    for reader in readers:
        state = {}
        for k, v in own.items():
            if "_fixed." in k or not Trainer.reader_key(k, reader):
                continue
            canonical = k.split("text_model.", 1)[-1] if ".bert." in f".{k}" else None
            if canonical is not None:  # rr and dn carry the same frozen BERT
                bert.setdefault(canonical, torch.randn(v.shape, generator=g))
                state[k] = bert[canonical].clone()
            else:
                state[k] = torch.randn(v.shape, generator=g)
        # the top-level and fusion_model names of one module hold the same values
        for k in list(state):
            if k.startswith("fusion_model.") and k[len("fusion_model.") :] in state:
                state[k] = state[k[len("fusion_model.") :]].clone()
        path = tmp_path / f"r2b-{reader}.pth.tar"
        torch.save({"state_dict": state, "epoch": 3}, path)
        paths[reader] = path
    return paths


def _load(model, args, paths, **override):
    for reader, path in {**paths, **override}.items():
        setattr(args, f"load_{reader}", None if path is None else str(path))
    _trainer(model, args).load_reader_parents(args)


@pytest.mark.parametrize("task", ["phenotyping", "in-hospital-mortality"])
def test_f2_strict_load_is_bit_identical(task, tmp_path, capsys):
    args, model = r3_model(task)
    paths = _parents(model, tmp_path, task)
    _load(model, args, paths)
    out = capsys.readouterr().out
    for reader, path in paths.items():
        assert f"{reader}: " in out and f"tensors bit-identical to {path}" in out
    own = model.state_dict()
    ehr = torch.load(paths["ehr"])["state_dict"]
    assert torch.equal(
        own["fusion_model.ehr_model_fixed.lstm.weight"], ehr["ehr_model.lstm.weight"]
    )


def test_f2_another_readers_weights_in_a_file_are_ignored(tmp_path, capsys):
    args, model = r3_model()
    paths = _parents(model, tmp_path)
    rr = torch.load(paths["rr"])
    rr["state_dict"]["ehr_model.lstm.weight"] = torch.zeros(D, 76)  # untrained EHR in an RR file
    rr["state_dict"]["fusion_model.ehr_model.lstm.weight"] = torch.zeros(D, 76)
    torch.save(rr, paths["rr"])
    _load(model, args, paths)
    ehr = torch.load(paths["ehr"])["state_dict"]
    assert torch.equal(model.state_dict()["ehr_model.lstm.weight"], ehr["ehr_model.lstm.weight"])
    assert "ignored other keys in the file: {'ehr_model': 1, 'fusion_model': 1}" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "damage, message",
    [
        ("missing", "missing in the file"),
        ("shape", "in the file"),
        ("bert", "BERT weights differ"),
    ],
)
def test_f2_refuses(damage, message, tmp_path):
    args, model = r3_model()
    paths = _parents(model, tmp_path)
    reader = "dn" if damage == "bert" else "cxr"
    ckpt = torch.load(paths[reader])
    state = ckpt["state_dict"]
    if damage == "missing":
        del state["fusion_model.cxr_classifier.classifier.fc.bias"]
    elif damage == "shape":  # broadcastable: Tensor.copy_ would have accepted it
        state["fusion_model.cxr_classifier.classifier.fc.bias"] = torch.zeros(1)
    else:
        key = next(k for k in state if ".bert." in f".{k}")
        state[key] = state[key] + 1
    torch.save(ckpt, paths[reader])
    with pytest.raises(SystemExit, match=message):
        _load(model, args, paths)


def test_f2_refuses_dn_for_mortality(tmp_path):
    args, model = r3_model("in-hospital-mortality")
    paths = _parents(model, tmp_path, "in-hospital-mortality")
    with pytest.raises(SystemExit, match="never used for in-hospital mortality"):
        _load(model, args, paths, dn=paths["rr"])


def test_f2_load_model_routes_round3_to_the_strict_loader(tmp_path):
    args, model = r3_model()
    paths = _parents(model, tmp_path)
    for reader, path in paths.items():
        setattr(args, f"load_{reader}", str(path))
    trainer = _trainer(model, args)
    called = []
    trainer.load_reader_parents = lambda a: called.append(a)
    trainer.load_state = lambda **k: pytest.fail("non-strict load_state used for Round 3")
    trainer.load_model(args)
    assert called == [args]


# ── F3: head chosen on validation, reported on test ──────────────────────────


def _ret(auroc):
    scores = np.full(25, auroc)
    return {
        "auroc_mean": auroc,
        "auprc_mean": auroc / 2,
        "auc_scores": scores,
        "auprc_scores": scores / 2,
        "ci_auroc": [(auroc, auroc)] * 25,
        "ci_auprc": [(auroc, auroc)] * 25,
    }


def _msma(tmp_path, monkeypatch, best_head=None):
    import trainers.MSMA_trainer as msma  # noqa: PLC0415

    monkeypatch.setattr(msma.wandb, "log", lambda *a, **k: None)
    trainer = msma.MSMA_Trainer.__new__(msma.MSMA_Trainer)
    trainer.args = r3_args(save_dir=str(tmp_path), lr=0.001)
    trainer.val_dl, trainer.test_dl = "VAL", "TEST"
    trainer.weights = nn.Parameter(torch.ones(3))
    trainer.model = nn.Linear(1, 1)
    if best_head:
        trainer.best_head = best_head
    # val: 'late' wins; test: 'high' would win -- the old code reported 'high'.
    val = [_ret(a) for a in (0.60, 0.61, 0.70, 0.65, 0.64)]
    test = [_ret(a) for a in (0.90, 0.50, 0.55, 0.52, 0.51)]
    trainer.validate = lambda dl, t: (*(test if dl == "TEST" else val), 0.1)
    trainer.load_state = lambda **k: None
    reported = []
    trainer.print_and_write = lambda ret, **k: reported.append((ret, k["prefix"]))
    return trainer, reported


@pytest.mark.parametrize("stored", [None, "late"])
def test_f3_reports_the_validation_head_on_test(stored, tmp_path, monkeypatch, capsys):
    trainer, reported = _msma(tmp_path, monkeypatch, best_head=stored)
    trainer.eval()
    ((ret, prefix),) = reported
    assert ret["auroc_mean"] == 0.55 and "head late (chosen on validation)" in prefix
    table = (tmp_path / "test_all_heads_0.001.csv").read_text().splitlines()
    assert table[0].startswith("# ALL FIVE HEADS ON TEST -- reference only")
    assert len(table) == 2 + 5 and "late,1," in table[4] and "high,0,0.900000" in table[2]


def test_f3_best_head_round_trips_through_the_checkpoint(tmp_path):
    from trainers.trainer import Trainer  # noqa: PLC0415

    args, model = r3_model(save_dir=str(tmp_path), lr=0.001)
    trainer = _trainer(model, args)
    trainer.epoch, trainer.best_auroc, trainer.patience = 4, 0.7, 0
    trainer.weights = nn.Parameter(torch.ones(3))
    trainer.optimizer = torch.optim.Adam(model.parameters())
    trainer.best_head = "combined"
    trainer.save_checkpoint()
    fresh = _trainer(model, args)
    fresh.weights = nn.Parameter(torch.ones(3))
    Trainer.load_state(fresh, trainer.checkpoint_path())
    assert fresh.best_head == "combined"


def test_f3_stat_test_never_selects_on_test(tmp_path, monkeypatch):
    import trainers.MSMA_trainer as msma  # noqa: PLC0415

    src = inspect.getsource(msma.MSMA_Trainer.validate)
    assert "[F3] no head chosen on validation; refusing to" in src
    assert "max(all_rets.items()" not in src


# ── F7: a final batch of one ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, classes, expected",
    [((1, 1), 1, (1,)), ((4, 1), 1, (4,)), ((1, 25), 25, (1, 25)), ((4, 25), 25, (4, 25))],
)
def test_f7_fusion_logits_shapes(shape, classes, expected):
    from trainers.MSMA_trainer import MSMA_Trainer  # noqa: PLC0415

    fake = SimpleNamespace(args=SimpleNamespace(num_classes=classes))
    out = MSMA_Trainer.fusion_logits(fake, {"late": torch.zeros(shape)}, "late")
    assert tuple(out.shape) == expected


@pytest.mark.parametrize("task", ["in-hospital-mortality", "phenotyping"])
def test_f7_validate_survives_a_last_batch_of_one(task, tmp_path, monkeypatch):
    import trainers.MSMA_trainer as msma  # noqa: PLC0415
    from models.loss_set import Loss  # noqa: PLC0415

    monkeypatch.setattr(msma.wandb, "log", lambda *a, **k: None)
    args, model = r3_model(task, save_dir=str(tmp_path), bootstrap_iters=3)
    trainer = msma.MSMA_Trainer.__new__(msma.MSMA_Trainer)
    trainer.args, trainer.model, trainer.device = args, model, torch.device("cpu")
    trainer.epoch, trainer.loss = 0, Loss(args)
    trainer.weights = nn.Parameter(torch.ones(3))
    trainer.scheduler = SimpleNamespace(step=lambda *_: None)
    trainer.class_names = None

    def loader(sizes):
        batches = []
        for n, size in enumerate(sizes):
            x, seq, img, pairs, rr, dn, y = r3_batch(size, task, missing=[(0, "CXR")], seed=n)
            y = np.arange(size * y[0].numel()).reshape(y.shape) % 2  # both classes present
            batches.append((x.numpy(), img, dn, rr, y.astype(np.float32), None, seq, pairs,
                            None, None, None, None))  # fmt: skip
        return batches

    trainer.val_dl = loader([3, 1])  # 3 + 1: the last batch has one stay
    *rets, loss = trainer.validate(trainer.val_dl, False)
    assert len(rets) == 5 and np.isfinite(loss)
