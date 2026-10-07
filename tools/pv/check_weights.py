"""Strict weight-coverage check of a parent checkpoint against the model that loads it.

    python -m tools.pv.check_weights --reader rr --task phenotyping \\
        --file "/content/drive/MyDrive/<...>/best_checkpoint_..._unimodal_rr_EHR-RR.pth.tar" \\
        --bert-model-name dmis-lab/biobert-v1.1

Builds exactly the modules the reader's Round 2 run builds (``--stage r2``,
default; ``--stage r2b`` checks an r2 file against the Round 2b model), on CPU,
from the same paper-script argv -- no data, no forward pass. It then compares
the checkpoint's state_dict with the model's, applies the file with medpatch's
own ``Trainer.load_state`` (what ``--load_<reader>`` does) and confirms the
values landed.

Why this exists: ``Trainer.load_state`` is not strict. It copies every key
whose name matches, prints the rest as "Not Loaded" / "Not Found" and carries
on, and its ``Tensor.copy_`` *broadcasts*, so a ``[512]`` or ``[1, 512]`` tensor
silently fills a ``[25, 512]`` parameter; only non-broadcastable shapes raise.
This check fails instead.

Exit code 1 when any encoder or classifier weight of the reader is not covered
by the file or has a different shape, or -- for rr/dn -- when the frozen BERT
weights in the file are not the pretrained weights of ``--bert-model-name``.
The confidence head (and, for r2b, the temperature) are new in the stage being
checked, so their absence is listed as "expected new", not as a failure.
Keys in the file the model does not have are listed as "unused" (warning).

Needs the reader's pretrained encoder (BERT / the ViT) to build the model, so
run it where those can be downloaded (Colab), like training itself.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.pv import manifest  # noqa: E402
from tools.pv.import_checkpoint import load_checkpoint  # noqa: E402
from tools.pv.medpatch_bridge import parse_args  # noqa: E402
from tools.pv.paper_scripts import (  # noqa: E402
    READERS,
    TASK_ALIASES,
    build_argv,
    canonical_task,
    refuse_reader,
)

ENCODER_PREFIX = {
    "ehr": "ehr_model.",
    "cxr": "cxr_model.",
    "rr": "text_model.",
    "dn": "text_model.",
}


def role(key: str, reader: str) -> str:
    """encoder | classifier | expected_new | other, for one state_dict key."""
    if "confidence_predictor" in key or key.endswith("_temperature"):
        return "expected_new"
    if "_classifier." in key:
        return "classifier"
    bare = key[len("fusion_model.") :] if key.startswith("fusion_model.") else key
    if bare.startswith(ENCODER_PREFIX[reader]):
        return "encoder"
    return "other"


def model_args(reader: str, task: str, stage: str, bert_model_name: str | None):
    """medpatch args for the model a run of ``stage`` builds (paper-script argv)."""
    overrides: dict[str, str | None] = {}
    if reader in manifest.TEXT_READERS:
        overrides["--bert_model_name"] = bert_model_name or manifest.DEFAULT_BERT
    return parse_args(build_argv(stage, reader, overrides, task=task))


def make_encoders(args):
    """(ehr, cxr, text) encoders exactly as MSMA_Trainer.__init__ builds them."""
    from models.cxr_models import CXR_encoder  # noqa: PLC0415
    from models.ehr_models import EHR_encoder  # noqa: PLC0415
    from models.text_models import Text_encoder  # noqa: PLC0415

    device = torch.device("cpu")
    ehr = EHR_encoder(args) if "EHR" in args.modalities and args.ehr_encoder else None
    cxr = CXR_encoder(args) if "CXR" in args.modalities and args.cxr_encoder else None
    wants_text = "RR" in args.modalities or "DN" in args.modalities
    text = Text_encoder(args, device) if wants_text and args.text_encoder else None
    return ehr, cxr, text


def build_model(args) -> torch.nn.Module:
    from models.fusion import Fusion  # noqa: PLC0415

    return Fusion(args, *make_encoders(args))


@dataclass
class Report:
    reader: str
    covered: list[str] = field(default_factory=list)
    mismatched: list[tuple[str, tuple, tuple]] = field(default_factory=list)
    uncovered: list[str] = field(default_factory=list)  # model keys the file lacks
    unused: list[str] = field(default_factory=list)  # file keys the model lacks
    bert_differs: list[str] = field(default_factory=list)
    applied: bool = False

    def required(self) -> list[str]:
        return [
            k
            for k in self.covered + self.uncovered + [m[0] for m in self.mismatched]
            if role(k, self.reader) in ("encoder", "classifier")
        ]

    @property
    def failures(self) -> list[str]:
        out = [
            f"not in the file: {k}"
            for k in self.uncovered
            if role(k, self.reader) in ("encoder", "classifier")
        ]
        out += [f"shape {fs} in the file, {ms} in the model: {k}" for k, fs, ms in self.mismatched]
        out += [f"BERT weight differs from the pretrained model: {k}" for k in self.bert_differs]
        return out

    @property
    def coverage(self) -> float:
        required = self.required()
        ok = [k for k in self.covered if role(k, self.reader) in ("encoder", "classifier")]
        return 100.0 * len(ok) / len(required) if required else 100.0


def compare(
    state: dict[str, torch.Tensor],
    own: dict[str, torch.Tensor],
    reader: str,
    bert_reference: dict[str, torch.Tensor] | None = None,
) -> Report:
    rep = Report(reader=reader)
    for key, tensor in own.items():
        if key not in state:
            rep.uncovered.append(key)
        elif tuple(state[key].shape) != tuple(tensor.shape):
            rep.mismatched.append((key, tuple(state[key].shape), tuple(tensor.shape)))
        else:
            rep.covered.append(key)
    rep.unused = [k for k in state if k not in own]
    if bert_reference is not None:
        rep.bert_differs = [
            k
            for k, ref in bert_reference.items()
            if k in state
            and tuple(state[k].shape) == tuple(ref.shape)
            and not torch.equal(state[k].to(ref.dtype), ref)
        ]
    return rep


def apply_like_the_trainer(model, args, path: Path, state: dict, rep: Report) -> None:
    """Load via Trainer.load_state (the --load_* path) and confirm every covered value."""
    from trainers.trainer import Trainer  # noqa: PLC0415

    holder = SimpleNamespace(model=model, args=args, device=torch.device("cpu"))
    Trainer.load_state(holder, str(path))
    own = model.state_dict()
    wrong = [k for k in rep.covered if not torch.equal(own[k], state[k].to(own[k].dtype))]
    if wrong:
        raise SystemExit(f"Trainer.load_state did not apply {len(wrong)} covered keys: {wrong[:5]}")
    rep.applied = True


def check(
    reader: str,
    task: str,
    path: Path,
    stage: str = "r2",
    bert_model_name: str | None = None,
    trust_pickle: bool = False,
) -> Report:
    task = canonical_task(task)
    refuse_reader(reader, task)
    checkpoint = load_checkpoint(path, trust_pickle)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise SystemExit(f"{path} is not a medpatch checkpoint (no 'state_dict').")
    state = checkpoint["state_dict"]
    args = model_args(reader, task, stage, bert_model_name)
    model = build_model(args)
    own = model.state_dict()
    bert_reference = None
    if reader in manifest.TEXT_READERS:
        # Before loading, the model holds the pretrained weights of --bert_model_name.
        bert_reference = {k: v.clone() for k, v in own.items() if ".bert." in f".{k}"}
    rep = compare(state, own, reader, bert_reference)
    if not rep.mismatched:  # load_state would raise or silently broadcast otherwise
        apply_like_the_trainer(model, args, path, state, rep)
    return rep


def describe(rep: Report, path: Path, stage: str, task: str, bert: str | None) -> str:
    expected_new = [k for k in rep.uncovered if role(k, rep.reader) == "expected_new"]
    other_missing = [k for k in rep.uncovered if role(k, rep.reader) == "other"]
    required = rep.required()
    ok = sum(1 for k in rep.covered if role(k, rep.reader) in ("encoder", "classifier"))
    lines = [
        f"[check_weights] {rep.reader}  task {task}  as parent of {stage}  file {path}",
        f"  encoder + classifier tensors : {len(required)} required, "
        f"{ok} covered ({rep.coverage:.1f}%)",
        f"  keys loaded (name + shape)   : {len(rep.covered)}",
        f"  expected new (not in an {'r1' if stage == 'r2' else 'r2'} file): "
        f"{len(expected_new)}  {expected_new}",
    ]
    if bert:
        state = "identical" if not rep.bert_differs else f"{len(rep.bert_differs)} DIFFER"
        lines.append(f"  BERT weights vs pretrained {bert}: {state}")
    if other_missing:
        lines.append(f"  model keys not in the file (other): {other_missing}")
    if rep.unused:
        lines.append(
            f"  WARNING file keys the model does not have ({len(rep.unused)}): "
            f"{rep.unused[:10]}{' ...' if len(rep.unused) > 10 else ''}"
        )
    lines.append(
        "  applied with Trainer.load_state and verified: "
        + ("yes" if rep.applied else "no (shape mismatch; would raise or broadcast)")
    )
    if rep.failures:
        lines.append(f"  FAIL ({len(rep.failures)}):")
        lines += [f"    - {f}" for f in rep.failures[:20]]
    else:
        lines.append("  PASS")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Report:
    p = argparse.ArgumentParser(
        prog="python -m tools.pv.check_weights", description=__doc__.splitlines()[0]
    )
    p.add_argument("--reader", choices=READERS, required=True)
    p.add_argument("--task", choices=sorted(TASK_ALIASES), default="phenotyping")
    p.add_argument("--file", type=Path, required=True)
    p.add_argument(
        "--stage",
        choices=("r2", "r2b"),
        default="r2",
        help="the run that will load this file (r2: an r1 file; r2b: an r2 file)",
    )
    p.add_argument(
        "--bert-model-name",
        dest="bert_model_name",
        help="rr/dn: the BERT the file was trained with (default: medpatch's)",
    )
    p.add_argument("--trust-pickle", action="store_true")
    args = p.parse_args(argv)
    path = args.file.resolve()
    if not path.is_file():
        raise SystemExit(f"No such file: {path}")
    try:
        rep = check(
            args.reader, args.task, path, args.stage, args.bert_model_name, args.trust_pickle
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    bert = None
    if args.reader in manifest.TEXT_READERS:
        bert = args.bert_model_name or manifest.DEFAULT_BERT
    print(describe(rep, path, args.stage, canonical_task(args.task), bert))
    if rep.failures:
        raise SystemExit(1)
    return rep


if __name__ == "__main__":
    main()
