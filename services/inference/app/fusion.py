"""The trained MedPatch fusion models (``INFERENCE_MODE=fusion``), Round 3.

One ``CMSMAFusion`` (``--fusion_type c-msma``) per task, each from a folder
holding its checkpoint and the ``run.json`` the model track wrote
(docs/REAL_INFERENCE_PLAN.md):

    <FUSION_DIR>/mortality/    r3-fusion-001  EHR-CXR-RR
    <FUSION_DIR>/phenotyping/  r3-fusion-002  EHR-CXR-RR-DN  (pneumonia = class 21)

Nothing here reimplements training code. The model is rebuilt from the run's
own ``argv`` with medpatch's parser (dropping ``--load_*``, ``--resume`` and
``--save_dir``: the checkpoint already holds every weight), its encoders are
built exactly as ``MSMA_Trainer.__init__`` builds them, and the checkpoint is
loaded strictly. Inputs go through medpatch's Discretizer, the run's
Normalizer, ``DataFusion.pad_zeros`` (as ``my_collate``), the CXR test
transform and ``Text_encoder.process_notes`` (inside the model). A missing
modality is what training saw: an empty note, an all-zero image tensor.

The service refuses to load -- and so to predict -- a checkpoint whose sha256
differs from its ``run.json`` manifest row, a run trained with another BERT
than the one it names, or one whose thresholds are not θ = DEFAULT_THETA.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .readers import cxr_image, dn_text, ehr_rows
from .schemas import EHR_WINDOW_HOURS, EhrInput, NoteInput, PredictRequest, Task

logger = logging.getLogger("pneumovision.inference.fusion")

#: Platform task -> (folder under FUSION_DIR, medpatch --task, output index).
TASKS: dict[Task, tuple[str, str, int]] = {
    "mortality": ("mortality", "in-hospital-mortality", 0),
    # Pneumonia is CCS phenotype class 21 of the 25 phenotyping outputs.
    "pneumonia": ("phenotyping", "phenotyping", 21),
}
#: MSMA_Trainer.HEADS, in its order.
HEADS = ("high", "low", "late", "combined", "less_combined")
#: Phenotyping pads every stay to this many hourly rows (DataFusion.pad_zeros).
PHENOTYPING_MIN_ROWS = 2646

_DROP_WITH_VALUE = ("--save_dir",)
_DROP_FLAG = ("--resume",)


class FusionLoadError(RuntimeError):
    """A fusion model folder that must not serve predictions."""


class FusionInputError(ValueError):
    """The request cannot be put into the shape the model was trained on."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_argv(argv: list[str]) -> list[str]:
    """The run's fusion_main arguments without ``--load_*``, ``--resume`` and
    ``--save_dir`` (paths on the training machine; the checkpoint holds every
    weight the parents would have loaded)."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        takes_value = token.startswith("--load_") or token in _DROP_WITH_VALUE
        if takes_value:
            i += 2 if i + 1 < len(argv) and not argv[i + 1].startswith("--") else 1
            continue
        if token in _DROP_FLAG:
            i += 1
            continue
        out.append(token)
        i += 1
    return out


def _thresholds_in_argv(argv: list[str]) -> dict[str, float]:
    found = {}
    for flag, value in zip(argv, argv[1:]):
        if flag.startswith("--") and flag.endswith("_confidence_threshold"):
            found[flag] = float(value)
    return found


def normalizer_file(args: Any, medpatch_root: Path) -> Path:
    """The normalizer the run trained with, resolved as fusion_main.py does:
    ``--normalizer_state`` (a path on the training machine -- the same file is
    bundled under medpatch/normalizers), else the phenotyping default for the
    timestep, whatever the task (CLAUDE.md, "Model status")."""
    bundled = medpatch_root / "normalizers"
    if getattr(args, "normalizer_state", None):
        path = bundled / Path(str(args.normalizer_state).replace("\\", "/")).name
    else:
        path = bundled / f"ph_ts{args.timestep}.input_str_previous.start_time_zero.normalizer"
    if not path.is_file():
        raise FusionLoadError(f"normalizer {path.name} is not bundled under {bundled}")
    return path


@dataclass
class FusionScore:
    """Everything one forward pass produced, for the task's output class."""

    probability: float
    head: str
    heads: dict[str, float]
    unimodal: dict[str, float]
    present: dict[str, bool]
    missingness_prediction: float
    alphas: dict[str, float]
    notes: list[str] = field(default_factory=list)


@dataclass
class FusionModel:
    task: Task
    run_id: str
    sha256: str
    head: str
    bert_model_name: str
    theta: float
    modalities: list[str]
    output_index: int
    args: Any
    model: Any
    normalizer: Any
    discretizer: Any
    cxr_transform: Any
    device: str
    _captured: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def fusion(self) -> Any:
        return self.model.fusion_model

    def install_hooks(self) -> None:
        """Record the per-modality classifiers' and the missingness head's
        outputs: CMSMAFusion computes them inside forward() but returns only
        the three heads."""

        def keep(name: str):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                self._captured[name] = output.detach()

            return hook

        for m in self.modalities:
            getattr(self.fusion, f"{m.lower()}_classifier").register_forward_hook(keep(m))
        self.fusion.missingness_classifier.register_forward_hook(keep("missingness"))
        self._share_frozen_reader_passes()

    def _share_frozen_reader_passes(self) -> None:
        """CMSMAFusion.forward runs every reader twice -- ``ehr_model(x)`` then
        ``ehr_model_fixed(x)``, and likewise for the ViT and BERT -- and in
        Round 3 ``*_fixed`` *is* the same module object, called on the same
        input objects. In eval mode both calls return identical tensors, so
        the second is answered from the first (within one forward pass only;
        ``score`` clears the memo). Halves the CPU time; the parity tests
        compare against medpatch's own, unmemoised forward."""
        fusion = self.fusion
        self._memo: dict[tuple[Any, ...], Any] = {}
        for live, fixed in (("ehr_model", "ehr_model_fixed"), ("cxr_model", "cxr_model_fixed"),
                            ("text_model", "text_model_fixed")):  # fmt: skip
            module = getattr(fusion, live, None)
            if module is None or getattr(fusion, fixed, None) is not module:
                continue
            original = module.forward

            def forward(*args: Any, _original: Any = original, _name: str = live, **kwargs: Any) -> Any:
                key = (_name, *map(id, args), *sorted((k, id(v)) for k, v in kwargs.items()))
                if key not in self._memo:
                    self._memo[key] = _original(*args, **kwargs)
                return self._memo[key]

            module.forward = forward

    # ── inputs, as in training ──────────────────────────────────────────────

    def ehr_tensor(self, ehr: EhrInput | None) -> tuple[Any, list[int], list[str]]:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from datasets.DataFusion import pad_zeros  # type: ignore[import-not-found]  # noqa: PLC0415

        from .readers import ReaderInputError  # noqa: PLC0415

        notes: list[str] = []
        if self.task == "mortality":
            hours: int | None = EHR_WINDOW_HOURS
            end = float(EHR_WINDOW_HOURS)
        else:
            hours = None
            end = float(len(ehr.values) if ehr is not None and ehr.values else EHR_WINDOW_HOURS)
        try:
            rows = ehr_rows(ehr, self.discretizer, hours) if ehr is not None else []
        except ReaderInputError as exc:
            raise FusionInputError(str(exc)) from exc
        if not rows:
            # Training always had EHR; with nothing charted the discretizer
            # imputes every hour with its normal values (masks all zero).
            rows = [["0.5"] + [""] * (len(self.discretizer._header) - 1)]
            notes.append(
                "No EHR values were sent: the stay was fed as all-normal imputed "
                "values (training always had EHR)."
            )
        data = self.discretizer.transform(np.array(rows), end=end)[0]
        data = self.normalizer.transform(data)
        x, seq_lengths = pad_zeros([data], args=self.args)
        if self.task == "pneumonia":
            notes.append(
                f"Phenotyping was trained on the whole ICU stay; this request sent "
                f"{data.shape[0]} h, padded with zeros to {max(PHENOTYPING_MIN_ROWS, data.shape[0])} "
                "rows as in training."
            )
        return torch.from_numpy(x).float().to(self.device), seq_lengths, notes

    def image_tensor(self, cxr_bytes: bytes | None) -> Any:
        import torch  # noqa: PLC0415

        if cxr_bytes is None:
            # DataFusion.my_collate: a missing X-ray is a raw all-zero tensor,
            # which detect_missingness_batch recognises.
            return torch.zeros(1, 3, 384, 384, device=self.device)
        return self.cxr_transform(cxr_image(cxr_bytes))[None].to(self.device)

    # ── scoring ─────────────────────────────────────────────────────────────

    def score(self, req: PredictRequest, cxr_bytes: bytes | None = None) -> FusionScore:
        import torch  # noqa: PLC0415

        notes_in = req.notes or []
        if self.task == "mortality" and any(n.type == "discharge" for n in notes_in):
            raise FusionInputError("discharge notes are never used for mortality")
        x, seq_lengths, notes = self.ehr_tensor(req.ehr)
        img = self.image_tensor(cxr_bytes)
        reports = [n.text for n in notes_in if n.type == "radiology" and n.text.strip()]
        rr = [" ".join(reports)]
        dn = [(dn_text(notes_in) or "")] if "DN" in self.modalities else None
        if len(reports) > 1:
            notes.append(
                f"{len(reports)} radiology reports were joined with a space, in the order "
                "sent (chronological), as training joined a stay's reports."
            )
        ignored = sorted({n.type for n in notes_in if n.type in ("progress", "nursing")})
        if ignored:
            read = "radiology reports" + (" and discharge notes" if dn is not None else "")
            notes.append(
                f"{' and '.join(ignored).capitalize()} notes are not read by any model: "
                f"it was trained on {read} only."
            )

        with self._lock, torch.inference_mode():
            self._captured.clear()
            self._memo.clear()
            out = self.model(x, seq_lengths, img, [cxr_bytes is not None], rr, dn)
            self._memo.clear()
            missing = self.fusion.detect_missingness_batch(1, cxr=img, dn=dn, rr=rr)[0]
            weights = self.fusion.last_late_weights[0, :, self.output_index].float().cpu()
            captured = dict(self._captured)

        k = self.output_index
        logits = {
            "high": out["high_conf"].reshape(-1)[k],
            "low": out["low"].reshape(-1)[k],
            "late": out["late"].reshape(-1)[k],
        }
        logits["combined"] = (logits["high"] + logits["low"] + logits["late"]) / 3
        logits["less_combined"] = (logits["high"] + logits["late"]) / 2
        heads = {h: float(torch.sigmoid(v)) for h, v in logits.items()}

        present = {m.lower(): bool(missing[i]) for i, m in enumerate(self.modalities)}
        unimodal = {
            m.lower(): float(torch.sigmoid(captured[m].reshape(-1)[k]))
            for m in self.modalities
            if present[m.lower()]
        }
        absent = [m for m in self.modalities if not present[m.lower()]]
        if absent:
            notes.append(
                f"Not available: {', '.join(absent)}; fed as in training (empty note, "
                "all-zero image) and masked out of the late fusion."
            )

        # Late-fusion columns: modalities..., missingness, low, high (fuse()).
        columns = [m.lower() for m in self.modalities] + ["miss", "low", "high"]
        total = float(weights.sum()) or 1.0
        alphas = {c: float(w) / total for c, w in zip(columns, weights.tolist(), strict=True)}

        return FusionScore(
            probability=heads[self.head],
            head=self.head,
            heads=heads,
            unimodal=unimodal,
            present=present,
            missingness_prediction=float(torch.sigmoid(captured["missingness"].reshape(-1)[k])),
            alphas=alphas,
            notes=notes,
        )


class FusionModels:
    """The loaded fusion model of each task."""

    def __init__(self, models: dict[Task, FusionModel]) -> None:
        self.by_task = models

    def score(self, req: PredictRequest, cxr_bytes: bytes | None = None) -> FusionScore:
        return self.by_task[req.task].score(req, cxr_bytes)

    @property
    def text_model(self) -> str:
        return ", ".join(sorted({m.bert_model_name for m in self.by_task.values()}))

    @property
    def version(self) -> str:
        return "medpatch-r3:" + ",".join(
            f"{m.run_id}@{m.sha256[:12]}" for m in self.by_task.values()
        )


def _read_run(folder: Path) -> tuple[dict[str, Any], Path]:
    run_file = folder / "run.json"
    if not run_file.is_file():
        raise FusionLoadError(f"{run_file} is missing (infra/aws/copy-checkpoints.md)")
    checkpoints = sorted(folder.glob("*.pth.tar"))
    if len(checkpoints) != 1:
        raise FusionLoadError(f"{folder} must hold exactly one *.pth.tar (found {len(checkpoints)})")
    run = json.loads(run_file.read_text(encoding="utf-8"))
    return run, checkpoints[0]


def load_task(
    task: Task, folder: Path, medpatch_root: Path, device: str, theta: float
) -> FusionModel:
    """Build, verify and strictly load one task's fusion model, or raise."""
    started = time.perf_counter()
    run, checkpoint_path = _read_run(folder)
    row = run.get("manifest_row") or {}
    run_id = str(run.get("id") or "?")
    expected = str(row.get("sha256") or "")
    if not expected:
        raise FusionLoadError(f"{folder}/run.json has no manifest_row.sha256 (not a recorded run)")
    if row.get("id") not in (None, run_id):
        raise FusionLoadError(f"{folder}/run.json: manifest row {row.get('id')} is not run {run_id}")
    actual = sha256_file(checkpoint_path)
    if actual != expected:
        raise FusionLoadError(
            f"{checkpoint_path.name}: sha256 {actual[:12]}... does not match {run_id}'s "
            f"manifest row ({expected[:12]}...); refusing to serve it"
        )

    root = str(medpatch_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import torch  # noqa: PLC0415
    from arguments import args_parser  # type: ignore[import-not-found]  # noqa: PLC0415
    from datasets.cxr_dataset import get_transforms  # type: ignore[import-not-found]  # noqa: PLC0415
    from ehr_utils.preprocessing import Discretizer, Normalizer  # type: ignore[import-not-found]  # noqa: PLC0415
    from models.cxr_models import CXR_encoder  # type: ignore[import-not-found]  # noqa: PLC0415
    from models.ehr_models import EHR_encoder  # type: ignore[import-not-found]  # noqa: PLC0415
    from models.fusion import Fusion  # type: ignore[import-not-found]  # noqa: PLC0415
    from models.text_models import Text_encoder  # type: ignore[import-not-found]  # noqa: PLC0415

    argv = list(run.get("argv") or [])
    args = args_parser().parse_args(clean_argv(argv))
    _, medpatch_task, output_index = TASKS[task]
    if args.fusion_type != "c-msma":
        raise FusionLoadError(f"{run_id}: fusion_type {args.fusion_type}, not c-msma (Round 3)")
    if args.task != medpatch_task:
        raise FusionLoadError(f"{run_id} is a {args.task} run, not {medpatch_task}")
    modalities = args.modalities.split("-")
    if task == "mortality" and "DN" in modalities:
        raise FusionLoadError(f"{run_id}: mortality must never read discharge notes (F8)")

    bert = str(row.get("bert_model_name") or run.get("bert_model_name") or "")
    if not bert or bert != args.bert_model_name:
        raise FusionLoadError(
            f"{run_id}: run.json names BERT {bert or '(none)'} but its argv builds "
            f"{args.bert_model_name}; refusing a mismatched text encoder"
        )
    thresholds = _thresholds_in_argv(argv)
    for m in modalities:
        thresholds.setdefault(
            f"--{m.lower()}_confidence_threshold",
            float(getattr(args, f"{m.lower()}_confidence_threshold")),
        )
    wrong = {k: v for k, v in thresholds.items() if v != theta}
    if wrong:
        raise FusionLoadError(f"{run_id}: thresholds {wrong} differ from DEFAULT_THETA={theta}")

    # Exactly MSMA_Trainer.__init__'s encoders, then Fusion(...).
    ehr_encoder = cxr_encoder = text_encoder = None
    if "EHR" in args.modalities and args.ehr_encoder is not None:
        ehr_encoder = EHR_encoder(args)
    if "CXR" in args.modalities and args.cxr_encoder is not None:
        cxr_encoder = CXR_encoder(args)
    if ("RR" in args.modalities or "DN" in args.modalities) and args.text_encoder is not None:
        text_encoder = Text_encoder(args, device)
        loaded_name = str(getattr(text_encoder.tokenizer, "name_or_path", ""))
        if loaded_name != bert:
            raise FusionLoadError(f"{run_id}: tokenizer loaded {loaded_name}, run.json names {bert}")
    model = Fusion(args, ehr_encoder, cxr_encoder, text_encoder)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stored_bert = checkpoint.get("bert_model_name")
    if stored_bert not in (None, bert):
        raise FusionLoadError(f"{run_id}: checkpoint was trained with {stored_bert}, not {bert}")
    head = checkpoint.get("best_head") or ((run.get("scores") or {}).get("main") or {}).get("head")
    if head not in HEADS:
        raise FusionLoadError(f"{run_id}: no chosen head recorded (best_head={head!r})")
    # Strict: every key present, none extra, equal shapes -- never
    # Trainer.load_state, which is non-strict and broadcasts.
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model = model.to(device).eval()

    discretizer = Discretizer(
        timestep=float(args.timestep),
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
        config_path=str(medpatch_root / "ehr_utils" / "resources" / "discretizer_config.json"),
    )
    import numpy as np  # noqa: PLC0415

    probe = [["0.5"] + [""] * (len(discretizer._header) - 1)]
    header = discretizer.transform(np.array(probe), end=1.0)[1].split(",")
    normalizer = Normalizer(fields=[i for i, name in enumerate(header) if "->" not in name])
    normalizer.load_params(str(normalizer_file(args, medpatch_root)))

    fm = FusionModel(
        task=task,
        run_id=run_id,
        sha256=actual,
        head=str(head),
        bert_model_name=bert,
        theta=theta,
        modalities=modalities,
        output_index=output_index,
        args=args,
        model=model,
        normalizer=normalizer,
        discretizer=discretizer,
        cxr_transform=get_transforms(args)[1],
        device=device,
    )
    fm.install_hooks()
    logger.info(
        "loaded %s (%s, head %s, sha256 %s) for %s in %.1fs",
        run_id, args.modalities, head, actual[:12], task, time.perf_counter() - started,
    )  # fmt: skip
    return fm


def load_fusion(fusion_dir: Path, medpatch_root: Path, device: str, theta: float) -> FusionModels:
    """Both tasks' models, warmed up, or raise FusionLoadError."""
    models = {
        task: load_task(task, fusion_dir / folder, medpatch_root, device, theta)
        for task, (folder, _, _) in TASKS.items()
    }
    fused = FusionModels(models)
    warm_up(fused)
    return fused


def warm_up(models: FusionModels) -> None:
    """One synthetic pass per task (every modality present), so the first
    request is not the slow one."""
    from PIL import Image  # noqa: PLC0415

    buffer = io.BytesIO()
    Image.new("L", (600, 700), 128).save(buffer, format="PNG")
    ehr = EhrInput(variables=["Heart Rate"], values=[[80.0]] + [[None]] * (EHR_WINDOW_HOURS - 1))
    for task in models.by_task:
        notes = [NoteInput(id="warm-rr", type="radiology", text="Synthetic warm-up report.")]
        if task == "pneumonia":
            notes.append(NoteInput(id="warm-dn", type="discharge", text="Synthetic warm-up summary."))
        models.score(PredictRequest(task=task, ehr=ehr, notes=notes), cxr_bytes=buffer.getvalue())



# ── result ───────────────────────────────────────────────────────────────────

_LABELS = {
    "ehr": "EHR reader (LSTM)",
    "cxr": "Chest X-ray reader (ViT-Small)",
    "rr": "Radiology-report reader (BioBERT)",
    "dn": "Discharge-note reader (BioBERT)",
}


def build_fusion_result(req: PredictRequest, cxr: Any, fm: FusionModel, score: FusionScore) -> Any:
    """A PredictionResult whose fused, joint, missingness, α and unimodal
    numbers all come from the trained model. The explainability maps (token
    confidences on the report) stay simulated and are labelled so (decision 8,
    docs/REAL_INFERENCE_PLAN.md)."""
    from .mock import _build_confidence, _seeded_rng, build_received  # noqa: PLC0415
    from .schemas import (  # noqa: PLC0415
        Alphas,
        FusionModelInfo,
        JointScores,
        Missingness,
        ModalitySource,
        PredictionResult,
        Sources,
        UnimodalScores,
    )

    present = {m: score.present.get(m, False) for m in ("ehr", "cxr", "rr", "dn")}
    rng = _seeded_rng(req.seed or "no-seed", req.task, "fusion")
    confidence = _build_confidence(req, rng, present, req.notes or [])

    def clamp(p: float) -> float:
        return min(1.0, max(0.0, p))

    def source(m: str) -> ModalitySource | None:
        if not present[m]:
            return None
        return ModalitySource(kind="trained", model=f"{_LABELS[m]}, inside {fm.run_id}")

    alphas = {c: a for c, a in score.alphas.items() if c in ("miss", "low", "high") or present.get(c)}
    total = sum(alphas.values()) or 1.0
    alphas = {c: a / total for c, a in alphas.items()}
    return PredictionResult(
        probability=clamp(score.probability),
        unimodal=UnimodalScores(**{m: clamp(p) for m, p in score.unimodal.items()}),
        joint=JointScores(high=clamp(score.heads["high"]), low=clamp(score.heads["low"])),
        missingness=Missingness(vector=present, prediction=clamp(score.missingness_prediction)),
        alphas=Alphas(
            high=alphas["high"],
            low=alphas["low"],
            miss=alphas["miss"],
            **{m: alphas.get(m) for m in ("ehr", "cxr", "rr", "dn")},
        ),
        confidence=confidence,
        received=build_received(req, cxr, mode="model"),
        sources=Sources(fusion="trained", **{m: source(m) for m in ("ehr", "cxr", "rr", "dn")}),
        fusion_model=FusionModelInfo(
            run_id=fm.run_id,
            sha256=fm.sha256[:12],
            head=score.head,  # type: ignore[arg-type]
            heads={h: clamp(p) for h, p in score.heads.items()},
            bert_model_name=fm.bert_model_name,
            theta=fm.theta,
            modalities=fm.modalities,
            calibrated=False,
            notes=[
                f"Probability is the raw sigmoid of the {score.head} head (chosen on "
                "validation): an uncalibrated probability.",
                *score.notes,
                "Token-confidence maps on this report are simulated; the confidence "
                "patching inside the model used the real Round 2b confidences.",
            ],
        ),
    )
