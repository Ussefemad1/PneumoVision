"""Trained single-modality readers (``INFERENCE_MODE=readers``).

The team's Round 1 readers are trained and evaluated, but the MedPatch fusion
stages (Rounds 2, 2b, 3) are not. This module runs the readers that exist, so
each modality's card on the report shows a real model output, while the fused
prediction stays simulated and is labelled as such (see ``Sources``).

Nothing here reimplements training code. Models are built with medpatch's own
classes (``Fusion``, ``EHR_encoder``, ``Text_encoder``) from the same arguments
the readers were trained with, weights are loaded strictly, and the EHR window
goes through medpatch's ``Discretizer`` and the ``Normalizer`` each reader was
trained with. Inputs are synthetic or user-entered; nothing is logged.
"""

from __future__ import annotations

import io
import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .mock import ReaderScore
from .schemas import EHR_WINDOW_HOURS, EhrInput, NoteInput, PredictRequest, Task

logger = logging.getLogger("pneumovision.inference.readers")

Modality = Literal["ehr", "cxr", "rr", "dn"]

# Pneumonia is CCS phenotype index 21 of the 25 phenotyping outputs.
PNEUMONIA_INDEX = 21
TEXT_MODEL = "dmis-lab/biobert-v1.1"

_COMMON_ARGS = ["--classifier", "mlp", "--loss", "bce", "--data_pairs", "paired"]
_TEXT_ARGS = ["--text_encoder", "biobert", "--bert_model_name", TEXT_MODEL]
_MORTALITY_ARGS = ["--task", "in-hospital-mortality", "--num_classes", "1", "--labels_set", "mortality"]
_PHENO_ARGS = ["--task", "phenotyping", "--num_classes", "25", "--labels_set", "pheno"]
# --use_cls_token cls: the CXR classifier reads the CLS token, as in every
# released CXR script (scripts/*/Unimodal/CXR.sh) and Youssef's Round 2/2b
# args. The argparse default ("mean") pools all 577 tokens instead and gives
# different (wrong) scores from the same weights.
_CXR_ARGS = [
    "--fusion_type", "unimodal_cxr", "--modalities", "EHR-CXR",
    "--cxr_encoder", "vit_small_patch16_384", "--use_cls_token", "cls",
]


@dataclass(frozen=True)
class ReaderSpec:
    key: str
    modality: Modality
    task: Task
    label: str
    args: list[str]
    # Relative to CHECKPOINT_DIR.
    checkpoint: str
    # Which output to read: 0 for mortality, PNEUMONIA_INDEX for phenotyping.
    output_index: int
    # Held-out test AUROC for this task's output, from the training logs.
    # None when the trainer has not reported one yet.
    test_auroc: float | None
    # EHR only: the normalizer file the reader was trained with (medpatch/normalizers).
    normalizer: str | None = None
    # Youssef's Round 2 confidence head on this same reader, with Round 2b
    # temperatures when ``calibrated``. Relative to CHECKPOINT_DIR.
    confidence_checkpoint: str | None = None
    calibrated: bool = False


# Every reader that exists, with the exact arguments and normalizer it was
# trained with (from its args.txt / training log). Test AUROCs are for the
# output the platform reads: mortality, or the pneumonia phenotype.
READERS: list[ReaderSpec] = [
    ReaderSpec(
        key="ehr-mortality",
        modality="ehr",
        task="mortality",
        label="EHR reader (LSTM)",
        args=["--fusion_type", "unimodal_ehr", "--modalities", "EHR", "--ehr_encoder", "lstm", *_MORTALITY_ARGS],
        checkpoint="mortality/r1/in-hospital-mortality/unimodal_ehr/"
        "best_checkpoint_0.001_in-hospital-mortality_unimodal_ehr_EHR_paired.pth.tar",
        output_index=0,
        test_auroc=0.861,
        # Caroll's run loaded the task-matched normalizer (her fusion_main fix).
        normalizer="ihm_ts1.0.input_str_previous.start_time_zero.normalizer",
    ),
    ReaderSpec(
        key="ehr-pneumonia",
        modality="ehr",
        task="pneumonia",
        label="EHR reader (LSTM)",
        args=["--fusion_type", "unimodal_ehr", "--modalities", "EHR", "--ehr_encoder", "lstm", *_PHENO_ARGS],
        checkpoint="phenotyping/r1/phenotyping/unimodal_ehr/"
        "best_checkpoint_0.001_phenotyping_unimodal_ehr_EHR_paired.pth.tar",
        output_index=PNEUMONIA_INDEX,
        test_auroc=0.812,
        normalizer="ph_ts1.0.input_str_previous.start_time_zero.normalizer",
    ),
    # Norhan's ViT-Small readers. Of the learning rates she ran, 3e-5 had the
    # best validation AUROC (mortality: 0.723 vs 0.664 at 1e-4 and 0.558 at
    # 1e-3; phenotyping: 0.702 mean over 25 labels, the only run). No test
    # log came with them, so no test AUROC is shown yet.
    ReaderSpec(
        key="cxr-mortality",
        modality="cxr",
        task="mortality",
        label="Chest X-ray reader (ViT-Small)",
        args=[*_CXR_ARGS, *_MORTALITY_ARGS],
        checkpoint="mortality/r1/in-hospital-mortality/unimodal_cxr/"
        "best_checkpoint_3e-05_in-hospital-mortality_unimodal_cxr_EHR-CXR_paired.pth.tar",
        output_index=0,
        test_auroc=None,
    ),
    ReaderSpec(
        key="cxr-pneumonia",
        modality="cxr",
        task="pneumonia",
        label="Chest X-ray reader (ViT-Small)",
        args=[*_CXR_ARGS, *_PHENO_ARGS],
        checkpoint="phenotyping/r1/phenotyping/unimodal_cxr/"
        "best_checkpoint_3e-05_phenotyping_unimodal_cxr_EHR-CXR_paired.pth.tar",
        output_index=PNEUMONIA_INDEX,
        test_auroc=None,
    ),
    ReaderSpec(
        key="rr-mortality",
        modality="rr",
        task="mortality",
        label="Radiology-report reader (BioBERT)",
        args=["--fusion_type", "unimodal_rr", "--modalities", "EHR-RR", *_TEXT_ARGS, *_MORTALITY_ARGS],
        checkpoint="mortality/r1/in-hospital-mortality/unimodal_rr/"
        "best_checkpoint_0.00012225031390095305_in-hospital-mortality_unimodal_rr_EHR-RR_paired.pth.tar",
        output_index=0,
        test_auroc=0.730,
    ),
    ReaderSpec(
        key="rr-pneumonia",
        modality="rr",
        task="pneumonia",
        label="Radiology-report reader (BioBERT)",
        args=["--fusion_type", "unimodal_rr", "--modalities", "EHR-RR", *_TEXT_ARGS, *_PHENO_ARGS],
        checkpoint="phenotyping/r1/phenotyping/unimodal_rr/"
        "best_checkpoint_0.00012225031390095305_phenotyping_unimodal_rr_EHR-RR_paired.pth.tar",
        output_index=PNEUMONIA_INDEX,
        test_auroc=0.859,
    ),
    ReaderSpec(
        key="dn-pneumonia",
        modality="dn",
        task="pneumonia",
        label="Discharge-note reader (BioBERT)",
        args=["--fusion_type", "unimodal_dn", "--modalities", "EHR-DN", *_TEXT_ARGS, *_PHENO_ARGS],
        checkpoint="phenotyping/r1/phenotyping/unimodal_dn/"
        "best_checkpoint_0.00012225031390095305_phenotyping_unimodal_dn_EHR-DN_paired.pth.tar",
        output_index=PNEUMONIA_INDEX,
        test_auroc=0.897,
    ),
]


# Round 2b (calibrated) for every reader; Round 2 would be used for one whose
# 2b file was missing.
# Every Round 2 reader carries exactly the Round 1 weights above (checked
# tensor by tensor), plus the confidence head; load_readers re-checks the
# classifier at start-up.
_CONFIDENCE: dict[str, tuple[str, bool]] = {
    "ehr-mortality": ("mortality/r2b/temp_c-unimodal_ehr.pth.tar", True),
    "ehr-pneumonia": ("phenotyping/r2b/temp_c-unimodal_ehr.pth.tar", True),
    "cxr-mortality": ("mortality/r2b/temp_c-unimodal_cxr.pth.tar", True),
    "cxr-pneumonia": ("phenotyping/r2b/temp_c-unimodal_cxr.pth.tar", True),
    "rr-mortality": ("mortality/r2b/temp_c-unimodal_rr.pth.tar", True),
    "rr-pneumonia": ("phenotyping/r2b/temp_c-unimodal_rr.pth.tar", True),
    "dn-pneumonia": ("phenotyping/r2b/temp_c-unimodal_dn.pth.tar", True),
}
READERS = [
    replace(spec, confidence_checkpoint=path, calibrated=calibrated)
    for spec in READERS
    for path, calibrated in [_CONFIDENCE.get(spec.key, (None, False))]
]


def _confidence_args(args: list[str], calibrated: bool) -> list[str]:
    """The reader's own arguments with the Round 2 / 2b fusion type."""
    out = list(args)
    i = out.index("--fusion_type") + 1
    out[i] = ("temp_c-" if calibrated else "c-") + out[i]
    return out


class ReaderInputError(ValueError):
    """The request cannot be put into the shape a reader was trained on."""


# ── Inputs, built exactly as in training ─────────────────────────────────────


def rr_text(notes: list[NoteInput]) -> str | None:
    """Radiology reports joined with a space, as DataFusion builds RR.

    The RR readers were trained on radiology reports only, so progress and
    nursing notes are not fed to them (the fusion mock still counts them).
    """
    reports = [n.text for n in notes if n.type == "radiology" and n.text.strip()]
    return " ".join(reports) if reports else None


def dn_text(notes: list[NoteInput]) -> str | None:
    texts = [n.text for n in notes if n.type == "discharge" and n.text.strip()]
    return " ".join(texts) if texts else None


def cxr_image(data: bytes) -> Any:
    """The radiograph as training saw it, before the dataset transforms.

    MIMIC-CXR JPEGs were shrunk offline by ``medpatch/resize.py`` (width 512,
    aspect kept, re-saved as JPEG) and then opened with ``.convert('RGB')``.
    ``resize.py`` runs as a script on import, so its two steps are repeated
    here rather than imported; the dataset's own test transforms (384x384,
    ImageNet normalisation) are applied by the caller.
    """
    from PIL import Image  # noqa: PLC0415

    img = Image.open(io.BytesIO(data))
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        # 16-bit PNGs: stretch to 8-bit grey, as a JPEG export would.
        import numpy as np  # noqa: PLC0415

        arr = np.asarray(img, dtype=np.float64)
        span = float(arr.max() - arr.min()) or 1.0
        img = Image.fromarray(((arr - arr.min()) / span * 255).astype(np.uint8), "L")
    elif img.mode not in ("L", "RGB"):
        img = img.convert("RGB")

    width = 512
    img = img.resize((width, int(img.height * width / img.width)))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def _format_value(channel: str, value: float | str, categorical: bool, possible: list[str]) -> str:
    """One cell as the string a MIMIC timeseries CSV would hold."""
    if categorical:
        candidates = [value] if isinstance(value, str) else []
        if not isinstance(value, str):
            number = float(value)
            candidates = [str(value), f"{number:.1f}"]
            if number.is_integer():
                candidates.append(str(int(number)))
        for candidate in candidates:
            if candidate in possible:
                return candidate
        raise ReaderInputError(f"{channel}: {value!r} is not one of the trained categories")
    try:
        return str(float(value))
    except (TypeError, ValueError) as exc:
        raise ReaderInputError(f"{channel}: {value!r} is not a number") from exc


def ehr_rows(ehr: EhrInput, discretizer: Any) -> list[list[str]]:
    """The hourly grid as timeseries rows: ``[Hours, 17 channels...]``.

    Row ``i`` is hour bin ``i`` of the window. Its time is ``i + 0.5`` so the
    discretizer's ``int(t / timestep - eps)`` places it in bin ``i``. Hours
    with nothing charted are left out, as in the extracted CSVs; the
    discretizer then imputes them ('previous' strategy, as in training).
    """
    header: list[str] = discretizer._header  # ["Hours", *id_to_channel]
    column = {name: i for i, name in enumerate(header)}
    unknown = [v for v in ehr.variables if v not in column or v == "Hours"]
    if unknown:
        raise ReaderInputError(f"unknown EHR variables: {', '.join(unknown)}")

    rows: list[list[str]] = []
    for hour, values in enumerate(ehr.values[:EHR_WINDOW_HOURS]):
        row = [""] * len(header)
        row[0] = str(hour + 0.5)
        any_value = False
        for col, value in enumerate(values):
            if value is None or col >= len(ehr.variables):
                continue
            channel = ehr.variables[col]
            row[column[channel]] = _format_value(
                channel,
                value,
                discretizer._is_categorical_channel[channel],
                discretizer._possible_values[channel],
            )
            any_value = True
        if any_value:
            rows.append(row)
    return rows


# ── Loading ──────────────────────────────────────────────────────────────────


@dataclass
class _Loaded:
    spec: ReaderSpec
    model: Any
    normalizer: Any | None = None
    confidence_model: Any | None = None
    confidence_key: str | None = None
    tokenizer: Any | None = None


class Readers:
    """The loaded readers, keyed by (task, modality)."""

    def __init__(
        self, loaded: list[_Loaded], discretizer: Any, cxr_transform: Any, device: str
    ) -> None:
        self._by_key = {(r.spec.task, r.spec.modality): r for r in loaded}
        self._discretizer = discretizer
        self._cxr_transform = cxr_transform
        self._device = device

    @property
    def keys(self) -> list[str]:
        return [r.spec.key for r in self._by_key.values()]

    @property
    def text_model(self) -> str:
        """The BERT the text readers actually loaded (their tokenizer's name)."""
        names = {r.tokenizer.name_or_path for r in self._by_key.values() if r.tokenizer is not None}
        return ", ".join(sorted(names)) or "none"

    def score(self, req: PredictRequest, cxr_bytes: bytes | None = None) -> dict[str, ReaderScore]:
        """Unimodal probabilities for every modality a trained reader covers.

        ``cxr_bytes`` is the radiograph already decoded and hash-checked by
        ``cxr.decode_cxr``.
        """
        import torch  # noqa: PLC0415

        notes = req.notes or []
        inputs: dict[Modality, Any] = {}
        if cxr_bytes is not None:
            inputs["cxr"] = cxr_image(cxr_bytes)
        if req.ehr is not None:
            rows = ehr_rows(req.ehr, self._discretizer)
            if rows:
                inputs["ehr"] = rows
        if (text := rr_text(notes)) is not None:
            inputs["rr"] = text
        # Discharge notes leak the mortality outcome. The API never sends them
        # for mortality; there is also no mortality DN reader to send them to.
        if req.task == "pneumonia" and (text := dn_text(notes)) is not None:
            inputs["dn"] = text

        scores: dict[str, ReaderScore] = {}
        with torch.inference_mode():
            for modality, value in inputs.items():
                reader = self._by_key.get((req.task, modality))
                if reader is None:
                    continue
                kwargs = self._model_inputs(reader, modality, value)
                logits = reader.model(**kwargs)["unified"]
                probability = float(torch.sigmoid(logits).reshape(-1)[reader.spec.output_index])
                scores[modality] = ReaderScore(
                    probability=min(0.999, max(0.001, probability)),
                    model=reader.spec.label,
                    test_auroc=reader.spec.test_auroc,
                    token_confidence=self._token_confidence(reader, kwargs, value),
                    calibrated=reader.spec.calibrated,
                )
        return scores

    def _model_inputs(self, reader: _Loaded, modality: Modality, value: Any) -> dict[str, Any]:
        """The forward() arguments, shared by the reader and its confidence head."""
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if modality == "ehr":
            data = self._discretizer.transform(np.array(value), end=float(EHR_WINDOW_HOURS))[0]
            data = reader.normalizer.transform(data)
            x = torch.tensor(data[None], dtype=torch.float32, device=self._device)
            return {"x": x, "seq_lengths": [x.shape[1]]}
        if modality == "cxr":
            return {"img": self._cxr_transform(value)[None].to(self._device)}
        return {modality: [value]}

    def _token_confidence(
        self, reader: _Loaded, kwargs: dict[str, Any], value: Any
    ) -> tuple[float, ...] | None:
        """gamma = max(p, 1 - p) per token, as MedPatch §3.2 defines it."""
        import torch  # noqa: PLC0415

        if reader.confidence_model is None:
            return None
        logits = reader.confidence_model(**kwargs)[reader.confidence_key][0]
        if logits.dim() == 2:  # [tokens, classes]: this task's output only
            logits = logits[:, reader.spec.output_index]
        p = torch.sigmoid(logits.reshape(-1).float())
        gamma = torch.maximum(p, 1 - p)
        if reader.tokenizer is not None:
            # Text positions past the note are padding; the note's own tokens
            # (at most one 512-token window, pooled across chunks) are kept.
            real = min(gamma.shape[0], len(reader.tokenizer(value)["input_ids"]))
            gamma = gamma[:real]
        return tuple(float(g) for g in gamma.cpu())


def load_readers(medpatch_root: Path, checkpoint_dir: Path, device: str) -> Readers:
    """Build and load every reader, or raise.

    Refuses to start with a missing checkpoint rather than silently falling
    back to simulated numbers for that modality.
    """
    needed = [spec.checkpoint for spec in READERS] + [
        spec.confidence_checkpoint for spec in READERS if spec.confidence_checkpoint
    ]
    missing = [path for path in needed if not (checkpoint_dir / path).is_file()]
    if missing:
        raise RuntimeError(
            "INFERENCE_MODE=readers but these checkpoints are missing under "
            f"{checkpoint_dir}:\n  " + "\n  ".join(missing)
        )

    root = str(medpatch_root.resolve())
    if root not in sys.path:
        # Inserted first: medpatch's top-level packages (models, datasets, ...)
        # must win over any same-named installed package.
        sys.path.insert(0, root)

    import torch  # noqa: PLC0415
    from arguments import args_parser  # type: ignore[import-not-found]  # noqa: PLC0415
    from datasets.cxr_dataset import get_transforms  # type: ignore[import-not-found]  # noqa: PLC0415
    from ehr_utils.preprocessing import Discretizer, Normalizer  # type: ignore[import-not-found]  # noqa: PLC0415
    from models.cxr_models import CXR_encoder  # type: ignore[import-not-found]  # noqa: PLC0415
    from models.ehr_models import EHR_encoder  # type: ignore[import-not-found]  # noqa: PLC0415
    from models.fusion import Fusion  # type: ignore[import-not-found]  # noqa: PLC0415
    from models.text_models import Text_encoder  # type: ignore[import-not-found]  # noqa: PLC0415

    # Exactly fusion_main.py's discretizer.
    discretizer = Discretizer(
        timestep=1.0,
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
        config_path=str(medpatch_root / "ehr_utils" / "resources" / "discretizer_config.json"),
    )
    probe = [["0.5"] + [""] * (len(discretizer._header) - 1)]
    import numpy as np  # noqa: PLC0415

    header = discretizer.transform(np.array(probe), end=1.0)[1].split(",")
    continuous = [i for i, name in enumerate(header) if "->" not in name]

    shared_bert: Any = None
    cxr_transform: Any = None
    loaded: list[_Loaded] = []

    def build(spec: ReaderSpec, argv: list[str], path: str) -> tuple[Any, Any, Any]:
        """A medpatch model from its training arguments, strictly loaded."""
        nonlocal shared_bert, cxr_transform
        args = args_parser().parse_args([*argv, *_COMMON_ARGS])
        text_encoder = None
        if spec.modality == "ehr":
            model = Fusion(args, EHR_encoder(args), None, None)
        elif spec.modality == "cxr":
            model = Fusion(args, None, CXR_encoder(args), None)
            # The dataset's test-time transforms: Resize 384x384, CenterCrop,
            # ToTensor, ImageNet normalisation.
            cxr_transform = get_transforms(args)[1]
        else:
            text_encoder = Text_encoder(args, device)
            model = Fusion(args, None, None, text_encoder)

        checkpoint = torch.load(checkpoint_dir / path, map_location="cpu", weights_only=False)
        stored_bert = checkpoint.get("bert_model_name")
        if text_encoder is not None and stored_bert not in (None, TEXT_MODEL):
            raise RuntimeError(f"{spec.key}: {path} was trained with {stored_bert}, not {TEXT_MODEL}")
        # Strict: a key the model lacks, or one left at its initial value, is
        # a wrong architecture and must not produce numbers.
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint

        # The BERT inside every text model is frozen and identical, so they
        # share one copy (~440 MB instead of several) once confirmed equal.
        if text_encoder is not None:
            if shared_bert is None:
                shared_bert = text_encoder.bert
            else:
                if not _same_weights(shared_bert, text_encoder.bert):
                    raise RuntimeError(f"{spec.key}: {path} has a different frozen BERT")
                text_encoder.bert = shared_bert
        return model.to(device).eval(), args, text_encoder

    for spec in READERS:
        started = time.perf_counter()
        model, _, text_encoder = build(spec, spec.args, spec.checkpoint)

        confidence_model = confidence_key = None
        if spec.confidence_checkpoint:
            argv = _confidence_args(spec.args, spec.calibrated)
            confidence_model, cargs, _ = build(spec, argv, spec.confidence_checkpoint)
            confidence_key = cargs.fusion_type
            # The confidence head must sit on this very reader: its classifier
            # is copied from the Round 1 file the scores come from.
            if not _same_classifier(model, confidence_model):
                raise RuntimeError(
                    f"{spec.key}: {spec.confidence_checkpoint} was trained on a different reader"
                )

        normalizer = None
        if spec.normalizer is not None:
            normalizer = Normalizer(fields=continuous)
            normalizer.load_params(str(medpatch_root / "normalizers" / spec.normalizer))
        loaded.append(
            _Loaded(
                spec=spec,
                model=model,
                normalizer=normalizer,
                confidence_model=confidence_model,
                confidence_key=confidence_key,
                tokenizer=text_encoder.tokenizer if text_encoder is not None else None,
            )
        )
        logger.info("loaded reader %s in %.1fs", spec.key, time.perf_counter() - started)

    readers = Readers(loaded, discretizer, cxr_transform, device)
    _warm_up(readers)
    return readers


def _same_classifier(reader: Any, confidence_model: Any) -> bool:
    """Every ``*_classifier`` weight of the reader, unchanged in the head's file."""
    import torch  # noqa: PLC0415

    ra, rb = reader.state_dict(), confidence_model.state_dict()
    keys = [k for k in ra if "_classifier." in k]
    return bool(keys) and all(k in rb and torch.equal(ra[k].cpu(), rb[k].cpu()) for k in keys)


def _same_weights(a: Any, b: Any) -> bool:
    import torch  # noqa: PLC0415

    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k].cpu(), sb[k].cpu()) for k in sa)


def _warm_up(readers: Readers) -> None:
    """One synthetic pass through every reader, so the first request is fast."""
    synthetic = EhrInput(
        variables=["Heart Rate"],
        values=[[80.0]] + [[None]] * (EHR_WINDOW_HOURS - 1),
    )
    from PIL import Image  # noqa: PLC0415

    buffer = io.BytesIO()
    Image.new("L", (600, 700), 128).save(buffer, format="PNG")
    for task in ("mortality", "pneumonia"):
        readers.score(
            PredictRequest(
                task=task,
                ehr=synthetic,
                notes=[
                    NoteInput(id="warm-rr", type="radiology", text="Synthetic warm-up report."),
                    NoteInput(id="warm-dn", type="discharge", text="Synthetic warm-up summary."),
                ],
            ),
            cxr_bytes=buffer.getvalue(),
        )
