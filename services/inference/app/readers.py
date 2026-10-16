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
from dataclasses import dataclass
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
_CXR_ARGS = ["--fusion_type", "unimodal_cxr", "--modalities", "EHR-CXR", "--cxr_encoder", "vit_small_patch16_384"]


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
                logits = self._forward(reader, modality, value)
                probability = float(torch.sigmoid(logits).reshape(-1)[reader.spec.output_index])
                scores[modality] = ReaderScore(
                    probability=min(0.999, max(0.001, probability)),
                    model=reader.spec.label,
                    test_auroc=reader.spec.test_auroc,
                )
        return scores

    def _forward(self, reader: _Loaded, modality: Modality, value: Any) -> Any:
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if modality == "ehr":
            data = self._discretizer.transform(np.array(value), end=float(EHR_WINDOW_HOURS))[0]
            data = reader.normalizer.transform(data)
            x = torch.tensor(data[None], dtype=torch.float32, device=self._device)
            return reader.model(x=x, seq_lengths=[x.shape[1]])["unified"]
        if modality == "cxr":
            x = self._cxr_transform(value)[None].to(self._device)
            return reader.model(img=x)["unified"]
        if modality == "rr":
            return reader.model(rr=[value])["unified"]
        return reader.model(dn=[value])["unified"]


def load_readers(medpatch_root: Path, checkpoint_dir: Path, device: str) -> Readers:
    """Build and load every reader, or raise.

    Refuses to start with a missing checkpoint rather than silently falling
    back to simulated numbers for that modality.
    """
    missing = [spec.checkpoint for spec in READERS if not (checkpoint_dir / spec.checkpoint).is_file()]
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
    for spec in READERS:
        started = time.perf_counter()
        args = args_parser().parse_args([*spec.args, *_COMMON_ARGS])
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

        checkpoint = torch.load(checkpoint_dir / spec.checkpoint, map_location="cpu", weights_only=False)
        stored_bert = checkpoint.get("bert_model_name")
        if text_encoder is not None and stored_bert not in (None, TEXT_MODEL):
            raise RuntimeError(f"{spec.key}: checkpoint was trained with {stored_bert}, not {TEXT_MODEL}")
        # Strict: a key the model lacks, or one left at its initial value, is
        # a wrong architecture and must not produce numbers.
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint

        # The BERT inside every text reader is frozen and identical, so the
        # readers share one copy (~440 MB instead of ~1.3 GB) once the weights
        # are confirmed equal.
        if text_encoder is not None:
            if shared_bert is None:
                shared_bert = text_encoder.bert
            else:
                if not _same_weights(shared_bert, text_encoder.bert):
                    raise RuntimeError(f"{spec.key}: its frozen BERT differs from the other readers'")
                text_encoder.bert = shared_bert

        model.to(device).eval()
        normalizer = None
        if spec.normalizer is not None:
            normalizer = Normalizer(fields=continuous)
            normalizer.load_params(str(medpatch_root / "normalizers" / spec.normalizer))
        loaded.append(_Loaded(spec=spec, model=model, normalizer=normalizer))
        logger.info("loaded reader %s in %.1fs", spec.key, time.perf_counter() - started)

    readers = Readers(loaded, discretizer, cxr_transform, device)
    _warm_up(readers)
    return readers


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
