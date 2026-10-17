"""Extract the Round 1 standalone readers' classifier heads for fusion mode.

    python -m tools.pv.extract_reader_heads --checkpoint-dir checkpoints --out checkpoints/round3

Run once on a laptop (repo .venv, CPU, no training, no data). Reads the seven
Round 1 reader checkpoints in the readers layout under --checkpoint-dir
(services/inference/READERS.md, infra/aws/copy-checkpoints.md) and writes

    <out>/reader_heads.pt    each reader's classifier-head tensors, the args and
                             normalizer it needs, and a digest of every encoder
                             tensor (a few MB)
    <out>/reader_heads.json  each source checkpoint's name and sha256, and the
                             .pt file's sha256

Copy both to the server's checkpoints/round3/. The inference service verifies
the sha256, checks that the fusion models' encoders equal the readers' bit for
bit, and then shows each reader's own score under "By source".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "services" / "inference"))
sys.path.insert(0, str(REPO / "medpatch"))

from app import reader_heads  # noqa: E402
from app.readers import READERS  # noqa: E402

#: The Round 1 files the Round 3 runs descend from (team manifest r1 rows, the
#: parents of the r2b parents of r3-fusion-001 / -002), by reader key.
LINEAGE = {
    "ehr-pneumonia": ("r1-ehr-001", "1dc8b962f3e7bbc0059424e22b39f99038bce14491ba73b611fde7dcc1953e74"),
    "cxr-pneumonia": ("r1-cxr-001", "065ecd77330b7a156fe14341aef691a32994b15cce0fd0c3f5548380ded31c40"),
    "rr-pneumonia": ("r1-rr-001", "ad4037e57564f48d40131e38eeb8144d320d63125567b79d8f7e911edd478c75"),
    "dn-pneumonia": ("r1-dn-001", "9bd950e3cb71e9d354ab5e62c12f525f10e607dbcdc007c92e4f9010fdc57817"),
    "ehr-mortality": ("r1-ehr-002", "1e9cb87cc1f27468be2e05d4346a924eccdd63df78b0bbac1595c1a640638756"),
    "cxr-mortality": ("r1-cxr-002", "4b272fcba45093e7178866cc013c1d55b6b385dcd4d75870c455b77dcdfdc75a"),
    "rr-mortality": ("r1-rr-002", "1092a832942d2f24b8f1e79efdb47301690f115dde84527e25216c3934f81a70"),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.pv.extract_reader_heads", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    p.add_argument("--checkpoint-dir", type=Path, default=REPO / "checkpoints",
                   help="folder holding mortality/r1/... and phenotyping/r1/... (default: checkpoints/)")  # fmt: skip
    p.add_argument("--out", type=Path, default=REPO / "checkpoints" / "round3",
                   help="where reader_heads.pt/.json go (default: checkpoints/round3/)")  # fmt: skip
    args = p.parse_args(argv)
    print("Round 1 readers expected under", args.checkpoint_dir)
    for spec in READERS:
        path = args.checkpoint_dir / spec.checkpoint
        print(f"  {spec.key:<14} {'ok     ' if path.is_file() else 'MISSING'} {spec.checkpoint}")
    try:
        manifest = reader_heads.extract(args.checkpoint_dir, args.out)
    except reader_heads.ReaderHeadsError as exc:
        print(f"\nextract_reader_heads: {exc}", file=sys.stderr)
        return 1
    heads = args.out / reader_heads.HEADS_FILE
    print(f"\nwrote {heads} ({heads.stat().st_size / 2**20:.1f} MiB), sha256 {manifest['sha256']}")
    print(f"wrote {args.out / reader_heads.MANIFEST_FILE}")
    for source in manifest["sources"]:
        row, expected = LINEAGE.get(source["reader"], ("?", ""))
        lineage = f"= manifest {row}" if source["sha256"] == expected else (
            f"WARNING: not manifest {row} ({expected[:12]}); the service still checks the "
            "encoders bit for bit and falls back if they differ"
        )
        print(f"  {source['reader']:<14} {source['sha256'][:12]}  {lineage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
