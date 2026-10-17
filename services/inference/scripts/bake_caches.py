"""Download the pretrained caches into HF_HOME at image build time.

    HF_HOME=/opt/hf python services/inference/scripts/bake_caches.py

Fetches the BioBERT model + tokenizer and timm's ViT-small/16-384 weights
(app/offline.py names them), then re-checks them the way the service will at
start-up, with HF_HUB_OFFLINE=1, so a broken cache fails the build, not the
deploy.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.offline import BERT_MODEL, CXR_TIMM_MODEL, missing_caches  # noqa: E402

CHECK = f"""
import timm
from transformers import BertModel, BertTokenizerFast
BertModel.from_pretrained({BERT_MODEL!r})
BertTokenizerFast.from_pretrained({BERT_MODEL!r})
timm.create_model({CXR_TIMM_MODEL!r}, pretrained=True)
print('offline load OK')
"""


def main() -> int:
    if os.environ.get("HF_HUB_OFFLINE"):
        print("unset HF_HUB_OFFLINE to bake the caches", file=sys.stderr)
        return 2
    import timm
    from transformers import BertModel, BertTokenizerFast

    BertModel.from_pretrained(BERT_MODEL)
    BertTokenizerFast.from_pretrained(BERT_MODEL)
    timm.create_model(CXR_TIMM_MODEL, pretrained=True)

    missing = missing_caches([BERT_MODEL], [CXR_TIMM_MODEL])
    if missing:
        print(f"still missing after download: {missing}", file=sys.stderr)
        return 1
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    subprocess.run([sys.executable, "-c", CHECK], env=env, check=True)
    print(f"baked {BERT_MODEL} and {CXR_TIMM_MODEL} into {os.environ.get('HF_HOME')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
