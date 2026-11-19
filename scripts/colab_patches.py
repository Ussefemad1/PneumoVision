"""Patches to apply in Colab after `git clone`, before training.

Both notebooks clone `medpatch/` fresh from GitHub on every run, so editing a
local copy has no effect on training. Paste the block you want as a new cell
between the clone cell and the training cell.

Each patch is idempotent: running it twice changes nothing the second time, and
it prints what it did so a silent no-op is impossible to miss.

----------------------------------------------------------------------------
PATCH 1 -- augmentation (ablation for the CXR track)
----------------------------------------------------------------------------

    from pathlib import Path

    f = Path("/content/PneumoVision/medpatch/datasets/cxr_dataset.py")
    s = f.read_text()
    before = s

    # A chest radiograph is acquired with the patient squared to the detector.
    # Real positioning varies by roughly 10 degrees, not 45, so the upstream
    # setting spends capacity on poses that never occur at test time.
    s = s.replace(
        "transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)),",
        "transforms.RandomAffine(degrees=10, scale=(.9, 1.1), shear=0, translate=(0.05, 0.05)),",
    )

    # A mirror flip swaps left and right. Cardiac silhouette and lobe anatomy
    # are not left-right symmetric, so the flipped image is not a plausible
    # chest film.
    s = s.replace("        transforms.RandomHorizontalFlip(),\\n", "")

    f.write_text(s)
    print("augmentation patch:", "applied" if s != before else "NO CHANGE (already patched?)")
    import re
    print(re.search(r"train_transforms = transforms.Compose\\(\\[.*?\\]\\)", s, re.S).group())

Run the baseline first and keep the number, then run this and compare. Change
nothing else, so the comparison has one variable.

RESULT (18 Nov 2026, pneumonia-48h, unimodal_cxr, lr 3e-5): **no improvement**.
Test pneumonia AUROC 0.692 (0.660-0.726) against a baseline of 0.703
(0.672-0.740) -- inside the margin of error, slightly down.

Why it failed is the useful part. The gentler augmentation ended at train AUROC
0.9758 / val 0.6568, a *wider* gap than the baseline's 0.9325 / 0.6518. The
upstream 45-degree rotation and mirror flip were not teaching anatomy; they were
regularising a 27.6M-parameter model against 4,956 training images. Removing
them removed the brake. The bottleneck is the training-set size, not the
augmentation -- so the remaining runs attack overfitting directly and keep the
paper's augmentation. Do not apply this patch in later runs.

----------------------------------------------------------------------------
PATCH 2 -- the pneumonia-48h task (Phase 2)
----------------------------------------------------------------------------

Teaches DataFusion the new task name. The window logic is mortality's, already
in the file; the labels come from `--labels_set pheno`, which is independent of
the task name, so nothing else has to change.

Build the data folder first, once, with scripts/make_pneumonia48h.py.

    from pathlib import Path

    f = Path("/content/PneumoVision/medpatch/datasets/DataFusion.py")
    s = f.read_text()
    n = 0

    # Every place the 48-hour window is chosen by task name.
    for old, new in [
        ("if args.task == 'in-hospital-mortality':",
         "if args.task in ('in-hospital-mortality', 'pneumonia-48h'):"),
        ('if args.task == "in-hospital-mortality":',
         'if args.task in ("in-hospital-mortality", "pneumonia-48h"):'),
    ]:
        n += s.count(old)
        s = s.replace(old, new)

    f.write_text(s)
    print(f"DataFusion: {n} task checks widened to include pneumonia-48h")
    assert n >= 3, "expected at least 3 -- the file changed upstream, re-check"

Then train with:

    --task pneumonia-48h \\
    --labels_set pheno \\
    --num_classes 25 \\
    --ehr_data_dir /content/ehr \\
    --normalizer_state normalizers/ihm_ts1.0.input_str_previous.start_time_zero.normalizer

and report label column 21 (pneumonia) of the 25.

Note on the normalizer: fusion_main.py defaults to the *phenotyping* file for
every task (fusion_main.py:112). pneumonia-48h uses 48-hour windows, so the
`ihm_` file is the matching one. It does not affect a `unimodal_cxr` run, whose
predictions never touch the EHR tensor, but it matters as soon as the branch is
fused -- so pass it from the start and the fused run inherits a correct setting.

----------------------------------------------------------------------------
PATCH 3 -- split learning rate: backbone 10x slower than the head
----------------------------------------------------------------------------

`unimodal_cxr` is routed to MSMA_Trainer (fusion_main.py:175, the else branch),
which puts every parameter in one Adam at `args.lr` (MSMA_trainer.py:67). So the
ImageNet-pretrained ViT and the randomly initialised MLP head move at the same
speed: the head needs that speed, the backbone is destroyed by it and memorises
instead. This is the standard fine-tuning fix and the direct answer to the
overfitting PATCH 1 exposed.

Apply this one WITHOUT patch 1 -- keep the paper's augmentation, so the only
change from baseline is the learning rate.

The split is by parameter name, not by module path, so it survives a wrapper
change: only `cxr_classifier` contains "classifier" (checked -- the CXR encoder
has `heads`/`feats_dim`, no `classifier`).

    from pathlib import Path

    f = Path("/content/PneumoVision/medpatch/trainers/MSMA_trainer.py")
    s = f.read_text()
    before = s

    old = '''        else:
            self.optimizer = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))'''

    new = '''        else:
            _head, _back = [], []
            for _n, _p in self.model.named_parameters():
                (_head if 'classifier' in _n else _back).append(_p)
            assert _head and _back, "lr-split: an empty group -- parameter names changed"
            print(f"[lr-split] backbone {len(_back)} tensors @ {args.lr * 0.1:.2e}"
                  f"  |  head {len(_head)} tensors @ {args.lr:.2e}")
            self.optimizer = optim.Adam(
                [{'params': _back, 'lr': args.lr * 0.1},
                 {'params': _head, 'lr': args.lr}],
                args.lr, betas=(0.9, self.args.beta_1))'''

    assert s.count(old) == 1, f"expected 1 match, found {s.count(old)}"
    s = s.replace(old, new)
    f.write_text(s)
    print("lr split:", "patched" if s != before else "NO CHANGE")

ReduceLROnPlateau scales every group by the same factor, so the 10x ratio holds
for the whole run. Verify it took effect from the `[lr-split]` line and from the
per-step `lr:` column, which prints the FIRST group -- it will read 3.0000E-06,
not 3.0000E-05. That is the backbone; the head is still at 3e-5.

Train into a fresh --save_dir (e.g. .../pneumonia48h-lrsplit) so the baseline
and the PATCH 1 run are both kept.

----------------------------------------------------------------------------
PATCH 4 -- fine-tune RAD-DINO as the CXR encoder
----------------------------------------------------------------------------

The frozen probe (scripts/frozen_probe.py) established that a chest-radiograph
encoder beats the paper's ImageNet ViT by +0.050 AUROC on pneumonia-48h
(0.683 -> 0.733; paired bootstrap CI +0.017..+0.081, p = 0.004). Both numbers
there are frozen, so neither is comparable to the fine-tuned 0.703 baseline.
This patch makes RAD-DINO trainable inside the normal loop so one directly
comparable number can be produced.

`CXR_encoder` only accepts timm token models, so --cxr_encoder rad-dino falls
through to the torchvision densenet branch and silently trains the wrong model
(the same shape as defect 6). This adds an explicit branch.

What UnimodalCXR actually requires is small: `.feats_dim`, and a forward
returning either [B, N, D] tokens or [B, D]. The 578-token constants in
fusion.py belong to the confidence and fusion heads, not to unimodal_cxr, so
RAD-DINO's 730 tokens (384/14 = 27, 27^2 + 1) are not a problem here. They
WOULD be for c-msma -- do not reuse this patch for a fusion run without
widening those tensors first.

Most of the backbone stays frozen. 86.6M trainable parameters against 4,956
training images is the overfitting problem of PATCH 1 and 3 again, only four
times larger; unfreezing the last few blocks is the standard answer and it
also keeps the run inside one Colab session. Set RADDINO_UNFREEZE to change
how many (0 = freeze everything, a fine-tuned equivalent of the probe).

    import os
    from pathlib import Path

    os.environ["RADDINO_UNFREEZE"] = "4"      # last 4 of 12 blocks

    f = Path("/content/PneumoVision/medpatch/models/cxr_models.py")
    s = f.read_text(); before = s

    old = '''    import timm
    from .cxr_encoder import CXRTransformer

    name = getattr(args, 'cxr_encoder', None)'''

    new = '''    import timm
    from .cxr_encoder import CXRTransformer

    name = getattr(args, 'cxr_encoder', None)

    if name in ('rad-dino', 'rad_dino', 'microsoft/rad-dino'):
        import os as _os
        import torch.nn as _nn
        from transformers import AutoModel as _AutoModel

        class _RadDino(_nn.Module):
            def __init__(self, unfreeze_last):
                super().__init__()
                self.backbone = _AutoModel.from_pretrained('microsoft/rad-dino')
                self.feats_dim = self.backbone.config.hidden_size
                self.full_feats_dim = self.feats_dim
                blocks = self.backbone.encoder.layer
                for p in self.backbone.parameters():
                    p.requires_grad = False
                for blk in (blocks[-unfreeze_last:] if unfreeze_last else []):
                    for p in blk.parameters():
                        p.requires_grad = True
                for p in self.backbone.layernorm.parameters():
                    p.requires_grad = True
                self.n_blocks = len(blocks)
                self.unfrozen = unfreeze_last

            def forward(self, img):
                v = self.backbone(pixel_values=img).last_hidden_state
                return v, v[:, 0]

        _n = int(_os.environ.get('RADDINO_UNFREEZE', '4'))
        model = _RadDino(_n).to(device)
        _tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _all = sum(p.numel() for p in model.parameters())
        return _describe(
            model, 'microsoft/rad-dino',
            extra=(f"trainable      : {_tr/1e6:.1f}M of {_all/1e6:.1f}M "
                   f"(last {_n} of {model.n_blocks} blocks + final norm)\\n"
                   f"  tokens/image   : 730 (384/14 = 27, 27^2 + 1 CLS)")
        )'''

    assert s.count(old) == 1, f"expected 1 match, found {s.count(old)}"
    s = s.replace(old, new)
    f.write_text(s)
    print("rad-dino encoder:", "patched" if s != before else "NO CHANGE")

Apply PATCH 3 as well: an 86.6M backbone at the head's learning rate is the
failure PATCH 3 exists to prevent, and the split is by parameter name so it
covers this encoder unchanged. Frozen tensors land in the backbone group and
are skipped by Adam, which never sees a gradient for them.

Then train with --cxr_encoder rad-dino and everything else as the baseline:

    --cxr_encoder rad-dino \\
    --use_cls_token cls \\
    --task pneumonia-48h --labels_set pheno --num_classes 25 \\
    --fusion_type unimodal_cxr --modalities EHR-CXR \\
    --normalizer_state normalizers/ihm_ts1.0.input_str_previous.start_time_zero.normalizer

Keep `cls`, not `mean`. The probe preferred `mean` for frozen RAD-DINO
(+0.018), but changing the pooling and the encoder at once breaks the
one-variable comparison against the 0.703 baseline. Test `mean` afterwards as
its own run.

Check the `[cxr]` banner before letting it train: it must name
microsoft/rad-dino and report ~28M trainable. A densenet line there means the
patch did not apply and the run is meaningless.

Expect a slower epoch than the ViT run -- the encoder is four times the size
and carries 730 tokens instead of 577. If CUDA runs out of memory, drop
--batch_size to 8 and record that the baseline used 16.
"""
