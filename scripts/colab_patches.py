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
"""
