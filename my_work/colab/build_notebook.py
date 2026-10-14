"""Generates PneumoVision_text_training.ipynb (run locally: python build_notebook.py)."""
import json
import os

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# PneumoVision: text training (RR + DN), Task B phenotyping

**Before starting:** Runtime → Change runtime type → **T4 GPU**.

Run the cells **top to bottom**. If Colab disconnects: reconnect and run all cells again.
Training continues from the last finished epoch (it does not start over).

Patient data rule: this notebook prints only counts, shapes and scores, never note text or patient rows.
""")

md("## 1. Settings (the only cell you may need to edit)")
code(r"""
# Where Caroll's shared folder appears after you add the shortcut to My Drive
SHARED = '/content/drive/MyDrive/PneumoVision_shared/data'
# Folder that holds the code zip (a shortcut to another account's folder is fine: it is only read)
MINE = '/content/drive/MyDrive/PneumoVision_farida'
# Where checkpoints, logs and the BioBERT cache are SAVED. A shortcut to another account's folder works:
# new files are OWNED by the account you mounted and use its storage, wherever the folder lives.
OUT = MINE

SEED = 1002          # keep 1002 for the first run (the repo's default)
LR = 0.001           # the repo scripts' value
# Text model. Paper says BioBERT. (Repo code default was emilyalsentzer/Bio_ClinicalBERT.)
BERT = 'dmis-lab/biobert-v1.1'
""")

md("## 2. Check the GPU and connect Google Drive")
code(r"""
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')
""")

md("## 3. Check every needed file is on Drive")
code(r"""
import os, glob
need = {
    'code zip':        f'{MINE}/medpatch_code.zip',
    'all_stays':       f'{SHARED}/ehr/root/all_stays.csv',
    'train listfile':  f'{SHARED}/ehr/phenotyping/train_listfile.csv',
    'val listfile':    f'{SHARED}/ehr/phenotyping/val_listfile.csv',
    'test listfile':   f'{SHARED}/ehr/phenotyping/test_listfile.csv',
    'timeseries rar':  f'{SHARED}/ehr/phenotyping/phenotype_labels.rar',
}
def first_existing(*names):
    hits = [f'{SHARED}/notes/{n}' for n in names if os.path.exists(f'{SHARED}/notes/{n}')]
    return hits[0] if hits else None
RAD = first_existing('radiology.zip', 'radiology.csv', 'radiology.csv.gz')
DIS = first_existing('discharge.zip', 'discharge.csv', 'discharge.csv.gz')
ok = True
for name, p in need.items():
    exists = os.path.exists(p)
    ok &= exists
    print(f"OK       {name:15s} {p}  {os.path.getsize(p)/1e6:,.0f} MB" if exists else f"MISSING  {name:15s} {p}")
print(f"{'OK     ' if RAD else 'MISSING'}  radiology       {RAD}")
print(f"{'OK     ' if DIS else 'MISSING'}  discharge       {DIS}")
ok &= RAD is not None
assert ok, 'Fix the MISSING files above before continuing (RR needs everything except discharge).'
""")

md("## 4. Install packages and unpack the code (about 2 minutes)")
code(r"""
# 4.45.2, not the repo's 4.44.2: Colab is on Python 3.13, which has no prebuilt tokenizers 0.19 wheel
!pip -q install transformers==4.45.2 timm==1.0.9 einops wandb
!rm -rf /content/medpatch && mkdir -p /content/medpatch
!unzip -q -o "{MINE}/medpatch_code.zip" -d /content/medpatch
!md5sum "{MINE}/medpatch_code.zip" | tee /content/code_md5.txt
import os
os.environ['WANDB_MODE'] = 'offline'                 # no wandb login needed
os.environ['PYTHONUNBUFFERED'] = '1'                 # show progress lines immediately
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'  # repo checkpoints are saved with numpy values
os.environ['TOKENIZERS_PARALLELISM'] = 'false'       # silences the "process just got forked" warning
import torch; print('torch', torch.__version__, 'GPU:', torch.cuda.is_available())
""")

md("""
## 5. Prepare the data on Colab's local disk (about 5–15 minutes)
Unpacks the timeseries, removes the extra MIMIC-IV v3 patients (expected: 42,328 / 4,756 / 11,845 rows),
and copies the notes locally (much faster than reading from Drive). Drive files are not changed.
""")
code(r"""
import os, glob
!apt-get -qq install -y unrar > /dev/null
if not os.path.exists('/content/ts_done'):
    !mkdir -p /content/ts && unrar x -o+ -inul "{SHARED}/ehr/phenotyping/phenotype_labels.rar" /content/ts/ && touch /content/ts_done
# find the folder that holds train/ and test/ (works whatever the rar's inner folder name is)
hits = glob.glob('/content/ts/**/train/14991576_episode3_timeseries.csv', recursive=True)
assert hits, 'Could not find the train/ timeseries folder inside the rar'
TS = os.path.dirname(os.path.dirname(hits[0]))
print('timeseries folder:', TS, '| train files:', len(os.listdir(f'{TS}/train')), '| test files:', len(os.listdir(f'{TS}/test')))

common = (f'--all_stays "{SHARED}/ehr/root/all_stays.csv" --listfile_dir "{SHARED}/ehr/phenotyping" '
          f'--timeseries_dir "{TS}" --repo /content/medpatch')
!rm -rf /content/ehr /content/ehr_small
!python /content/medpatch/prepare_data.py {common} --out /content/ehr
!python /content/medpatch/prepare_data.py {common} --out /content/ehr_small --small 64 > /dev/null

!mkdir -p /content/notes

def unpack_note(src, name):
    # puts the file at /content/notes/<name>.csv whatever the upload format (.zip / .csv.gz / .csv)
    dst = f'/content/notes/{name}.csv'
    if src is None or os.path.exists(dst):
        return
    if src.endswith('.zip'):
        !rm -rf /content/tmp_unzip && mkdir -p /content/tmp_unzip && unzip -q -o "{src}" -d /content/tmp_unzip
        csvs = [p for p in glob.glob('/content/tmp_unzip/**/*.csv', recursive=True) if '__MACOSX' not in p]
        assert len(csvs) == 1, f'expected exactly one csv inside {src}, found {len(csvs)}'
        os.rename(csvs[0], dst)
    elif src.endswith('.gz'):
        !gunzip -c "{src}" > "{dst}"
    else:
        !cp "{src}" "{dst}"

unpack_note(RAD, 'radiology')
unpack_note(DIS, 'discharge')
# must be the ORIGINAL PhysioNet files (column names only are printed)
import pandas as pd
for n in ['radiology', 'discharge']:
    f = f'/content/notes/{n}.csv'
    if os.path.exists(f):
        cols = list(pd.read_csv(f, nrows=0).columns)
        good = {'subject_id', 'hadm_id', 'charttime', 'text'} <= set(cols)
        print(f'{n}.csv columns: {cols} -> {"OK" if good else "WRONG FILE (need the original with a text column)"}')
!ls -la /content/notes
""")

md("## 6. Training helper (run this cell, nothing happens yet)")
code(r"""
import datetime, subprocess, os

def train(modality, ehr_dir='/content/ehr', run_name=None, epochs=100, resume=True):
    mod = modality.upper()            # 'RR' or 'DN'
    m = mod.lower()
    run_name = run_name or f'taskB_unimodal_{m}_biobert_seed{SEED}'
    run_dir = f'{OUT}/runs/{run_name}'
    os.makedirs(run_dir, exist_ok=True)
    cmd = (f'python fusion_main.py --mode train --epochs {epochs} --batch_size 16 --lr {LR} '
           f'--num_classes 25 --modalities EHR-{mod} --fusion_type unimodal_{m} --text_encoder biobert '
           f'--classifier mlp --loss bce --task phenotyping --labels_set pheno --output_dim_{m} 512 '
           f'--data_pairs paired --seed {SEED} --num_workers 2 --bert_model_name {BERT} '
           f'--ehr_data_dir {ehr_dir} --notes_data_dir /content/notes --cxr_data_dir NONE '
           f'--save_dir "{run_dir}/checkpoints"' + (' --resume' if resume else ''))
    with open(f'{run_dir}/command.txt', 'a') as f:
        f.write(f'# {datetime.datetime.now():%Y-%m-%d %H:%M}  code md5: {open("/content/code_md5.txt").read().split()[0]}\n{cmd}\n\n')
    print('RUN FOLDER:', run_dir)
    print(cmd)
    # output goes to the screen AND to train.log on Drive (appended, so resumed runs keep history)
    !cd /content/medpatch && {cmd} 2>&1 | grep --line-buffered -v "Warning\|warn(" | tee -a "{run_dir}/train.log"
    print('\nResults files:');
    # the repo does not write a results file; the test summary is in train.log
    lines = open(f'{run_dir}/train.log').readlines()
    tests = [i for i, l in enumerate(lines) if 'test' in l and 'best mean auc' in l]
    if tests:
        print('\nFINAL TEST RESULT (from train.log):')
        print(''.join(lines[tests[-1] - 28: tests[-1] + 8]))  # 25 disease lines + 3 blank lines above
""")

md("""
## 7. FAST path: write the two scripts and the helper (run this, nothing trains yet)
BioBERT is frozen, so its output for a patient never changes. The fast path runs BioBERT **once** per patient,
saves one vector per patient to Drive (`PneumoVision_farida/cache/`), then trains only the small layer on top
(same model, optimizer, scheduler, early stopping as the repo), with the paper's 10-learning-rate search.
Difference vs the repo: BioBERT dropout is off while training (the repo leaves it on). Write that in the Sheet.
""")
code("!mkdir -p /content/fast_text")
for _name in ['cache_embeddings.py', 'train_head.py']:
    _src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fast_text', _name),
                encoding='utf-8').read()
    code(f'%%writefile /content/fast_text/{_name}\n' + _src)
code(r"""
import os, datetime

def fast(modality, limit=0, n_sweeps=10, epochs=100, lrs=None):
    # lrs=[x]: skip the 10-LR search and train only at learning rate x (used for extra seeds)
    mod = modality.upper()            # 'RR' or 'DN'
    m = mod.lower()
    tag = f'_limit{limit}' if limit else ''
    cache = f'{OUT}/cache/taskB_{m}_biobert{tag}'
    lr_tag = '_lr' + '_'.join(f'{x:.2e}' for x in lrs) if lrs else ''
    run_dir = f'{OUT}/runs/fast_taskB_unimodal_{m}_biobert_seed{SEED}{lr_tag}{tag}'
    os.makedirs(cache, exist_ok=True); os.makedirs(run_dir, exist_ok=True)
    md5 = open('/content/code_md5.txt').read().split()[0]
    # step A: BioBERT once per patient (skips splits and parts already saved, so it resumes after a disconnect)
    todo = [s for s in ['train', 'val', 'test'] if not os.path.exists(f'{cache}/{s}_X.npy')]
    if todo:
        cmd = (f'python /content/fast_text/cache_embeddings.py --repo /content/medpatch --modality {mod} '
               f'--bert_model_name {BERT} --ehr_data_dir /content/ehr --notes_data_dir /content/notes '
               f'--out "{cache}" --limit {limit} --splits {" ".join(todo)}')
        with open(f'{cache}/command.txt', 'a') as f:
            f.write(f'# {datetime.datetime.now():%Y-%m-%d %H:%M}  code md5: {md5}\n{cmd}\n\n')
        print('CACHE FOLDER:', cache)
        !cd /content/medpatch && {cmd} 2>&1 | grep --line-buffered -v "Warning\|warn(\|row_indexer\|pandas-docs" | tee -a "{cache}/cache.log"
    missing = [s for s in ['train', 'val', 'test'] if not os.path.exists(f'{cache}/{s}_X.npy')]
    assert not missing, f'caching did not finish for {missing}: run this cell again (it continues where it stopped)'
    # step B: train the small layer on the saved vectors (minutes per learning rate)
    cmd = (f'python /content/fast_text/train_head.py --repo /content/medpatch --cache "{cache}" --out "{run_dir}" '
           f'--seed {SEED} --n_sweeps {n_sweeps} --epochs {epochs}'
           + (' --lrs ' + ' '.join(repr(x) for x in lrs) if lrs else ''))
    with open(f'{run_dir}/command.txt', 'a') as f:
        f.write(f'# {datetime.datetime.now():%Y-%m-%d %H:%M}  code md5: {md5}\n{cmd}\n\n')
    print('RUN FOLDER:', run_dir)
    !cd /content/medpatch && {cmd} 2>&1 | grep --line-buffered -v "epoch .*val AUROC"
    print(f'\nper-epoch details: {run_dir}/train.log   all scores: {run_dir}/metrics.json')
""")

md("""
## 8. FAST sanity test: 512 patients per split, 1 learning rate, 2 epochs (about 5–10 minutes)
Must finish without errors before the full fast runs. Its score means nothing.
""")
code(r"""
fast('RR', limit=512, n_sweeps=1, epochs=2)
""")

md("""
## 9. FAST full run: Radiology Reports (target AUROC 0.776, AUPRC 0.474)
Step A (BioBERT once) takes a few hours on a T4; step B takes about 1 hour for the 10 learning rates.
If Colab disconnects: reconnect, run cells 1–7 again, then this cell. Step A continues from the last saved part.
""")
code(r"""
fast('RR')
""")

md("""
## 10. FAST full run: Discharge Notes (target AUROC 0.856, AUPRC 0.601)
Discharge notes are longer than radiology reports, so step A takes longer. Same disconnect rule.
""")
code(r"""
fast('DN')
""")

md("""
## 11. Extra seeds for DN (about 7 minutes per seed; needs step 10's cache)
Re-trains the small layer at the learning rate step 10 chose, with other seeds, then prints mean ± std
of the test scores (seed 1002 included). Step A is not repeated.
""")
code(r"""
import json, numpy as np
DN_LR = 0.00012225031390095305       # the CHOSEN lr from step 10 (exact value, from its config.json 'lrs')
EXTRA_SEEDS = [1, 2, 3, 4]           # plus 1002 from step 10 = 5 seeds
for s in EXTRA_SEEDS:
    SEED = s
    fast('DN', lrs=[DN_LR])
SEED = 1002
runs = [f'{OUT}/runs/fast_taskB_unimodal_dn_biobert_seed1002'] + \
       [f'{OUT}/runs/fast_taskB_unimodal_dn_biobert_seed{s}_lr{DN_LR:.2e}' for s in EXTRA_SEEDS]
res = [json.load(open(f'{r}/metrics.json')) for r in runs]
au = np.array([r['test_mean_auroc'] for r in res]); ap = np.array([r['test_mean_auprc'] for r in res])
print('per seed AUROC:', au, ' AUPRC:', ap)
print(f'DN 5 seeds: test AUROC {au.mean():.4f} ± {au.std(ddof=1):.4f}   AUPRC {ap.mean():.4f} ± {ap.std(ddof=1):.4f}')
""")

md("""
## After each full run
Log in the team Sheet: Who=Farida · Date · What (unimodal RR / DN, **fast path**) · Task B phenotyping ·
File path (`PneumoVision_farida/runs/fast_taskB_unimodal_<rr|dn>_biobert_seed1002/best_head.pth`) ·
Score (test mean AUROC and AUPRC: last lines of the output, or `metrics.json`) ·
Notes (seed, chosen lr, "fast path: cached frozen BioBERT, dropout off", "restricted to repo split subjects").

---
## Optional: the original (slow) repo pipeline
About 2–2.5 hours per epoch on a T4 because it re-runs BioBERT on every patient every epoch. Not needed if you use
the fast path above.
""")
code(r"""
train('RR', ehr_dir='/content/ehr_small', run_name='SANITY_rr', epochs=1, resume=False)   # slow sanity test
""")
code(r"""
train('RR')
""")
code(r"""
train('DN')
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PneumoVision_text_training.ipynb')
json.dump(nb, open(out, 'w', encoding='utf-8'), indent=1)
print('wrote', out)
