"""Fast path, step 1: run the frozen text encoder ONCE and cache one 768-d vector per stay.

Why this is equivalent to the repo (models/fusion.py UnimodalRR / UnimodalDN + models/text_models.py):
  repo:  chunks(512, stride 0, padding=max_length) -> BERT -> mean over chunks -> fc (Linear 768->512)
         -> mean over the 512 token positions -> MLPClassifier
  here:  chunks -> BERT -> mean over chunks -> mean over the 512 positions  (cached)
         and the head applies fc -> MLPClassifier.
  fc is affine, so mean_t(fc(x_t)) == fc(mean_t(x_t)) exactly (up to float rounding).
  Texts and labels come from the repo's own loadmetadata() + listfile merge (DataFusion.py), unchanged.
Known difference: the repo leaves BERT dropout ON during training epochs (model.train()); the cache is computed
with BERT in eval mode, which is exactly what the repo uses for validation/test.

Output (derived patient data -> must stay under data/, which is gitignored):
  <out>/<split>_X.npy (N x 768 float32), <split>_y.npy (N x 25), <split>_stay_id.npy, meta.json
Prints only counts and timings.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True, help='path to medpatch/')
ap.add_argument('--modality', choices=['RR', 'DN'], required=True)
ap.add_argument('--bert_model_name', default='dmis-lab/biobert-v1.1')
ap.add_argument('--ehr_data_dir', required=True)
ap.add_argument('--notes_data_dir', required=True)
ap.add_argument('--notes_file', default=None, help='override the notes csv name (e.g. a cleaned/cut variant)')
ap.add_argument('--out', required=True)
ap.add_argument('--chunk_batch', type=int, default=32, help='BERT chunks per forward pass (lower if GPU OOM)')
ap.add_argument('--limit', type=int, default=0, help='only the first N stays per split (timing test)')
ap.add_argument('--task', default='phenotyping', choices=['phenotyping', 'in-hospital-mortality'],
                help='in-hospital-mortality keeps only reports in [ICU intime, intime+48h] (repo loadmetadata)')
ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], help='which splits to (re)compute')
cli = ap.parse_args()
PART = 2048  # stays per saved part file (resume unit after a disconnect)

sys.path.insert(0, os.path.abspath(cli.repo))
from datasets.DataFusion import loadmetadata, CLASSES  # noqa: E402
from transformers import BertModel, BertTokenizerFast  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, torch.cuda.get_device_name(0) if device.type == 'cuda' else '')
print('TEXT ENCODER MODEL:', cli.bert_model_name)

# --- same data path as fusion_main.py with --modalities EHR-<M> --data_pairs paired --task <task>
assert not (cli.task == 'in-hospital-mortality' and cli.modality == 'DN'), 'DN is not used for mortality (outcome leakage)'
args = argparse.Namespace(modalities=f'EHR-{cli.modality}', data_pairs='paired', task=cli.task,
                          cxr_data_dir='NONE', ehr_data_dir=cli.ehr_data_dir,
                          labels_set='pheno' if cli.task == 'phenotyping' else 'mortality')
LABELS = CLASSES if cli.task == 'phenotyping' else ['y_true']
print('TASK:', cli.task)
cols = ['subject_id', 'hadm_id', 'charttime', 'text']
empty = pd.DataFrame(columns=cols)
fname = cli.notes_file or ('discharge.csv' if cli.modality == 'DN' else 'radiology.csv')
notes = pd.read_csv(os.path.join(cli.notes_data_dir, fname), usecols=cols)
print(f'notes file {fname}: {len(notes)} rows')
dn, rr = (notes, empty) if cli.modality == 'DN' else (empty, notes)
meta = loadmetadata(args, dn, rr, None)
text_col = 'discharge_text' if cli.modality == 'DN' else 'radiology_text'

tok = BertTokenizerFast.from_pretrained(cli.bert_model_name)
bert = BertModel.from_pretrained(cli.bert_model_name).to(device).eval()
hidden = bert.config.hidden_size


@torch.no_grad()
def encode(texts):
    """Same tokenisation as Text_encoder.process_notes; returns per-note mean over chunks and 512 positions."""
    enc = tok(texts, padding='max_length', truncation=True, max_length=512, stride=0,
              return_overflowing_tokens=True, return_tensors='pt')
    mapping = enc['overflow_to_sample_mapping'].numpy()
    chunk_vecs = []
    for s in range(0, len(mapping), cli.chunk_batch):
        ids = enc['input_ids'][s:s + cli.chunk_batch].to(device)
        mask = enc['attention_mask'][s:s + cli.chunk_batch].to(device)
        h = bert(input_ids=ids, attention_mask=mask).last_hidden_state  # [c, 512, 768]
        chunk_vecs.append(h.mean(dim=1).float().cpu())                    # mean over all 512 positions
    chunk_vecs = torch.cat(chunk_vecs).numpy()
    out = np.zeros((len(texts), hidden), dtype=np.float32)
    for i in range(len(texts)):
        out[i] = chunk_vecs[mapping == i].mean(axis=0)                    # mean over that note's chunks
    return out, len(mapping)


os.makedirs(cli.out, exist_ok=True)
info = {'model': cli.bert_model_name, 'modality': cli.modality, 'notes_file': fname, 'task': cli.task, 'splits': {}}
if os.path.exists(f'{cli.out}/meta.json'):                      # keep info about splits done earlier
    old = json.load(open(f'{cli.out}/meta.json'))
    assert old['model'] == cli.bert_model_name and old['notes_file'] == fname and old.get('task', 'phenotyping') == cli.task, \
        'cache folder holds a different model/notes/task'
    info['splits'].update(old.get('splits', {}))
for split in cli.splits:
    lf = pd.read_csv(f'{cli.ehr_data_dir}/{cli.task}/{split}_listfile.csv')
    m = meta.merge(lf, how='inner', on='stay_id')      # identical to load_cxr_ehr_rr_dn
    if cli.limit:
        m = m.head(cli.limit)
    texts = ['' if pd.isna(t) else str(t) for t in m[text_col].values]  # identical to my_collate
    n_chunks, t0, new = 0, time.time(), 0
    step = 64
    # work in parts of PART stays; a finished part is saved, so a rerun after a disconnect skips it
    parts = [f'{cli.out}/{split}_part{p // PART:03d}_of{len(texts)}.npy' for p in range(0, len(texts), PART)]
    for k, part in enumerate(parts):
        if os.path.exists(part):
            continue
        lo, hi = k * PART, min((k + 1) * PART, len(texts))
        Xp = np.zeros((hi - lo, hidden), dtype=np.float32)
        for s in range(lo, hi, step):
            Xp[s - lo:s - lo + step], c = encode(texts[s:min(s + step, hi)])
            n_chunks += c
            new += len(texts[s:min(s + step, hi)])
            if (s // step) % 50 == 0:
                el = time.time() - t0
                print(f'  {split}: {min(s + step, hi)}/{len(texts)} stays, {n_chunks} chunks, {el / 60:.1f} min, '
                      f'est. left {el / new * (len(texts) - min(s + step, hi)) / 60:.1f} min', flush=True)
        np.save(part + '.tmp.npy', Xp)
        os.replace(part + '.tmp.npy', part)  # the part file only appears once it is complete
    X = np.concatenate([np.load(p) for p in parts]) if parts else np.zeros((0, hidden), dtype=np.float32)
    assert len(X) == len(texts)
    np.save(f'{cli.out}/{split}_X.npy', X)
    for p in parts:
        os.remove(p)
    np.save(f'{cli.out}/{split}_y.npy', m[LABELS].values.astype(np.float32))
    np.save(f'{cli.out}/{split}_stay_id.npy', m['stay_id'].values)
    info['splits'][split] = {'stays': len(texts), 'chunks': int(n_chunks), 'minutes': round((time.time() - t0) / 60, 2)}
    print(f'{split}: {len(texts)} stays ({new} encoded this run), {n_chunks} chunks, {(time.time() - t0) / 60:.1f} min')
    json.dump(info, open(f'{cli.out}/meta.json', 'w'), indent=1)  # saved after every split
json.dump(info, open(f'{cli.out}/meta.json', 'w'), indent=1)
print('cache written to', os.path.abspath(cli.out))
