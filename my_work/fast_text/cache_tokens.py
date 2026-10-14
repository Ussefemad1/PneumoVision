"""Round 2 fast path, step 1: cache the token-level text features the confidence predictor sees.

Repo (models/text_models.py process_notes + fusion.UnimodalRRConfidence):
  chunks(512, stride 0, padding=max_length) -> frozen BERT -> mean over a note's chunks -> full_rr_feats [512, 768]
  -> rr_confidence_predictor = Linear(768 -> C) on every one of the 512 positions.
Here the [512, 768] matrix per stay is computed once (BERT in eval mode, as in repo validation) and stored as float16
in a memory-mapped file. Texts/labels come from the repo's loadmetadata() + listfile merge (same as cache_embeddings.py).

Output (derived patient data: keep OUTSIDE the repo, e.g. D:\\PneumoVision_cache\\...):
  <out>/<split>_tokens.f16 (N x 512 x 768 memmap), <split>_y.npy, <split>_stay_id.npy, <split>_done.txt, meta.json
Resumable: <split>_done.txt holds how many stays are finished. Prints only counts and timings.
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
ap.add_argument('--repo', required=True)
ap.add_argument('--modality', choices=['RR', 'DN'], required=True)
ap.add_argument('--task', choices=['phenotyping', 'in-hospital-mortality'], required=True)
ap.add_argument('--bert_model_name', default='dmis-lab/biobert-v1.1')
ap.add_argument('--ehr_data_dir', required=True)
ap.add_argument('--notes_data_dir', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--splits', nargs='+', default=['train', 'val'])
ap.add_argument('--chunk_batch', type=int, default=32)
ap.add_argument('--limit', type=int, default=0)
cli = ap.parse_args()
assert not (cli.task == 'in-hospital-mortality' and cli.modality == 'DN'), 'DN is not used for mortality'

sys.path.insert(0, os.path.abspath(cli.repo))
from datasets.DataFusion import loadmetadata, CLASSES  # noqa: E402
from transformers import BertModel, BertTokenizerFast  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, '| TEXT ENCODER MODEL:', cli.bert_model_name, '| TASK:', cli.task, flush=True)
args = argparse.Namespace(modalities=f'EHR-{cli.modality}', data_pairs='paired', task=cli.task, cxr_data_dir='NONE',
                          ehr_data_dir=cli.ehr_data_dir, labels_set='pheno' if cli.task == 'phenotyping' else 'mortality')
LABELS = CLASSES if cli.task == 'phenotyping' else ['y_true']
cols = ['subject_id', 'hadm_id', 'charttime', 'text']
empty = pd.DataFrame(columns=cols)
fname = 'discharge.csv' if cli.modality == 'DN' else 'radiology.csv'
notes = pd.read_csv(os.path.join(cli.notes_data_dir, fname), usecols=cols)
meta = loadmetadata(args, *((notes, empty) if cli.modality == 'DN' else (empty, notes)), None)
del notes
text_col = 'discharge_text' if cli.modality == 'DN' else 'radiology_text'

tok = BertTokenizerFast.from_pretrained(cli.bert_model_name)
bert = BertModel.from_pretrained(cli.bert_model_name).to(device).eval()


@torch.no_grad()
def encode(texts):
    """Exactly Text_encoder.process_notes up to batch_embeddings: per-note mean over chunks, all 512 positions."""
    enc = tok(texts, padding='max_length', truncation=True, max_length=512, stride=0,
              return_overflowing_tokens=True, return_tensors='pt')
    mapping = enc['overflow_to_sample_mapping'].to(device)
    hs = []
    for s in range(0, len(mapping), cli.chunk_batch):
        hs.append(bert(input_ids=enc['input_ids'][s:s + cli.chunk_batch].to(device),
                       attention_mask=enc['attention_mask'][s:s + cli.chunk_batch].to(device)).last_hidden_state)
    h = torch.cat(hs)                                                     # [chunks, 512, 768] float32
    return torch.stack([h[mapping == i].mean(0) for i in range(len(texts))]).half().cpu().numpy(), len(mapping)


os.makedirs(cli.out, exist_ok=True)
info = {'model': cli.bert_model_name, 'modality': cli.modality, 'task': cli.task, 'dtype': 'float16',
        'shape_per_stay': [512, 768], 'splits': {}}
if os.path.exists(f'{cli.out}/meta.json'):
    old = json.load(open(f'{cli.out}/meta.json'))
    assert (old['model'], old['modality'], old['task']) == (cli.bert_model_name, cli.modality, cli.task)
    info['splits'].update(old['splits'])
for split in cli.splits:
    m = meta.merge(pd.read_csv(f'{cli.ehr_data_dir}/{cli.task}/{split}_listfile.csv'), how='inner', on='stay_id')
    if cli.limit:
        m = m.head(cli.limit)
    texts = ['' if pd.isna(t) else str(t) for t in m[text_col].values]
    N = len(texts)
    np.save(f'{cli.out}/{split}_y.npy', m[LABELS].values.astype(np.float32))
    np.save(f'{cli.out}/{split}_stay_id.npy', m['stay_id'].values)
    path, donef = f'{cli.out}/{split}_tokens.f16', f'{cli.out}/{split}_done.txt'
    X = np.memmap(path, dtype=np.float16, mode='r+' if os.path.exists(path) else 'w+', shape=(N, 512, 768))
    done = int(open(donef).read()) if os.path.exists(donef) else 0
    t0, n_chunks, step = time.time(), 0, 32
    for s in range(done, N, step):
        X[s:s + step], c = encode(texts[s:s + step])
        n_chunks += c
        if (s // step) % 100 == 0 or s + step >= N:
            X.flush()
            open(donef, 'w').write(str(min(s + step, N)))
            el = time.time() - t0
            print(f'  {split}: {min(s + step, N)}/{N} stays, {el / 60:.1f} min, '
                  f'est. left {el / (min(s + step, N) - done) * (N - min(s + step, N)) / 60:.1f} min', flush=True)
    X.flush(); del X
    open(donef, 'w').write(str(N))
    info['splits'][split] = {'stays': N}
    json.dump(info, open(f'{cli.out}/meta.json', 'w'), indent=1)
    print(f'{split}: {N} stays done ({(time.time() - t0) / 60:.1f} min this run)', flush=True)
print('token cache written to', os.path.abspath(cli.out))
