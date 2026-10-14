"""Fast path, step 2: train the RR/DN head on cached embeddings, with the paper's LR search.

Mirrors the repo's unimodal_rr / unimodal_dn training (trainers/MSMA_trainer.py):
  model      fc = Linear(768 -> 512)  then  Classifier('mlp') = LayerNorm(512) + Linear(512 -> 25)
  loss       BCEWithLogits;  Adam(lr, betas=(0.9, 0.9))  [repo passes beta_1=0.9 as beta2]
  scheduler  ReduceLROnPlateau(factor=0.5, patience=10, mode='min') stepped on validation loss
  loop       up to 100 epochs; validate BEFORE each train epoch; keep best val mean AUROC;
             early stop after 15 epochs without improvement; batch 16, shuffle, drop_last
Paper (§4.4): 10 sweeps, LR sampled in [1e-5, 1e-3], select on validation AUROC.
Unspecified in the paper: the sampling distribution -> log-uniform here (stated in the output).
The test set is evaluated ONCE, for the selected LR only.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True)
ap.add_argument('--cache', required=True, help='folder written by cache_embeddings.py')
ap.add_argument('--out', required=True, help='run folder (config, logs, metrics)')
ap.add_argument('--lrs', type=float, nargs='*', default=None, help='explicit LRs (otherwise sampled)')
ap.add_argument('--n_sweeps', type=int, default=10)
ap.add_argument('--sweep_seed', type=int, default=0, help='seed for sampling the LRs')
ap.add_argument('--seed', type=int, default=1002, help='training seed (init + shuffling)')
ap.add_argument('--epochs', type=int, default=100)
ap.add_argument('--patience', type=int, default=15)
ap.add_argument('--batch_size', type=int, default=16)
cli = ap.parse_args()

sys.path.insert(0, os.path.abspath(cli.repo))
from trainers.utils import get_model_performance  # noqa: E402  (repo's bootstrap CI)
import pandas as pd  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(cli.out, exist_ok=True)
log = open(os.path.join(cli.out, 'train.log'), 'a', encoding='utf-8')


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + '\n'); log.flush()


def load(split):
    return (torch.tensor(np.load(f'{cli.cache}/{split}_X.npy')),
            torch.tensor(np.load(f'{cli.cache}/{split}_y.npy')))


Xtr, ytr = load('train'); Xva, yva = load('val'); Xte, yte = load('test')
meta = json.load(open(f'{cli.cache}/meta.json'))
say(f'== {datetime.datetime.now():%Y-%m-%d %H:%M}  cache model={meta["model"]} modality={meta["modality"]} '
    f'notes={meta.get("notes_file")} task={meta.get("task", "phenotyping")}  train/val/test = {len(Xtr)}/{len(Xva)}/{len(Xte)}  device={device}')


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 512)            # Text_encoder.fc_rr / fc_dn
        self.layernorm = nn.LayerNorm(512)       # MLPClassifier
        self.out = nn.Linear(512, ytr.shape[1])

    def forward(self, x):
        return self.out(self.layernorm(self.fc(x)))


def mean_auc(y, p):
    return float(np.mean([roc_auc_score(y[:, i], p[:, i]) for i in range(y.shape[1])]))


@torch.no_grad()
def predict(model, X):
    model.eval()
    return torch.cat([model(X[s:s + 1024].to(device)).cpu() for s in range(0, len(X), 1024)])


def run_one(lr):
    torch.manual_seed(cli.seed); np.random.seed(cli.seed)
    model = Head().to(device)
    opt = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.9))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, mode='min')
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(cli.seed)
    best, best_state, best_epoch, patience, hist = -1, None, -1, 0, []
    for epoch in range(cli.epochs):
        logits = predict(model, Xva)                                     # validate first (as repo)
        vloss = lossf(logits, yva).item()
        sched.step(vloss)
        auc = mean_auc(yva.numpy(), torch.sigmoid(logits).numpy())
        hist.append(round(auc, 4))
        if auc > best:
            best, best_epoch, patience = auc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        say(f'    lr={lr:.2e} epoch {epoch:3d}  val AUROC {auc:.4f}  val loss {vloss:.4f}'
            f'{"  *best*" if patience == 0 else ""}')
        model.train()                                                    # train epoch
        perm = torch.randperm(len(Xtr), generator=g)
        for s in range(0, len(perm) - cli.batch_size + 1, cli.batch_size):  # drop_last
            idx = perm[s:s + cli.batch_size]
            loss = lossf(model(Xtr[idx].to(device)), ytr[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        if patience >= cli.patience:
            break
    return best, best_epoch, best_state, hist


if cli.lrs:
    lrs = cli.lrs
    dist = 'explicit'
else:
    rng = np.random.default_rng(cli.sweep_seed)
    lrs = list(10 ** rng.uniform(-5, -3, cli.n_sweeps))
    dist = f'log-uniform in [1e-5, 1e-3], sweep_seed={cli.sweep_seed}'
try:
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                            cwd=cli.repo).stdout.strip()
except Exception:
    commit = 'unknown'
json.dump({'command': ' '.join(sys.argv), 'git_commit': commit, 'cache_meta': meta, 'lrs': lrs,
           'lr_distribution': dist, 'seed': cli.seed, 'epochs': cli.epochs, 'patience': cli.patience,
           'batch_size': cli.batch_size}, open(os.path.join(cli.out, 'config.json'), 'w'), indent=1)
say(f'LR sweep ({dist}): ' + ', '.join(f'{lr:.2e}' for lr in lrs))

results = []
for k, lr in enumerate(lrs):
    t0 = time.time()
    best, ep, state, hist = run_one(lr)
    results.append({'lr': lr, 'best_val_auroc': round(best, 4), 'best_epoch': ep, 'epochs_run': len(hist),
                    'val_auroc_by_epoch': hist, 'state': state})
    say(f'sweep {k + 1}/{len(lrs)}  lr={lr:.2e}  best val AUROC={best:.4f} at epoch {ep}  '
        f'({len(hist)} epochs, {time.time() - t0:.0f}s)')

chosen = max(results, key=lambda r: r['best_val_auroc'])             # selection on VALIDATION only
model = Head().to(device); model.load_state_dict(chosen['state'])
torch.save({'state_dict': chosen['state'], 'lr': chosen['lr'], 'cache_meta': meta},
           os.path.join(cli.out, 'best_head.pth'))

p = torch.sigmoid(predict(model, Xte)).numpy(); y = yte.numpy()       # TEST: once, chosen LR only
aucs, auprcs, rows = [], [], []
names = ['label_%d' % i for i in range(y.shape[1])]
if y.shape[1] == 1:
    names = ['in-hospital mortality']
else:
    from datasets.DataFusion import CLASSES
    names = CLASSES
for i in range(y.shape[1]):
    (ap_, ap_hi, ap_lo), (au, au_hi, au_lo) = get_model_performance(pd.DataFrame({'y_truth': y[:, i], 'y_pred': p[:, i]}))
    aucs.append(au); auprcs.append(ap_)
    rows.append({'label': names[i], 'auroc': round(au, 4), 'auroc_ci': [round(au_lo, 4), round(au_hi, 4)],
                 'auprc': round(ap_, 4), 'auprc_ci': [round(ap_lo, 4), round(ap_hi, 4)]})
summary = {'chosen_lr': chosen['lr'], 'chosen_val_auroc': chosen['best_val_auroc'],
           'test_mean_auroc': round(float(np.mean(aucs)), 4), 'test_mean_auprc': round(float(np.mean(auprcs)), 4),
           'per_label': rows,
           'sweeps': [{k: v for k, v in r.items() if k != 'state'} for r in results]}
json.dump(summary, open(os.path.join(cli.out, 'metrics.json'), 'w'), indent=1)
say(f'CHOSEN lr={chosen["lr"]:.2e} (val {chosen["best_val_auroc"]:.4f})  ->  '
    f'TEST mean AUROC {summary["test_mean_auroc"]:.4f}  mean AUPRC {summary["test_mean_auprc"]:.4f}')
pn = [r for r in rows if r['label'].startswith('Pneumonia')]
if pn:
    say(f'  pneumonia: AUROC {pn[0]["auroc"]} {pn[0]["auroc_ci"]}  AUPRC {pn[0]["auprc"]}')
say('metrics written to', os.path.abspath(os.path.join(cli.out, 'metrics.json')))
