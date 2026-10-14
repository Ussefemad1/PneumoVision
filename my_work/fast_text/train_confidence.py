"""Round 2 (confidence) and Round 2b (temperature calibration) for RR/DN on cached token features, then repo checkpoints.

Round 2 mirrors MSMA_Trainer with --fusion_type c-unimodal_<m>:
  model      rr_confidence_predictor = Linear(768 -> C) on each of the 512 positions (BERT, fc, classifier frozen/unused)
  loss       BCEWithLogits over [B, 512(, C)] with the stay label repeated on every position
  optim      Adam(lr, betas=(0.9, 0.9)); ReduceLROnPlateau(0.5, patience 10) stepped on val loss
  loop       validate first; keep LOWEST val loss; early stop after 15 epochs w/o improvement; up to 100 epochs;
             batch 16, shuffle, drop_last. LR: paper's 10 sweeps in [1e-5, 1e-3] (log-uniform), selected on val loss.
Round 2b mirrors trainers/Calibration.py with --fusion_type temp_c-unimodal_<m> (VALIDATION set only):
  logits / rr_temperature[512] (init 1), BCE, Adam(lr, betas=(0.9, 0.9)), batches of 16 in loader order, N epochs
  (paper: 5). ECE exactly as Calibration.compute_ece (10 bins; for num_classes=1 this reads token position 0 only),
  computed on the probabilities produced during each training pass; keep the temperature with the lowest mean ECE,
  starting from the pre-calibration ECE.
Known difference: the repo runs BERT in train mode (dropout on) during these training passes; the cache is eval-mode.
Then writes repo checkpoints for c-unimodal_<m> and temp_c-unimodal_<m> (exact MSMA/Calibration save paths).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True)
ap.add_argument('--tokens', required=True, help='folder from cache_tokens.py')
ap.add_argument('--r1_checkpoint', required=True, help='repo-format Round 1 checkpoint (to_repo_checkpoint.py)')
ap.add_argument('--save_dir_r2', required=True)
ap.add_argument('--save_dir_r2b', required=True)
ap.add_argument('--out', required=True, help='run folder for logs/metrics')
ap.add_argument('--lrs', type=float, nargs='*', default=None)
ap.add_argument('--n_sweeps', type=int, default=10)
ap.add_argument('--sweep_seed', type=int, default=0)
ap.add_argument('--seed', type=int, default=1002)
ap.add_argument('--epochs', type=int, default=100)
ap.add_argument('--patience', type=int, default=15)
ap.add_argument('--calib_epochs', type=int, default=5)
ap.add_argument('--calib_lr', type=float, default=0.001, help='repo Calibrate-*.sh pass --lr 0.001')
ap.add_argument('--batch_size', type=int, default=16)
cli = ap.parse_args()

sys.path.insert(0, os.path.abspath(cli.repo))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(cli.out, exist_ok=True)
log = open(os.path.join(cli.out, 'train.log'), 'a', encoding='utf-8')


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); log.write(s + '\n'); log.flush()


meta = json.load(open(f'{cli.tokens}/meta.json'))
m, M, task = meta['modality'].lower(), meta['modality'], meta['task']
C = 25 if task == 'phenotyping' else 1


def load(split):
    y = torch.tensor(np.load(f'{cli.tokens}/{split}_y.npy'))
    if C == 1:
        y = y[:, 0]                                    # repo: mortality label is a scalar per stay -> [B]
    X = np.memmap(f'{cli.tokens}/{split}_tokens.f16', dtype=np.float16, mode='r', shape=(len(y), 512, 768))
    assert int(open(f'{cli.tokens}/{split}_done.txt').read()) == len(y), f'{split} cache incomplete'
    return X, y


Xtr, ytr = load('train'); Xva, yva = load('val')
say(f'== {time.strftime("%Y-%m-%d %H:%M")}  {meta["model"]} {M} task={task}  train/val = {len(ytr)}/{len(yva)}  device={device}')


def batch_x(X, idx):
    idx = np.sort(np.asarray(idx))                     # sorted reads from the memmap are faster; order within a batch
    return torch.from_numpy(np.ascontiguousarray(X[idx])).to(device).float(), idx   # does not change the mean loss


def rep(y, L=512):                                     # MSMA: y.unsqueeze(1).repeat(1, pred.shape[1](, 1))
    return y.unsqueeze(1).repeat(1, L) if C == 1 else y.unsqueeze(1).repeat(1, L, 1)


def logits(layer, x):                                  # ConfidencePredictor: Linear then squeeze(-1)
    return layer(x).squeeze(-1)


lossf = nn.BCEWithLogitsLoss()


@torch.no_grad()
def val_loss(layer):
    layer.eval(); tot, n = 0.0, 0
    bs = cli.batch_size
    for s in range(0, len(yva), bs):
        x, _ = batch_x(Xva, range(s, min(s + bs, len(yva))))
        y = yva[s:s + bs].to(device)
        tot += lossf(logits(layer, x), rep(y)).item(); n += 1
    return tot / n


def run_r2(lr):
    torch.manual_seed(cli.seed); np.random.seed(cli.seed)
    layer = nn.Linear(768, C).to(device)
    opt = torch.optim.Adam(layer.parameters(), lr, betas=(0.9, 0.9))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, mode='min')
    g = torch.Generator().manual_seed(cli.seed)
    best, best_state, best_ep, patience, hist = float('inf'), None, -1, 0, []
    for ep in range(cli.epochs):
        t0 = time.time()
        vl = val_loss(layer); sched.step(vl); hist.append(round(vl, 5))
        if vl < best:
            best, best_ep, patience = vl, ep, 0
            best_state = {k: v.detach().clone().cpu() for k, v in layer.state_dict().items()}
        else:
            patience += 1
        layer.train()
        perm = torch.randperm(len(ytr), generator=g).numpy()
        for s in range(0, len(perm) - cli.batch_size + 1, cli.batch_size):
            idx = perm[s:s + cli.batch_size]
            x, sidx = batch_x(Xtr, idx)
            loss = lossf(logits(layer, x), rep(ytr[sidx].to(device)))
            opt.zero_grad(); loss.backward(); opt.step()
        say(f'    r2 lr={lr:.2e} epoch {ep:3d}  val loss {vl:.5f}{"  *best*" if patience == 0 else ""}  ({time.time() - t0:.0f}s)')
        if patience >= cli.patience:
            break
    return best, best_ep, best_state, hist


# ---------------- Round 2: confidence predictor, LR sweep selected on validation loss
if cli.lrs:
    lrs, dist = cli.lrs, 'explicit'
else:
    lrs = list(10 ** np.random.default_rng(cli.sweep_seed).uniform(-5, -3, cli.n_sweeps))
    dist = f'log-uniform in [1e-5, 1e-3], sweep_seed={cli.sweep_seed}'
say(f'R2 LR sweep ({dist}): ' + ', '.join(f'{lr:.2e}' for lr in lrs))
res = []
for k, lr in enumerate(lrs):
    done_path = os.path.join(cli.out, f'r2_sweep_{lr:.8e}_seed{cli.seed}.pth')   # finished sweeps survive a restart
    if os.path.exists(done_path):
        res.append(torch.load(done_path, weights_only=False))
        say(f'r2 sweep {k + 1}/{len(lrs)}  lr={lr:.2e}  (already done) best val loss={res[-1]["best_val_loss"]:.5f}')
        continue
    t0 = time.time()
    b, ep, st, hist = run_r2(lr)
    res.append({'lr': lr, 'best_val_loss': b, 'best_epoch': ep, 'epochs_run': len(hist), 'val_loss_by_epoch': hist, 'state': st})
    torch.save(res[-1], done_path)
    say(f'r2 sweep {k + 1}/{len(lrs)}  lr={lr:.2e}  best val loss={b:.5f} at epoch {ep}  ({len(hist)} epochs, {time.time() - t0:.0f}s)')
r2 = min(res, key=lambda r: r['best_val_loss'])
layer = nn.Linear(768, C).to(device); layer.load_state_dict(r2['state']); layer.eval()
say(f'R2 CHOSEN lr={r2["lr"]:.2e}  val loss {r2["best_val_loss"]:.5f}')


# ---------------- Round 2b: temperature on VALIDATION only (Calibration.py)
def compute_ece(probs, labels, n_bins=10):              # copy of Calibration.compute_ece
    ece = torch.zeros(C)
    bounds = torch.linspace(0, 1, n_bins + 1)
    for c in range(C):
        cp, cl = probs[:, c], labels[:, c]
        pl = (cp >= 0.5).long()
        conf = torch.where(pl == 1, cp, 1.0 - cp)
        e = torch.zeros(1)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            inb = (conf > lo) & (conf <= hi)
            p = inb.float().mean()
            if p.item() > 0:
                e += torch.abs(conf[inb].mean() - (pl[inb] == cl[inb]).float().mean()) * p
        ece[c] = e
    return ece


with torch.no_grad():                                   # confidence logits on val are fixed (everything frozen)
    Lva = torch.cat([logits(layer, batch_x(Xva, range(s, min(s + 256, len(yva))))[0]).cpu()
                     for s in range(0, len(yva), 256)])  # [N, 512(, C)]
temp = nn.Parameter(torch.ones(512) if C == 1 else torch.ones(512, C))
opt = torch.optim.Adam([temp], cli.calib_lr, betas=(0.9, 0.9))


def calib_pass(train):
    probs, labs = [], []
    for s in range(0, len(yva), cli.batch_size):
        lg = Lva[s:s + cli.batch_size]; y = rep(yva[s:s + cli.batch_size])
        scaled = lg / temp.clamp_min(1e-9).unsqueeze(0)
        loss = lossf(scaled, y)
        probs.append(torch.sigmoid(scaled).detach()); labs.append(y)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
    return torch.cat(probs), torch.cat(labs)


p, l = calib_pass(False)
best_ece = compute_ece(p, l).mean().item()
pre_ece, best_temp = best_ece, temp.detach().clone()
say(f'R2b pre-calibration ECE {pre_ece:.5f}')
for ep in range(cli.calib_epochs):
    p, l = calib_pass(True)
    e = compute_ece(p, l).mean().item()
    better = e < best_ece
    if better:
        best_ece, best_temp = e, temp.detach().clone()
    say(f'    r2b epoch {ep}  ECE {e:.5f}{"  *best*" if better else ""}')
say(f'R2b best ECE {best_ece:.5f} (pre {pre_ece:.5f}); temperature mean {best_temp.mean():.4f} min {best_temp.min():.4f} max {best_temp.max():.4f}')

# ---------------- repo checkpoints
from arguments import args_parser  # noqa: E402
from models.fusion import Fusion  # noqa: E402
from models.text_models import Text_encoder  # noqa: E402

r1 = torch.load(cli.r1_checkpoint, map_location='cpu', weights_only=False)
lr_name = r1['lr']
out = {}
for ft, sdir, extra in [(f'c-unimodal_{m}', cli.save_dir_r2, {}),
                        (f'temp_c-unimodal_{m}', cli.save_dir_r2b, {f'fusion_model.{m}_temperature': best_temp})]:
    a = args_parser().parse_args([
        '--fusion_type', ft, '--modalities', f'EHR-{M}', '--task', task, '--num_classes', str(C),
        '--labels_set', 'pheno' if task == 'phenotyping' else 'mortality', '--text_encoder', 'biobert',
        '--classifier', 'mlp', '--loss', 'bce', '--data_pairs', 'paired', '--lr', repr(lr_name),
        '--bert_model_name', meta['model']])
    model = Fusion(a, None, None, Text_encoder(a, torch.device('cpu')))
    sd = model.state_dict()
    n_r1 = 0
    for k, v in r1['state_dict'].items():               # BERT + fc + classifier from Round 1 (as --load_<m> does)
        if k in sd:
            sd[k].copy_(v); n_r1 += 1
    sd[f'fusion_model.{m}_confidence_predictor.confidence_layer.weight'].copy_(r2['state']['weight'])
    sd[f'fusion_model.{m}_confidence_predictor.confidence_layer.bias'].copy_(r2['state']['bias'])
    for k, v in extra.items():
        assert sd[k].shape == v.shape, (k, sd[k].shape, v.shape); sd[k].copy_(v)
    d = os.path.join(sdir, task, ft); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f'best_checkpoint_{a.lr}_{task}_{ft}_EHR-{M}_paired.pth.tar')
    torch.save({'epoch': -1, 'state_dict': sd, 'best_auroc': None, 'optimizer': None,
                'converted_from': 'my_work/fast_text/train_confidence.py', 'bert_model_name': meta['model']}, path)
    say(f'REPO CHECKPOINT {ft}: {os.path.abspath(path)}  ({n_r1} tensors from Round 1)')
    out[ft] = os.path.abspath(path)

json.dump({'r2_lr_distribution': dist, 'r2_chosen_lr': r2['lr'], 'r2_best_val_loss': r2['best_val_loss'],
           'r2_sweeps': [{k: v for k, v in r.items() if k != 'state'} for r in res],
           'r2b_pre_ece': pre_ece, 'r2b_best_ece': best_ece, 'r2b_epochs': cli.calib_epochs, 'r2b_lr': cli.calib_lr,
           'checkpoints': out, 'seed': cli.seed}, open(os.path.join(cli.out, 'metrics.json'), 'w'), indent=1)
torch.save({'confidence_layer': r2['state'], 'temperature': best_temp}, os.path.join(cli.out, 'r2_r2b_fast.pth'))
say('done')
