"""Convert a fast-path head (train_head.py -> best_head.pth) into a repo checkpoint for --fusion_type unimodal_rr / unimodal_dn.

Builds the repo's own model (models.fusion.Fusion + models.text_models.Text_encoder, same args as fusion_main.py),
which loads the frozen pretrained BERT, then copies the trained head:
  head.fc        -> text_model.fc_<m>                       (repo: Linear 768 -> 512)
  head.layernorm -> fusion_model.<m>_classifier.classifier.layernorm
  head.out       -> fusion_model.<m>_classifier.classifier.fc
and saves {'epoch','state_dict','best_auroc', ...} at the exact path MSMA_Trainer.save_checkpoint/eval use:
  <save_dir>/<task>/unimodal_<m>/best_checkpoint_<lr>_<task>_unimodal_<m>_EHR-<M>_paired.pth.tar
Prints only parameter names/shapes and the output path.
"""
import argparse
import os
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True)
ap.add_argument('--head', required=True, help='best_head.pth from train_head.py')
ap.add_argument('--modality', choices=['RR', 'DN'], required=True)
ap.add_argument('--task', choices=['phenotyping', 'in-hospital-mortality'], required=True)
ap.add_argument('--save_dir', required=True, help='the --save_dir later commands will use')
cli = ap.parse_args()

sys.path.insert(0, os.path.abspath(cli.repo))
from arguments import args_parser  # noqa: E402
from models.fusion import Fusion  # noqa: E402
from models.text_models import Text_encoder  # noqa: E402

head = torch.load(cli.head, map_location='cpu', weights_only=False)
meta = head['cache_meta']
assert meta['modality'] == cli.modality and meta.get('task', 'phenotyping') == cli.task, f'head was trained for {meta}'
lr = head['lr']
m, M = cli.modality.lower(), cli.modality
nc = 25 if cli.task == 'phenotyping' else 1
a = args_parser().parse_args([
    '--fusion_type', f'unimodal_{m}', '--modalities', f'EHR-{M}', '--task', cli.task, '--num_classes', str(nc),
    '--labels_set', 'pheno' if cli.task == 'phenotyping' else 'mortality', '--text_encoder', 'biobert',
    '--classifier', 'mlp', '--loss', 'bce', '--data_pairs', 'paired', '--lr', repr(lr),
    '--bert_model_name', meta['model']])
model = Fusion(a, None, None, Text_encoder(a, torch.device('cpu')))
sd, hs = model.state_dict(), head['state_dict']
mapping = {f'text_model.fc_{m}.weight': 'fc.weight', f'text_model.fc_{m}.bias': 'fc.bias',
           f'fusion_model.{m}_classifier.classifier.layernorm.weight': 'layernorm.weight',
           f'fusion_model.{m}_classifier.classifier.layernorm.bias': 'layernorm.bias',
           f'fusion_model.{m}_classifier.classifier.fc.weight': 'out.weight',
           f'fusion_model.{m}_classifier.classifier.fc.bias': 'out.bias'}
for repo_key, head_key in mapping.items():
    assert sd[repo_key].shape == hs[head_key].shape, (repo_key, sd[repo_key].shape, hs[head_key].shape)
    sd[repo_key].copy_(hs[head_key])
    print(f'  {head_key:18s} -> {repo_key}  {tuple(hs[head_key].shape)}')
# text_model is shared between Fusion.text_model and fusion_model.text_model: check both names now hold the head
assert torch.equal(sd[f'fusion_model.text_model.fc_{m}.weight'], hs['fc.weight'])

out_dir = os.path.join(cli.save_dir, cli.task, f'unimodal_{m}')
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, f'best_checkpoint_{a.lr}_{cli.task}_unimodal_{m}_EHR-{M}_paired.pth.tar')
torch.save({'epoch': -1, 'state_dict': sd, 'best_auroc': None, 'optimizer': None,
            'converted_from': os.path.abspath(cli.head), 'bert_model_name': meta['model'], 'lr': lr}, path)
print('REPO CHECKPOINT:', os.path.abspath(path))
