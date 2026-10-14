"""Pack medpatch/ (code only) + prepare_data.py into medpatch_code.zip for Google Drive."""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, '..', '..', 'medpatch')
OUT = os.path.join(HERE, 'medpatch_code.zip')
SKIP_DIRS = {'__pycache__', '.pytest_cache', 'wandb', 'checkpoints'}
SKIP_EXT = {'.pyc', '.pth', '.tar', '.ckpt', '.log'}

n = 0
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if os.path.splitext(f)[1] in SKIP_EXT:
                continue
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, REPO).replace(os.sep, '/'))
            n += 1
    z.write(os.path.join(HERE, 'prepare_data.py'), 'prepare_data.py')
print(f'wrote {OUT}: {n + 1} files, {os.path.getsize(OUT) / 1e6:.1f} MB')
