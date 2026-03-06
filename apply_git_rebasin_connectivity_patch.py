#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

SRC_ROOT = Path('/mnt/data/connectivity_bundle')
FILES = [
    'connectivity/__init__.py',
    'connectivity/perm_spec_resnet.py',
    'connectivity/weight_matching_torch.py',
    'connectivity/git_rebasin_resnet.py',
    'connectivity/functional_modes.py',
    'connectivity/run_connectivity_experiment.py',
    'scripts/experiment/run_df1_seed42_connectivity.sh',
]


def main(repo_root: Path) -> None:
    for rel in FILES:
        src = SRC_ROOT / rel
        dst = repo_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            backup = dst.with_suffix(dst.suffix + '.bak')
            shutil.copy2(dst, backup)
            print(f'backup: {backup}')
        shutil.copy2(src, dst)
        print(f'updated: {dst}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('repo_root', nargs='?', default='.')
    args = parser.parse_args()
    main(Path(args.repo_root).resolve())
