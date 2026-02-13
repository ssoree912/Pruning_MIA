#!/usr/bin/env python3
"""
Late-branch training runner:
  Stage-1: shared training (same init_seed, same data_seed) up to split_epoch
  Stage-2: particle tails (resume from split ckpt; same init, different data_seed per particle)
"""

import argparse
import subprocess
from pathlib import Path


def run(cmd):
    print(" ".join(map(str, cmd)))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["static", "dpf", "dense"])
    ap.add_argument("--sparsity", type=float, default=None)
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--arch", default="resnet")
    ap.add_argument("--total-epochs", type=int, default=200)
    ap.add_argument("--split-epoch", type=int, default=150)

    ap.add_argument("--init-seed", type=int, default=1234)
    ap.add_argument("--base-data-seed", type=int, default=777)
    ap.add_argument("--start-seed", type=int, default=42)
    ap.add_argument("--num-seeds", type=int, default=3)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data-seed-offset", type=int, default=0)

    ap.add_argument("--runs", default="./runs")
    args = ap.parse_args()

    runs = Path(args.runs)
    if args.method == "dense":
        base_dir = runs / "dense" / args.dataset
    else:
        base_dir = runs / args.method / f"sparsity_{args.sparsity}" / args.dataset

    base_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: shared training ----
    stage1_dir = base_dir / "seed0_stage1"
    stage1_dir.mkdir(parents=True, exist_ok=True)

    save_split_epoch = args.split_epoch - 1
    stage1_cmd = [
        "python", "-u", "run_experiment.py",
        "--name", f"latebranch_stage1_{args.method}",
        "--save-dir", str(stage1_dir),
        "--dataset", args.dataset,
        "--arch", args.arch,
        "--epochs", str(args.total_epochs),
        "--seed", "0",
        "--init-seed", str(args.init_seed),
        "--data-seed", str(args.base_data_seed),
        "--gpu", str(args.gpu),
        "--save-split-ckpt-epoch", str(save_split_epoch),
    ]
    if args.method != "dense":
        stage1_cmd += ["--prune", "--prune-method", args.method, "--sparsity", str(args.sparsity)]
    run(stage1_cmd)

    split_ckpt = stage1_dir / f"split_ckpt_epoch{save_split_epoch}.pth"
    if not split_ckpt.exists():
        raise SystemExit(f"split ckpt not found: {split_ckpt}")

    # ---- Stage 2: particle tails ----
    for i in range(args.num_seeds):
        seed = args.start_seed + i
        particle_dir = base_dir / f"seed{seed}"
        particle_dir.mkdir(parents=True, exist_ok=True)

        data_seed = seed + args.data_seed_offset
        stage2_cmd = [
            "python", "-u", "run_experiment.py",
            "--name", f"latebranch_stage2_{args.method}_seed{seed}",
            "--save-dir", str(particle_dir),
            "--dataset", args.dataset,
            "--arch", args.arch,
            "--epochs", str(args.total_epochs),
            "--start-epoch", str(args.split_epoch),
            "--resume", str(split_ckpt),
            "--seed", str(seed),
            "--init-seed", str(args.init_seed),
            "--data-seed", str(data_seed),
            "--gpu", str(args.gpu),
        ]
        if args.method != "dense":
            stage2_cmd += ["--prune", "--prune-method", args.method, "--sparsity", str(args.sparsity)]
        run(stage2_cmd)

    print("[DONE] late-branch training complete.")
    print(f"Stage-1 split ckpt: {split_ckpt}")


if __name__ == "__main__":
    main()
