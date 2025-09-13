# Pruning + MIA Pipeline

Train One‑Shot/Static/DPF and DWA pruned models, then evaluate MIA (LiRA, NN, etc.) 

## File Tree

```
.
├── train.py                      # One‑Shot / Static / DPF training (single/multi‑seed)
├── train_dwa.py                  # DWA training (single/multi‑seed)
├── run_experiment.py             # One‑Shot internal runner
├── run_experiment_dwa.py         # DWA internal runner
├── datasets.py                   # Canonical datasets (root‑level)
├── data.py                       # CIFAR DataLoader helpers
├── scripts/
│   └── summarize_runs.py         # Scan runs/ and write summary CSV
├── mia_eval/
│   ├── runners/
│   │   ├── run_single_mia.py     # Run MIA for a single combination
│   │   └── run_mia_pipeline.py   # Scan runs/ and evaluate many combos
│   ├── create_data/
│   │   └── create_fixed_data_splits.py  # Create fixed pkl splits for MIA
│   └── core/
│       ├── mia_modi.py           # MIA core (LiRA, NN, threshold)
│       ├── attackers.py
│       └── mia_utils.py
├── runs/                         # Checkpoints (git‑ignored)
├── mia_results/                  # MIA outputs (git‑ignored)
├── mia_data_splits/              # Split pkls (git‑ignored)
└── Readme.md
```

## Quick Start

1) Install dependencies

```
pip install -r requirements.txt
```

2) Train models (examples below)

3) Create fixed MIA splits (pkl)

```
python mia_eval/create_data/create_fixed_data_splits.py \
  --dataset cifar10 --seed 7 \
  --victim_seed 42 \
  --shadow_seeds 43 44 45 46 47 48 49 50 \
  --verify
```

4) Run MIA

```
# single combination
python mia_eval/runners/run_single_mia.py \
  --dataset cifar10 \
  --sparsity 0.9 --alpha 5.0 --beta 5.0 \
  --prune_method dwa --prune_type reactivate_only \
  --victim_seed 42 --shadow_seeds 43 44 45 46 47 48 49 50 \
  --device 0 --forward_mode standard \
  --attacks samia,threshold,nn,nn_top3,nn_cls,lira \
  --debug

# batch over runs/
python mia_eval/runners/run_mia_pipeline.py --dataset cifar10 --debug
```

5) Summarize

```
python scripts/summarize_runs.py --runs ./runs --out results/runs_summary.csv
```

## Setup (requirements.txt)

```
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# CPU (or default)
pip install --upgrade pip
pip install -r requirements.txt

# GPU (example: CUDA 12.1)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements.txt
```

## Setup (Docker)


Build / Run:

```
docker build -t prune-mia .
docker run --rm -it --gpus all --ipc=host -v $(pwd):/workspace -w /workspace prune-mia bash
```

## Training

One‑Shot / Static / DPF

```
# single seed
python train.py --methods static --sparsities 0.9 --epochs 200 --freeze-epoch 180 --seed 42 --gpu 0

# multi‑seed (contiguous)
python train.py --methods static dpf --sparsities 0.7 0.8 0.9 \
  --epochs 200 --freeze-epoch 180 --multi-seed --num-seeds 32 --start-seed 43 --gpu 0
```

DWA

```
# single seed
python train_dwa.py --dwa-modes reactivate_only --sparsities 0.9 \
  --dwa-alphas 5.0 --dwa-betas 5.0 \
  --epochs 200 --target-epoch 75 --prune-freq 16 --seed 42 --gpu 0

# multi‑seed
python train_dwa.py --dwa-modes reactivate_only kill_active_plain_dead --sparsities 0.7 0.8 0.9 \
  --dwa-alphas 5.0 --dwa-betas 5.0 \
  --epochs 200 --target-epoch 75 --prune-freq 16 --multi-seed --num-seeds 32 --start-seed 43 --gpu 0
```

