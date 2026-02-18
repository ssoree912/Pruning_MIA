"""
This code is modified from https://github.com/Machine-Learning-Security-Lab/mia_prune
"""
import argparse
import json
import pickle
import random
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
THIS_DIR = Path(__file__).resolve().parent

def _find_repo_root(start: Path) -> Path:
    for cand in [start] + list(start.parents):
        if (cand / '.git').exists():
            return cand
        if (cand / 'base_model.py').exists() and (cand / 'mia_eval').exists():
            return cand
    return start.parents[2]

REPO_ROOT = _find_repo_root(THIS_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from mia_eval.core.attackers import MiaAttack
except ImportError:
    from attackers import MiaAttack

from base_model import BaseModel
from mia_eval.core.mia_utils import load_pruned_model
from datasets import get_dataset
from torch.utils.data import ConcatDataset, DataLoader, Subset

try:
    from mia_metrics import compute_mia_metrics, print_mia_metrics
    HAS_METRICS = True
except ImportError:
    print("Warning: mia_metrics not found, using basic metrics only")
    HAS_METRICS = False


parser = argparse.ArgumentParser(description='Membership inference Attacks on Network Pruning')
parser.add_argument('--device', default=0, type=int, help="GPU id to use")
parser.add_argument('--config_path', default=None, type=str, help="config file path")
parser.add_argument('--dataset_name', default='cifar10', type=str)
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=3, type=int)
parser.add_argument('--image_size', default=32, type=int)
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--early_stop', default=5, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--victim_seed', default=42, type=int, help="victim model seed")
parser.add_argument('--shadow_seeds', default=[43,44,45,46,47,48,49,50], nargs='+', type=int, help="shadow model seeds")
parser.add_argument('--defend', default='', type=str)
parser.add_argument('--defend_arg', default=4, type=float)
parser.add_argument('--attacks', default="samia,threshold,nn,nn_top3,nn_cls,lira", type=str)
parser.add_argument('--original', action='store_true', help="Attack original models instead of pruned models")
parser.add_argument('--threshold_strategy', default='youden', choices=['youden', 'max_accuracy', 'fpr_1pct', 'equal_error_rate'], 
                   help="Threshold selection strategy for attacks")
parser.add_argument('--forward_mode', default='standard', choices=['standard', 'scaling', 'dpf'], 
                   help="Forward pass mode for model inference")
parser.add_argument('--debug', action='store_true', help='Print detailed MIA debug info (splits and basic stats)')
parser.add_argument('--tpr_fprs', type=str, default='0.1,1,5',
                    help='Comma-separated FPR percentages to report TPR@FPR (e.g., "0.1,1,5")')
parser.add_argument('--save_scores', action='store_true',
                    help='Save per-sample labels/scores for each attack alongside JSON')
parser.add_argument('--result_file', default=None, type=str,
                    help='Optional absolute/relative JSON output path (overrides default mia_results path)')
parser.add_argument('--victim_ckpt_path', required=True, type=str,
                    help='Direct victim checkpoint path.')
parser.add_argument('--victim_config_path', default=None, type=str,
                    help='Optional direct victim config.json path. Used with --victim_ckpt_path.')
parser.add_argument('--shadow_ckpt_paths', nargs='+', required=True, type=str,
                    help='Direct shadow checkpoint paths (same order as --shadow_seeds).')
parser.add_argument('--shadow_config_paths', nargs='*', default=None, type=str,
                    help='Optional direct shadow config.json paths aligned with --shadow_ckpt_paths.')
parser.add_argument('--failfast_min_acc', default=0.12, type=float,
                    help='Fail-fast threshold on 1-2 batch probe accuracy (fraction). Set <=0 to disable.')
parser.add_argument('--failfast_batches', default=2, type=int,
                    help='Number of probe batches for fail-fast sanity.')


class LoadedModelAdapter(BaseModel):
    """
    BaseModel API adapter backed by a prebuilt checkpoint model.
    Avoids constructing any fallback architecture in MIA.
    """
    def __init__(self, model: torch.nn.Module, device: str, num_cls: int):
        self.model = model.to(device)
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)
        self.num_cls = num_cls
        self.scheduler = None
        self.optimizer = None
        self.optimizer_risk = None


def _model_sanity_print(model: torch.nn.Module, label: str) -> None:
    keys = list(model.state_dict().keys())
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    l1_checksum = 0.0
    with torch.no_grad():
        for p in model.parameters():
            l1_checksum += float(p.detach().float().abs().sum().item())
    print(
        f"[SANITY] {label}: total_params={total_params:,} "
        f"trainable_params={trainable_params:,} "
        f"l1_checksum={l1_checksum:.4e} "
        f"first_state_keys={keys[:5]}"
    )


def _loader_batch_stats(loader: DataLoader, label: str) -> None:
    try:
        x, y = next(iter(loader))
    except StopIteration:
        print(f"[SANITY] {label}: empty loader")
        return
    print(
        f"[SANITY] {label}: x.mean={x.mean().item():.4f} x.std={x.std().item():.4f} "
        f"x.min={x.min().item():.4f} x.max={x.max().item():.4f} "
        f"n={x.size(0)} y0={int(y[0].item()) if y.numel() > 0 else -1}"
    )


def _build_explicit_cifar_eval_loader(dataset_name: str, indices, batch_size: int) -> DataLoader:
    import torchvision
    import torchvision.transforms as transforms

    if dataset_name == "cifar10":
        ds_cls = torchvision.datasets.CIFAR10
    elif dataset_name == "cifar100":
        ds_cls = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Explicit CIFAR eval loader is only supported for cifar10/cifar100, got: {dataset_name}")

    # Match the training/evaluation path used in this project.
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
    trainset = ds_cls(root=f"./data/datasets/{dataset_name}-data", train=True, download=True, transform=eval_tf)
    testset = ds_cls(root=f"./data/datasets/{dataset_name}-data", train=False, download=True, transform=eval_tf)
    total_dataset = ConcatDataset([trainset, testset])
    subset = Subset(total_dataset, list(indices))
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)


def _resolve_config_path(model_path: str, explicit_config_path: str = None) -> str:
    if explicit_config_path:
        cfg = os.path.abspath(os.path.expanduser(explicit_config_path))
        if not os.path.exists(cfg):
            raise FileNotFoundError(f"Provided config path does not exist: {cfg}")
        return cfg

    model_dir = os.path.dirname(model_path)
    candidate_cfgs = [
        os.path.join(model_dir, 'config.json'),
        os.path.join(os.path.dirname(model_dir), 'config.json'),
        os.path.join(os.path.dirname(os.path.dirname(model_dir)), 'config.json'),
    ]
    for c in candidate_cfgs:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"No config.json found near checkpoint: {model_path}. "
        "MIA requires config-aware loading to prevent arch mismatch."
    )


def _validate_optional_path_list(name: str, paths, expected_len: int) -> None:
    if paths is None:
        return
    if len(paths) != expected_len:
        raise ValueError(
            f"{name} length mismatch: expected {expected_len}, got {len(paths)}. "
            f"Provide one path per shadow seed."
        )


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.device}"
    cudnn.benchmark = True
    attack_original = bool(getattr(args, "original", False))
    print(f"Running MIA attack:")
    print(f"Victim seed: {args.victim_seed}")
    print(f"Shadow seeds: {args.shadow_seeds}")
    print(f"Attack mode: {'original(dense baseline)' if attack_original else 'target checkpoint(direct-ckpt)'}")

    _validate_optional_path_list("shadow_ckpt_paths", args.shadow_ckpt_paths, len(args.shadow_seeds))
    _validate_optional_path_list("shadow_config_paths", args.shadow_config_paths, len(args.shadow_seeds))

    # Result location
    if args.result_file:
        result_file = str(Path(args.result_file).expanduser().resolve())
        result_dir = str(Path(result_file).parent)
        os.makedirs(result_dir, exist_ok=True)
    else:
        result_dir = str(REPO_ROOT / 'mia_results' / 'direct')
        os.makedirs(result_dir, exist_ok=True)
        result_file = f"{result_dir}/{args.dataset_name}_victim{args.victim_seed}.json"
    os.makedirs(REPO_ROOT / 'log' / f'{args.dataset_name}', exist_ok=True)

    # Load data splits from fixed split file.
    print("Loading data splits from fixed split file...")
    data_split_path = str(REPO_ROOT / 'mia_data_splits' / f"{args.dataset_name}_seed{args.seed}_victim{args.victim_seed}.pkl")
    if not os.path.exists(data_split_path):
        print(f"❌ Data split file not found: {data_split_path}")
        print("Please run: python mia_eval/create_data/create_fixed_data_splits.py --dataset {args.dataset_name} --victim_seed {args.victim_seed}")
        raise FileNotFoundError(f"Data split file not found: {data_split_path}")
    with open(data_split_path, 'rb') as f:
        data_splits = pickle.load(f)
    print(f"✅ Loaded fixed splits from {data_split_path}")
    victim_train_indices = data_splits['victim']['train_indices']
    victim_test_indices  = data_splits['victim']['test_indices']
    
    # Load full dataset to create subsets
    trainset = get_dataset(args.dataset_name, train=True)
    testset = get_dataset(args.dataset_name, train=False)
    if testset is None:
        total_dataset = trainset
    else:
        total_dataset = ConcatDataset([trainset, testset])
    
    # Create victim datasets using selected indices
    victim_train_dataset = Subset(total_dataset, victim_train_indices)
    victim_test_dataset = Subset(total_dataset, victim_test_indices)
    
    victim_train_loader = DataLoader(victim_train_dataset, batch_size=args.batch_size, 
                                   shuffle=False, num_workers=4, pin_memory=False)
    victim_test_loader = DataLoader(victim_test_dataset, batch_size=args.batch_size, 
                                  shuffle=False, num_workers=4, pin_memory=False)

    # Load victim/shadow model
    def load_model_from_ckpt_path(
        model_path,
        device,
        forward_mode='standard',
        num_cls=10,
        explicit_config_path=None,
    ):
        model_path = os.path.abspath(os.path.expanduser(model_path))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        config_path = _resolve_config_path(model_path, explicit_config_path=explicit_config_path)
        print(f"[Loader] Using checkpoint: {model_path}")
        print(f"[Loader] Using config: {config_path}")
        print("[Loader] Fallback model construction: disabled (config-only + strict state_dict load)")

        try:
            loaded_model, loaded_config = load_pruned_model(model_path, config_path=config_path, device=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load/apply config at {config_path}: {e}")

        # Wrap into BaseModel-compatible interface without constructing any fallback model.
        wrapper = LoadedModelAdapter(loaded_model, device=device, num_cls=num_cls)
        wrapper.loaded_model_path = model_path
        wrapper.loaded_config_path = config_path

        model_cfg = loaded_config.get('model', {})
        arch = model_cfg.get('arch', 'unknown')
        layers = model_cfg.get('layers', 'unknown')
        if arch == 'unknown' or layers == 'unknown':
            raise RuntimeError(
                f"Invalid config at {config_path}: missing model.arch/model.layers for strict model reconstruction."
            )
        print(f"[Loader] Loaded model from config: arch={arch}, layers={layers}")

        # Best-effort: set forward behavior from config when available
        if loaded_config.get('pruning', {}).get('enabled', False):
            method = loaded_config['pruning'].get('method', '').lower()
            if method in ('static', 'dpf', 'dcil'):
                wrapper.preferred_type_value = 5 if method == 'static' else 6
        elif forward_mode == 'scaling' and hasattr(wrapper.model, 'set_scaling_mode'):
            wrapper.model.set_scaling_mode(True)
            print("[Loader] Enabled confidence scaling mode (from args)")
        elif forward_mode == 'dpf':
            wrapper.preferred_type_value = 6

        return wrapper, loaded_config

    print(f"Loading victim model (seed {args.victim_seed}) with forward_mode={args.forward_mode}...")
    victim_model, victim_cfg = load_model_from_ckpt_path(
        model_path=args.victim_ckpt_path,
        device=device,
        forward_mode=args.forward_mode,
        num_cls=args.num_cls,
        explicit_config_path=args.victim_config_path,
    )
    victim_ckpt_used = getattr(victim_model, "loaded_model_path", None)
    victim_cfg_used = getattr(victim_model, "loaded_config_path", None)
    _model_sanity_print(victim_model.model, f"victim(seed={args.victim_seed})")

    # Direct-ckpt mode does not infer separate dense baselines automatically.
    if attack_original:
        print("[WARN] --original requested, but direct-ckpt mode has no auto dense-loader; using victim checkpoint for both paths.")
        victim_dense_model = victim_model
    else:
        print("[INFO] --original is off; using victim checkpoint as target path.")
        victim_dense_model = victim_model
    # Auto-tune type_value if needed to maximize accuracy on a small sample
    def _sample_accuracy(model, loader, tv=None, max_batches=2):
        model.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for bi, (xb, yb) in enumerate(loader):
                if bi >= max_batches:
                    break
                xb, yb = xb.to(model.device), yb.to(model.device)
                try:
                    if tv is None:
                        logits = model._safe_forward(xb)
                    else:
                        logits = model.model(xb, type_value=tv)
                except TypeError:
                    logits = model._safe_forward(xb)
                _, pred = logits.max(1)
                correct += pred.eq(yb).sum().item()
                total += yb.size(0)
        return (correct / total) if total > 0 else 0.0

    def _failfast_loader_accuracy(model, train_loader, test_loader, label):
        probe_batches = max(1, int(args.failfast_batches))
        tr_probe = _sample_accuracy(model, train_loader, tv=None, max_batches=probe_batches)
        te_probe = _sample_accuracy(model, test_loader, tv=None, max_batches=probe_batches)
        print(
            f"[FAIL-FAST] {label}: "
            f"probe_train_acc={tr_probe:.4f}, probe_test_acc={te_probe:.4f}, "
            f"min_required={args.failfast_min_acc:.4f}, batches={probe_batches}"
        )
        if args.failfast_min_acc > 0:
            if tr_probe < args.failfast_min_acc or te_probe < args.failfast_min_acc:
                raise RuntimeError(
                    f"[FAIL-FAST] {label} probe accuracy too low. "
                    f"train={tr_probe:.4f}, test={te_probe:.4f}, min={args.failfast_min_acc:.4f}. "
                    "Likely ckpt/data/config mismatch; aborting MIA run."
                )

    # Try a few candidates commonly used in masked conv paths
    candidate_tvs = [0, 5, 6]
    acc_scores = {}
    for tv in candidate_tvs:
        acc_scores[tv] = _sample_accuracy(victim_model, victim_train_loader, tv=tv, max_batches=2)
    best_tv = max(acc_scores, key=acc_scores.get)
    victim_model.preferred_type_value = best_tv
    print(f"[Victim] Selected type_value={best_tv} (probe acc~{acc_scores[best_tv]*100:.2f}%)")

    _loader_batch_stats(victim_train_loader, "victim_train_loader")
    _loader_batch_stats(victim_test_loader, "victim_test_loader")
    _failfast_loader_accuracy(victim_model, victim_train_loader, victim_test_loader, "victim")

    victim_model.test(victim_train_loader, "Victim Model Train")
    test_acc, loss = victim_model.test(victim_test_loader, "Victim Model Test")
    print(f"Victim model test accuracy: {test_acc:.3f}")
    if args.dataset_name in ("cifar10", "cifar100"):
        try:
            explicit_test_loader = _build_explicit_cifar_eval_loader(
                args.dataset_name,
                victim_test_indices,
                args.batch_size,
            )
            _loader_batch_stats(explicit_test_loader, "explicit_eval_test_loader")
            explicit_acc, _ = victim_model.test(explicit_test_loader, "Victim Model Test (Explicit Eval TF)")
            print(
                f"[SANITY] victim_test_acc(default_loader)={test_acc:.3f} "
                f"victim_test_acc(explicit_eval_tf)={explicit_acc:.3f}"
            )
        except Exception as e:
            print(f"[SANITY] explicit eval-transform comparison skipped: {e}")

    # Debug: split integrity and member/non-member gaps
    if args.debug:
        try:
            # 1) Split integrity
            vt = set(victim_train_indices)
            vte = set(victim_test_indices)
            inter = vt.intersection(vte)
            print(f"[DEBUG] Victim split sizes: train={len(vt)} test={len(vte)} overlap={len(inter)}")
            if len(inter) > 0:
                print("[WARN] Victim train/test overlap detected; splits may be invalid.")
            # Check shadow overlaps with victim train
            split_shadow_map = (locals().get('data_splits', {}) or {}).get('shadows', {}) or {}
            for sid, sdata in split_shadow_map.items():
                st = set(sdata['train_indices'])
                ov = len(vt.intersection(st))
                if ov > 0:
                    print(f"[WARN] Shadow {sid} member set overlaps victim members: {ov} examples")
        except Exception as e:
            print(f"[DEBUG] Split integrity check failed: {e}")

        def _basic_stats(model, loader, label):
            import torch
            import torch.nn.functional as F
            model.model.eval()
            tot_n, loss_sum, conf_sum = 0, 0.0, 0.0
            with torch.no_grad():
                for bi, (xb, yb) in enumerate(loader):
                    xb, yb = xb.to(model.device), yb.to(model.device)
                    try:
                        tv = getattr(model, 'preferred_type_value', None)
                        logits = model.model(xb) if tv is None else model.model(xb, type_value=tv)
                    except TypeError:
                        logits = model.model(xb)
                    probs = torch.softmax(logits, dim=1)
                    conf, _ = probs.max(1)
                    ce = F.cross_entropy(logits, yb, reduction='sum')
                    loss_sum += float(ce.item())
                    conf_sum += float(conf.sum().item())
                    tot_n += yb.size(0)
                    # keep debug pass light
                    if bi >= 10:
                        break
            if tot_n > 0:
                print(f"[DEBUG] {label}: n~{tot_n} mean_loss={loss_sum/tot_n:.4f} mean_conf={conf_sum/tot_n:.4f}")
            else:
                print(f"[DEBUG] {label}: no samples")

        # Victim stats
        _basic_stats(victim_model, victim_train_loader, 'victim members (train)')
        _basic_stats(victim_model, victim_test_loader,  'victim non-members (test)')

    # Helper: extract minimal training meta from saved config.json
    def _extract_training_meta(cfg_dict):
        if not isinstance(cfg_dict, dict):
            return None
        pr = (cfg_dict.get('pruning') or {})
        enabled = bool(pr.get('enabled', True))
        method = (pr.get('method') or ('dense' if not enabled else '')).lower()
        sparsity = pr.get('sparsity', None)
        seed = cfg_dict.get('seed')
        if seed is None:
            seed = (cfg_dict.get('system') or {}).get('seed')
        if seed is None:
            seed = (cfg_dict.get('training') or {}).get('seed')
        try:
            seed = int(seed) if seed is not None else None
        except Exception:
            pass
        return {'enabled': enabled, 'method': method, 'sparsity': sparsity, 'seed': seed}

    victim_meta = _extract_training_meta(victim_cfg)
    if args.debug and victim_meta:
        print(f"[DEBUG] Victim meta: enabled={victim_meta['enabled']} method={victim_meta['method']} sparsity={victim_meta['sparsity']} seed={victim_meta['seed']}")

    # Load shadow models with fixed data splits
    shadow_model_list = []          # target checkpoint shadows
    shadow_dense_model_list = []    # dense shadows (used only in --original mode)
    shadow_train_loader_list = []
    shadow_test_loader_list = []
    shadow_cfg_map = {}
    shadow_ckpt_used_map = {}
    shadow_config_used_map = {}
    
    total_shadows = len(args.shadow_seeds)
    for i, shadow_seed in enumerate(args.shadow_seeds):
        if shadow_seed not in data_splits['shadows']:
            print(f"⚠️ Warning: Shadow seed {shadow_seed} not in data splits, skipping...")
            continue
            
        print(f"[{i+1}/{total_shadows}] Loading shadow model (seed {shadow_seed}) with forward_mode={args.forward_mode}...")
        s_ckpt = args.shadow_ckpt_paths[i]
        s_cfg_override = args.shadow_config_paths[i] if args.shadow_config_paths is not None else None
        shadow_model, s_cfg = load_model_from_ckpt_path(
            model_path=s_ckpt,
            device=device,
            forward_mode=args.forward_mode,
            num_cls=args.num_cls,
            explicit_config_path=s_cfg_override,
        )
        # Direct-ckpt mode does not infer separate dense baselines automatically.
        if attack_original:
            shadow_dense_model = shadow_model
        else:
            shadow_dense_model = shadow_model
        shadow_cfg_map[str(shadow_seed)] = s_cfg
        shadow_ckpt_used_map[str(shadow_seed)] = getattr(shadow_model, "loaded_model_path", None)
        shadow_config_used_map[str(shadow_seed)] = getattr(shadow_model, "loaded_config_path", None)

        # Validate shadow config vs victim to guard against misfoldered runs
        s_meta = _extract_training_meta(s_cfg)
        problems = []
        if s_meta is None:
            problems.append('no_config')
        else:
            # Folder seed vs config seed
            if s_meta.get('seed') is not None and s_meta['seed'] != shadow_seed:
                problems.append(f"seed_mismatch(cfg={s_meta['seed']} vs folder={shadow_seed})")
            if victim_meta is not None:
                # Dense vs pruned must match
                if s_meta['enabled'] != victim_meta['enabled']:
                    problems.append(f"enabled_mismatch(shadow={s_meta['enabled']} vs victim={victim_meta['enabled']})")
                # Sparsity must match when pruned
                if s_meta['enabled']:
                    try:
                        vs_raw = victim_meta.get('sparsity')
                        ss_raw = s_meta.get('sparsity')
                        if vs_raw is not None and ss_raw is not None:
                            vs = float(vs_raw)
                            ss = float(ss_raw)
                            if abs(vs - ss) > 1e-6:
                                problems.append(f"sparsity_mismatch(shadow={ss} vs victim={vs})")
                    except Exception:
                        pass
                # Method check (be lenient with 'dcil' backend)
                vm = (victim_meta.get('method') or '').lower()
                sm = (s_meta.get('method') or '').lower()
                def _norm(m):
                    return 'dense' if m in ('', None) else ('pruned' if m in ('dcil','static','dpf') else m)
                if _norm(vm) != _norm(sm):
                    problems.append(f"method_mismatch(shadow={sm} vs victim={vm})")

        if problems:
            print(f"[WARN] Skipping shadow seed {shadow_seed}: {'; '.join(problems)}")
            continue
        
        shadow_data = data_splits['shadows'][shadow_seed]
        shadow_train_indices = shadow_data['train_indices']  # members
        shadow_test_indices  = shadow_data['test_indices']   # non-members
        
        shadow_train_dataset = Subset(total_dataset, shadow_train_indices)
        shadow_test_dataset = Subset(total_dataset, shadow_test_indices)
        
        shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=args.batch_size,
                                       shuffle=False, num_workers=4, pin_memory=False)
        shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=4, pin_memory=False)

        # Auto-tune type_value for shadow on its own member split, then fail-fast sanity.
        acc_scores = {tv: _sample_accuracy(shadow_model, shadow_train_loader, tv=tv, max_batches=max(1, int(args.failfast_batches))) for tv in candidate_tvs}
        best_tv = max(acc_scores, key=acc_scores.get)
        shadow_model.preferred_type_value = best_tv
        print(f"[Shadow {shadow_seed}] Selected type_value={best_tv} (probe acc~{acc_scores[best_tv]*100:.2f}%)")
        _failfast_loader_accuracy(shadow_model, shadow_train_loader, shadow_test_loader, f"shadow(seed={shadow_seed})")
        
        print(f"[{i+1}/{total_shadows}] Shadow {shadow_seed}: {len(shadow_train_indices)} members, {len(shadow_test_indices)} non-members")
        shadow_model.test(shadow_train_loader, f"[{i+1}/{total_shadows}] Shadow Model {shadow_seed} Train (Members)")
        shadow_model.test(shadow_test_loader, f"[{i+1}/{total_shadows}] Shadow Model {shadow_seed} Test (Non-members)")

        shadow_model_list.append(shadow_model)
        shadow_dense_model_list.append(shadow_dense_model)
        shadow_train_loader_list.append(shadow_train_loader)
        shadow_test_loader_list.append(shadow_test_loader)

        # Debug: print quick stats for first few shadows
        if args.debug and i < 3:
            _basic_stats(shadow_model, shadow_train_loader, f'shadow {shadow_seed} members (train)')
            _basic_stats(shadow_model, shadow_test_loader,  f'shadow {shadow_seed} non-members (test)')

    print("Start Membership Inference Attacks")
    
    # Prepare optional scores directory
    scores_dir = None
    if args.save_scores:
        base = Path(result_dir) / (Path(result_file).stem + "_scores")
        base.mkdir(parents=True, exist_ok=True)
        scores_dir = str(base)

    attacker = MiaAttack(
        victim_dense_model, victim_model, victim_train_loader, victim_test_loader,
        shadow_dense_model_list, shadow_model_list, shadow_train_loader_list, shadow_test_loader_list,
        num_cls=args.num_cls, device=device, batch_size=args.batch_size,
        attack_original=attack_original,
        tpr_fprs=args.tpr_fprs,  # propagate desired FPR levels
        save_scores_dir=scores_dir
    )

    attacks = args.attacks.split(',')
    results = {}
    
    if "samia" in attacks:
        samia_metrics = attacker.nn_attack("nn_sens_cls", model_name="transformer")
        results['samia'] = samia_metrics
        print(f"SAMIA: Acc={samia_metrics['accuracy']:.3f}, AUC={samia_metrics['auc']:.3f}, BalAcc={samia_metrics['balanced_accuracy']:.3f}, Adv={samia_metrics['advantage']:.3f}")
    
    if "threshold" in attacks:
        conf, xent, mentr, top1_conf = attacker.threshold_attack()
        results['confidence'] = conf
        results['entropy'] = xent
        results['modified_entropy'] = mentr
        results['top1_conf'] = top1_conf
        
        print(f"Confidence attack accuracy: {conf:.3f}")
        print(f"Entropy attack accuracy: {xent:.3f}")
        print(f"Modified entropy attack accuracy: {mentr:.3f}")
        print(f"Top1 confidence attack accuracy: {top1_conf:.3f}")
        
        # Extended metrics (inline): AUROC, Balanced Accuracy, Advantage using Youden threshold,
        # and TPR@X%FPR for requested X values
        try:
            from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_recall_fscore_support, average_precision_score
            import numpy as _np
            vin = attacker.victim_in_predicts.max(dim=1)[0].detach().cpu().numpy()
            vout = attacker.victim_out_predicts.max(dim=1)[0].detach().cpu().numpy()
            y_true = _np.concatenate([_np.ones_like(vin), _np.zeros_like(vout)])
            y_score = _np.concatenate([vin, vout])
            auroc = float(roc_auc_score(y_true, y_score)) if len(_np.unique(y_true)) > 1 else 0.0
            vals = _np.unique(y_score)
            best_adv, best_thr = -1.0, 0.5
            for thr in vals:
                y_pred = (y_score >= thr).astype(int)
                tp = ((y_pred == 1) & (y_true == 1)).sum(); fn = ((y_pred == 0) & (y_true == 1)).sum()
                tn = ((y_pred == 0) & (y_true == 0)).sum(); fp = ((y_pred == 1) & (y_true == 0)).sum()
                tpr = tp / (tp + fn + 1e-8); fpr = fp / (fp + tn + 1e-8)
                adv = tpr - fpr
                if adv > best_adv:
                    best_adv, best_thr = adv, thr
            y_pred = (y_score >= best_thr).astype(int)
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
            # PR/F1 at chosen threshold
            try:
                prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            except Exception:
                prec = rec = f1 = 0.0
            # AP (area under PR curve)
            try:
                ap = float(average_precision_score(y_true, y_score)) if len(_np.unique(y_true)) > 1 else 0.0
            except Exception:
                ap = 0.0
            # TPR@X%FPR via quantiles of non-member scores
            non_member = y_score[y_true == 0]
            member = y_score[y_true == 1]
            tpr_levels = {}
            try:
                want_fprs = [float(s.strip()) for s in (args.tpr_fprs or '').split(',') if s.strip()]
            except Exception:
                want_fprs = [1.0]
            if non_member.size > 0 and member.size > 0:
                for fpr_pct in want_fprs:
                    q = max(0.0, min(1.0, 1.0 - (fpr_pct/100.0)))
                    tau = _np.quantile(non_member, q)
                    tpr_val = float((member >= tau).mean())
                    key = f"{fpr_pct:g}"
                    tpr_levels[key] = tpr_val
            # Back-compat single 1%% metric if requested
            tpr_at_1fpr = tpr_levels.get('1', None)
            ce = {
                'auroc': auroc,
                'balanced_accuracy': bal_acc,
                'advantage': float(best_adv),
                'threshold': float(best_thr),
                'precision': float(prec),
                'recall': float(rec),
                'f1': float(f1),
                'ap': ap,
                'tpr_at_fprs': tpr_levels,
                **({'tpr_at_1fpr': tpr_at_1fpr} if tpr_at_1fpr is not None else {})
            }
            results['confidence_extended'] = ce
            results['threshold_strategy'] = 'youden'
            tprs_msg = ", ".join([f"TPR@{k}%FPR={v:.4f}" for k, v in sorted(tpr_levels.items(), key=lambda x: float(x[0]))]) if tpr_levels else ""
            print(f"\n📊 Confidence extended metrics: AUROC={auroc:.4f}, BalAcc={bal_acc:.4f}, Adv={best_adv:.4f}, Thr={best_thr:.4f}{(' | ' + tprs_msg) if tprs_msg else ''}")
            # Optional: save per-sample arrays for threshold confidence
            if scores_dir:
                try:
                    import numpy as _np
                    outp = Path(scores_dir) / "threshold_confidence.npz"
                    _np.savez(outp, labels=y_true, scores=y_score)
                    # Track path for convenience
                    results.setdefault('raw_scores', {})['threshold_confidence'] = str(outp)
                except Exception:
                    pass
        except Exception as e:
            print(f"Could not compute extended metrics inline: {e}")
    
    if "nn" in attacks:
        nn_metrics = attacker.nn_attack("nn")
        results['nn'] = nn_metrics
        print(f"NN: Acc={nn_metrics['accuracy']:.3f}, AUC={nn_metrics['auc']:.3f}, BalAcc={nn_metrics['balanced_accuracy']:.3f}, Adv={nn_metrics['advantage']:.3f}")
    
    if "nn_top3" in attacks:
        nn_top3_metrics = attacker.nn_attack("nn_top3")
        results['nn_top3'] = nn_top3_metrics
        print(f"Top3-NN: Acc={nn_top3_metrics['accuracy']:.3f}, AUC={nn_top3_metrics['auc']:.3f}, BalAcc={nn_top3_metrics['balanced_accuracy']:.3f}, Adv={nn_top3_metrics['advantage']:.3f}")
    
    if "nn_cls" in attacks:
        nn_cls_metrics = attacker.nn_attack("nn_cls")
        results['nn_cls'] = nn_cls_metrics
        print(f"NN-Cls: Acc={nn_cls_metrics['accuracy']:.3f}, AUC={nn_cls_metrics['auc']:.3f}, BalAcc={nn_cls_metrics['balanced_accuracy']:.3f}, Adv={nn_cls_metrics['advantage']:.3f}")

    if "lira" in attacks:
        lira = attacker.lira_attack()
        results['lira'] = lira
        print(f"LiRA: AUC={lira['auc']:.3f}, Acc={lira['accuracy']:.3f}, BalAcc={lira['balanced_accuracy']:.3f}, Adv={lira['advantage']:.3f}")
    
    # Build data split summary
    victim_members_count = len(data_splits['victim']['train_indices'])
    victim_nonmembers_count = len(data_splits['victim']['test_indices'])
    shadow_counts = {str(k): len(v['train_indices']) for k, v in data_splits['shadows'].items()}
    split_source = data_split_path

    # Save results with data split info
    import json
    with open(result_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': results,
            'victim_test_acc': test_acc,
            'experiment_info': {
                'forward_mode': args.forward_mode,
                'threshold_strategy': args.threshold_strategy,
                'attack_mode': 'original' if (hasattr(args, 'original') and args.original) else 'direct-ckpt',
                'loader_paths': {
                    'victim_ckpt_path': victim_ckpt_used,
                    'victim_config_path': victim_cfg_used,
                    'shadow_ckpt_paths': shadow_ckpt_used_map,
                    'shadow_config_paths': shadow_config_used_map,
                },
                'victim_config': victim_cfg,
                'shadow_configs': shadow_cfg_map
            },
            'data_splits_info': {
                'victim_members': victim_members_count,
                'victim_nonmembers': victim_nonmembers_count,
                'shadow_counts': shadow_counts,
                'split_file': split_source
            }
        }, f, indent=2)
    
    print(f"Results saved to {result_file}")

if __name__ == '__main__':
    args = parser.parse_args()
    
    # config 파일이 지정되면 로드
    if args.config_path and os.path.exists(args.config_path):
        print(f"Loading config from {args.config_path}")
        with open(args.config_path) as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    
    print(f"Arguments: {args}")
    main(args)
