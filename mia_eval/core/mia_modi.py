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
from mia_eval.core.mia_utils import load_dwa_model
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
parser.add_argument('--model_name', default='auto', type=str, help='Model resolved from config; use only as fallback')
parser.add_argument('--num_cls', default=10, type=int)
parser.add_argument('--input_dim', default=3, type=int)
parser.add_argument('--image_size', default=32, type=int)
parser.add_argument('--hidden_size', default=128, type=int)
parser.add_argument('--seed', default=7, type=int)
parser.add_argument('--early_stop', default=5, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--pruner_name', default='l1unstructure', type=str, help="prune method for victim model")
parser.add_argument('--sparsity', default=0.9, type=float, help="sparsity level (same for all models)")
parser.add_argument('--victim_seed', default=42, type=int, help="victim model seed")
parser.add_argument('--shadow_seeds', default=[43,44,45,46,47,48,49,50], nargs='+', type=int, help="shadow model seeds")
parser.add_argument('--alpha', default=5.0, type=float)
parser.add_argument('--beta', default=5.0, type=float)
parser.add_argument('--prune_method', default='dwa', type=str, choices=['dwa','static','dpf','dense'])
parser.add_argument('--prune_type', default='reactivate_only', type=str, help='DWA mode (ignored for others)')
parser.add_argument('--freeze_tag', default=None, type=str, help='DPF only: sparsity_<s>_<freeze_tag> selector (e.g., freeze180 or nofreeze)')
parser.add_argument('--defend', default='', type=str)
parser.add_argument('--defend_arg', default=4, type=float)
parser.add_argument('--attacks', default="samia,threshold,nn,nn_top3,nn_cls,lira", type=str)
parser.add_argument('--original', action='store_true', help="Attack original models instead of pruned models")
parser.add_argument('--threshold_strategy', default='youden', choices=['youden', 'max_accuracy', 'fpr_1pct', 'equal_error_rate'], 
                   help="Threshold selection strategy for attacks")
parser.add_argument('--forward_mode', default='standard', choices=['standard', 'dwa_adaptive', 'scaling', 'dpf'], 
                   help="Forward pass mode for model inference")
parser.add_argument('--debug', action='store_true', help='Print detailed MIA debug info (splits and basic stats)')
parser.add_argument('--tpr_fprs', type=str, default='0.1,1,5',
                    help='Comma-separated FPR percentages to report TPR@FPR (e.g., "0.1,1,5")')


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = f"cuda:{args.device}"
    cudnn.benchmark = True
    print(f"Running MIA attack:")
    print(f"Victim seed: {args.victim_seed}")
    print(f"Shadow seeds: {args.shadow_seeds}")
    print(f"Sparsity: {args.sparsity}")
    print(f"Alpha: {args.alpha}, Beta: {args.beta}")

    base_path = str(REPO_ROOT / "runs")
    # Result location: keep rich naming for DWA; simplify for others
    if args.prune_method == 'dwa':
        result_dir = str(REPO_ROOT / 'mia_results' / f"{args.prune_method}_{args.prune_type}")
        os.makedirs(result_dir, exist_ok=True)
        result_file = f"{result_dir}/sparsity_{args.sparsity}_alpha{args.alpha}_beta{args.beta}_victim{args.victim_seed}.json"
    elif args.prune_method == 'dpf':
        tag = f"_{args.freeze_tag}" if args.freeze_tag else ''
        result_dir = str(REPO_ROOT / 'mia_results' / f"dpf{tag}")
        os.makedirs(result_dir, exist_ok=True)
        result_file = f"{result_dir}/{args.dataset_name}_sparsity_{args.sparsity}_victim{args.victim_seed}.json"
    elif args.prune_method == 'static':
        result_dir = str(REPO_ROOT / 'mia_results' / 'static')
        os.makedirs(result_dir, exist_ok=True)
        result_file = f"{result_dir}/{args.dataset_name}_sparsity_{args.sparsity}_victim{args.victim_seed}.json"
    else:  # dense
        result_dir = str(REPO_ROOT / 'mia_results' / 'dense')
        os.makedirs(result_dir, exist_ok=True)
        result_file = f"{result_dir}/{args.dataset_name}_victim{args.victim_seed}.json"
    os.makedirs(REPO_ROOT / 'log' / f'{args.dataset_name}_{args.model_name}', exist_ok=True)

    # Load data splits: prefer training-time data_prepare.pkl if available
    print("Loading data splits (prefer training-time data_prepare.pkl if available)...")

    # Try to locate experiment directory for this victim
    # Locate experiment directory depending on method
    if args.prune_method == 'dwa':
        exp_dir = Path(base_path) / args.prune_method / args.prune_type / f"sparsity_{args.sparsity}" / args.dataset_name / f"alpha{args.alpha}_beta{args.beta}"
    elif args.prune_method == 'static':
        exp_dir = Path(base_path) / 'static' / f"sparsity_{args.sparsity}" / args.dataset_name
    elif args.prune_method == 'dpf':
        tag = f"_{args.freeze_tag}" if args.freeze_tag else ''
        exp_dir = Path(base_path) / 'dpf' / f"sparsity_{args.sparsity}{tag}" / args.dataset_name
    elif args.prune_method == 'dense':
        exp_dir = Path(base_path) / 'dense' / args.dataset_name
    else:
        exp_dir = Path(base_path)
    data_prepare_path = exp_dir / 'data_prepare.pkl'

    use_training_splits = False
    if data_prepare_path.exists():
        try:
            with open(data_prepare_path, 'rb') as f:
                vp_idx, v_tr, v_dv, v_te, attack_split_list, shadow_train_list = pickle.load(f)
            # Extract indices from Subset objects
            victim_train_indices = getattr(v_tr, 'indices', None) or v_tr.dataset.indices
            victim_test_indices = getattr(v_te, 'indices', None) or v_te.dataset.indices
            training_shadow_splits = []
            for tr, dv, te in attack_split_list:
                tr_idx = getattr(tr, 'indices', None) or tr.dataset.indices
                te_idx = getattr(te, 'indices', None) or te.dataset.indices
                training_shadow_splits.append((tr_idx, te_idx))
            use_training_splits = True
            print(f"✅ Using training-time splits: {data_prepare_path}")
        except Exception as e:
            print(f"⚠️ Failed to parse training splits ({e}); falling back to fixed pkl.")

    if not use_training_splits:
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

    # Load victim model
    def load_model_from_seed_folder(base_path, seed, dataset_name, model_name, sparsity, alpha, beta, 
                                   prune_method, prune_type, device, forward_mode='standard', num_cls=10, input_dim=3,
                                   freeze_tag=None):
        # Build seed folder by method
        if prune_method == 'dwa':
            model_dir = f"{base_path}/{prune_method}/{prune_type}/sparsity_{sparsity}/{dataset_name}/alpha{alpha}_beta{beta}/seed{seed}"
        elif prune_method == 'static':
            model_dir = f"{base_path}/static/sparsity_{sparsity}/{dataset_name}/seed{seed}"
        elif prune_method == 'dpf':
            tag = f"_{freeze_tag}" if freeze_tag else ''
            model_dir = f"{base_path}/dpf/sparsity_{sparsity}{tag}/{dataset_name}/seed{seed}"
        elif prune_method == 'dense':
            model_dir = f"{base_path}/dense/{dataset_name}/seed{seed}"
        else:
            model_dir = f"{base_path}/{prune_method}/{prune_type}/sparsity_{sparsity}/{dataset_name}/alpha{alpha}_beta{beta}/seed{seed}"
        model_path = f"{model_dir}/best_model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading model with forward_mode: {forward_mode}")

        # Prefer config-aware loading whenever a config.json is found (any method)
        loaded_config = None
        # Try to locate a config.json near the seed folder
        candidate_cfgs = [
            os.path.join(model_dir, 'config.json'),
            os.path.join(os.path.dirname(model_dir), 'config.json'),
            os.path.join(os.path.dirname(os.path.dirname(model_dir)), 'config.json')
        ]
        config_path = None
        for c in candidate_cfgs:
            if os.path.exists(c):
                config_path = c
                break
        if config_path:
            print(f"[Loader] Using config: {config_path}")
        try:
            if config_path:
                loaded_model, loaded_config = load_dwa_model(model_path, config_path=config_path, device=device)
                # Wrap into BaseModel interface for downstream predict_target_sensitivity
                safe_name = model_name if (model_name and model_name != 'auto') else 'resnet18'
                wrapper = BaseModel(safe_name, num_cls=num_cls, input_dim=input_dim, device=device)
                wrapper.model = loaded_model.to(device)

                # Best-effort: set forward behavior from config when available
                if loaded_config and loaded_config.get('pruning', {}).get('enabled', False):
                    method = loaded_config['pruning'].get('method', '').lower()
                    if method == 'dwa' and hasattr(wrapper.model, 'set_dwa_params'):
                        cfg_alpha = loaded_config['pruning'].get('dwa_alpha', alpha)
                        cfg_beta  = loaded_config['pruning'].get('dwa_beta', beta)
                        cfg_mode  = loaded_config['pruning'].get('dwa_mode', prune_type)
                        wrapper.model.set_dwa_params(alpha=cfg_alpha, beta=cfg_beta, mode=cfg_mode)
                        print(f"[Loader] Applied DWA forward: mode={cfg_mode}, alpha={cfg_alpha}, beta={cfg_beta}")
                    elif method == 'dwa' and not hasattr(wrapper.model, 'set_dwa_params'):
                        raise RuntimeError("Loaded model does not expose set_dwa_params for DWA mode")
                    # Other methods could be added here if model exposes toggles
                elif forward_mode == 'dwa_adaptive' and hasattr(wrapper.model, 'set_dwa_params'):
                    wrapper.model.set_dwa_params(alpha=alpha, beta=beta, mode=prune_type)
                    print(f"[Loader] Applied DWA forward (from args): mode={prune_type}, alpha={alpha}, beta={beta}")
                elif forward_mode == 'scaling' and hasattr(wrapper.model, 'set_scaling_mode'):
                    wrapper.model.set_scaling_mode(True)
                    print("[Loader] Enabled confidence scaling mode (from args)")
                elif forward_mode == 'dpf' and hasattr(wrapper.model, 'set_dp_mode'):
                    wrapper.model.set_dp_mode(True)
                    print("[Loader] Enabled DP forward mode (from args)")

                return wrapper, loaded_config
        except Exception as e:
            # Fail fast if config is present but cannot be honored
            if config_path is not None:
                raise RuntimeError(f"Failed to load/apply config at {config_path}: {e}")
            print(f"Warning: config-aware loading failed ({e}); falling back to generic model loader.")

        # Fallback: generic model + state_dict (may be partial)
        safe_name = model_name if (model_name and model_name != 'auto') else 'resnet18'
        wrapper = BaseModel(safe_name, num_cls=num_cls, input_dim=input_dim, device=device)
        try:
            state = torch.load(model_path, map_location=device)
            # Attempt robust load via BaseModel.load for relaxed matching
            wrapper.load(model_path, verbose=True)
        except Exception as e:
            print(f"Warning: Fallback load failed ({e}); using randomly initialized weights.")
        # Provide a minimal synthetic config so result JSON is not null
        if loaded_config is None:
            loaded_config = {
                'pruning': {
                    'enabled': prune_method.lower() != 'dense',
                    'method': prune_method,
                    'sparsity': sparsity,
                    'dwa_alpha': alpha,
                    'dwa_beta': beta,
                    'dwa_mode': prune_type
                },
                'data': {'dataset': dataset_name},
                'model': {'arch': safe_name, 'layers': 18},
            }
        return wrapper, loaded_config
    
    print(f"Loading victim model (seed {args.victim_seed}) with forward_mode={args.forward_mode}...")
    victim_model, victim_cfg = load_model_from_seed_folder(
        base_path, args.victim_seed, args.dataset_name, args.model_name, 
        args.sparsity, args.alpha, args.beta, args.prune_method, args.prune_type,
        device, args.forward_mode, args.num_cls, args.input_dim, args.freeze_tag
    )
    # Also prepare a dense (unpruned) victim model for original-mode comparison
    try:
        victim_dense_model, _ = load_model_from_seed_folder(
            base_path, args.victim_seed, args.dataset_name, args.model_name,
            0.0, args.alpha, args.beta, 'dense', args.prune_type,
            device, 'standard', args.num_cls, args.input_dim, args.freeze_tag
        )
    except Exception as e:
        print(f"[WARN] Failed to load dense victim model for seed {args.victim_seed}: {e}")
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
                    logits = model.model(xb) if tv is None else model.model(xb, type_value=tv)
                except TypeError:
                    logits = model.model(xb)
                _, pred = logits.max(1)
                correct += pred.eq(yb).sum().item()
                total += yb.size(0)
        return (correct / total) if total > 0 else 0.0

    # Try a few candidates commonly used in masked conv paths
    candidate_tvs = [0, 5, 6]
    acc_scores = {}
    for tv in candidate_tvs:
        acc_scores[tv] = _sample_accuracy(victim_model, victim_train_loader, tv=tv, max_batches=2)
    best_tv = max(acc_scores, key=acc_scores.get)
    victim_model.preferred_type_value = best_tv
    print(f"[Victim] Selected type_value={best_tv} (probe acc~{acc_scores[best_tv]*100:.2f}%)")

    victim_model.test(victim_train_loader, "Victim Model Train")
    test_acc, loss = victim_model.test(victim_test_loader, "Victim Model Test")
    print(f"Victim model test accuracy: {test_acc:.3f}")

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
            for sid, sdata in (data_splits.get('shadows', {}) or {}).items():
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
    shadow_model_list = []          # pruned shadows
    shadow_dense_model_list = []    # dense shadows
    shadow_train_loader_list = []
    shadow_test_loader_list = []
    shadow_cfg_map = {}
    
    total_shadows = len(args.shadow_seeds)
    for i, shadow_seed in enumerate(args.shadow_seeds):
        if shadow_seed not in data_splits['shadows']:
            print(f"⚠️ Warning: Shadow seed {shadow_seed} not in data splits, skipping...")
            continue
            
        print(f"[{i+1}/{total_shadows}] Loading shadow model (seed {shadow_seed}) with forward_mode={args.forward_mode}...")
        shadow_model, s_cfg = load_model_from_seed_folder(
            base_path, shadow_seed, args.dataset_name, args.model_name,
            args.sparsity, args.alpha, args.beta, args.prune_method, args.prune_type,
            device, args.forward_mode, args.num_cls, args.input_dim, args.freeze_tag
        )
        # dense counterpart for original-mode comparison
        try:
            shadow_dense_model, _ = load_model_from_seed_folder(
                base_path, shadow_seed, args.dataset_name, args.model_name,
                0.0, args.alpha, args.beta, 'dense', args.prune_type,
                device, 'standard', args.num_cls, args.input_dim, args.freeze_tag
            )
        except Exception as e:
            print(f"[WARN] Failed to load dense shadow model for seed {shadow_seed}: {e}")
            shadow_dense_model = shadow_model
        shadow_cfg_map[str(shadow_seed)] = s_cfg

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
                        vs = float(victim_meta.get('sparsity', args.sparsity))
                        ss = float(s_meta.get('sparsity', -1))
                        if abs(vs - ss) > 1e-6:
                            problems.append(f"sparsity_mismatch(shadow={ss} vs victim={vs})")
                    except Exception:
                        pass
                # Method check (be lenient with 'dcil' backend)
                vm = (victim_meta.get('method') or '').lower()
                sm = (s_meta.get('method') or '').lower()
                def _norm(m):
                    return 'dense' if m in ('', None) else ('pruned' if m in ('dcil','dwa','static','dpf') else m)
                if _norm(vm) != _norm(sm):
                    problems.append(f"method_mismatch(shadow={sm} vs victim={vm})")

        if problems:
            print(f"[WARN] Skipping shadow seed {shadow_seed}: {'; '.join(problems)}")
            continue
        # Auto-tune type_value for shadow
        acc_scores = {tv: _sample_accuracy(shadow_model, victim_train_loader, tv=tv, max_batches=2) for tv in candidate_tvs}
        best_tv = max(acc_scores, key=acc_scores.get)
        shadow_model.preferred_type_value = best_tv
        print(f"[Shadow {shadow_seed}] Selected type_value={best_tv} (probe acc~{acc_scores[best_tv]*100:.2f}%)")
        
        # Use shadow data splits (training-time if available)
        if use_training_splits and i < len(training_shadow_splits):
            shadow_train_indices, shadow_test_indices = training_shadow_splits[i]
        else:
            shadow_data = data_splits['shadows'][shadow_seed]
            shadow_train_indices = shadow_data['train_indices']  # members
            shadow_test_indices  = shadow_data['test_indices']   # non-members
        
        shadow_train_dataset = Subset(total_dataset, shadow_train_indices)
        shadow_test_dataset = Subset(total_dataset, shadow_test_indices)
        
        shadow_train_loader = DataLoader(shadow_train_dataset, batch_size=args.batch_size,
                                       shuffle=False, num_workers=4, pin_memory=False)
        shadow_test_loader = DataLoader(shadow_test_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=4, pin_memory=False)
        
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

    if hasattr(args, 'original') and args.original:
        attack_original = True
    else:
        attack_original = False
    
    print(f"Attack mode: {'Original models' if attack_original else 'Pruned models'}")
    
    attacker = MiaAttack(
        victim_dense_model, victim_model, victim_train_loader, victim_test_loader,
        shadow_dense_model_list, shadow_model_list, shadow_train_loader_list, shadow_test_loader_list,
        num_cls=args.num_cls, device=device, batch_size=args.batch_size,
        attack_original=attack_original  # 🔥 이제 제대로 전달됨
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
            from sklearn.metrics import roc_auc_score, balanced_accuracy_score
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
            results['confidence_extended'] = {
                'auroc': auroc,
                'balanced_accuracy': bal_acc,
                'advantage': float(best_adv),
                'threshold': float(best_thr),
                'tpr_at_fprs': tpr_levels,
                **({'tpr_at_1fpr': tpr_at_1fpr} if tpr_at_1fpr is not None else {})
            }
            results['threshold_strategy'] = 'youden'
            tprs_msg = ", ".join([f"TPR@{k}%FPR={v:.4f}" for k, v in sorted(tpr_levels.items(), key=lambda x: float(x[0]))]) if tpr_levels else ""
            print(f"\n📊 Confidence extended metrics: AUROC={auroc:.4f}, BalAcc={bal_acc:.4f}, Adv={best_adv:.4f}, Thr={best_thr:.4f}{(' | ' + tprs_msg) if tprs_msg else ''}")
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
    if use_training_splits:
        victim_members_count = len(victim_train_indices)
        victim_nonmembers_count = len(victim_test_indices)
        shadow_counts = {str(k): len(ts[0]) for k, ts in zip(args.shadow_seeds, training_shadow_splits)}
        split_source = str(data_prepare_path)
    else:
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
                'attack_mode': 'original' if (hasattr(args, 'original') and args.original) else 'pruned',
                'dwa_params': {
                    'alpha': args.alpha,
                    'beta': args.beta,
                    'prune_type': args.prune_type,
                    'sparsity': args.sparsity
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
