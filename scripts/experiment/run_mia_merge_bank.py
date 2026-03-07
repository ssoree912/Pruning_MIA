#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _clone_base_config(base_cfg: Dict[str, Any], *, dataset_name: str, seed_id: int) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    cfg.setdefault("seed", int(seed_id))
    cfg["seed"] = int(seed_id)
    cfg.setdefault("data", {})
    cfg["data"]["dataset"] = dataset_name
    cfg.setdefault("pruning", {})
    cfg["pruning"]["enabled"] = False
    cfg["pruning"]["method"] = "dense"
    cfg["pruning"]["sparsity"] = 0.0
    return cfg


def _norm_list(xs: Sequence[int]) -> List[int]:
    return [int(x) for x in xs]


def _overlap(a: Sequence[int], b: Sequence[int]) -> List[int]:
    sa = set(_norm_list(a))
    sb = set(_norm_list(b))
    return sorted(sa.intersection(sb))


def _extract_metrics(payload: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    res = payload.get("results", {}) if isinstance(payload, dict) else {}
    if not isinstance(res, dict):
        return out

    thr_block = None
    threshold_attacks = res.get("threshold_attacks", {})
    if isinstance(threshold_attacks, dict) and isinstance(threshold_attacks.get("confidence"), dict):
        thr_block = threshold_attacks.get("confidence")
    elif isinstance(res.get("confidence_extended", {}), dict):
        thr_block = res.get("confidence_extended")

    if isinstance(thr_block, dict):
        if "auc" in thr_block:
            out["threshold_auc"] = float(thr_block["auc"])
        elif "auroc" in thr_block:
            out["threshold_auc"] = float(thr_block["auroc"])
        if "advantage" in thr_block:
            out["threshold_advantage"] = float(thr_block["advantage"])
        if "tpr_at_1fpr" in thr_block:
            out["threshold_tpr_at_1fpr"] = float(thr_block["tpr_at_1fpr"])

    for key in ("lira", "nn", "samia", "nn_top3", "nn_cls"):
        block = res.get(key, {})
        if isinstance(block, dict):
            if "auc" in block:
                out[f"{key}_auc"] = float(block["auc"])
            if "advantage" in block:
                out[f"{key}_advantage"] = float(block["advantage"])
            if "tpr_at_1fpr" in block:
                out[f"{key}_tpr_at_1fpr"] = float(block["tpr_at_1fpr"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MIA on a victim model with a matched shadow bank.")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--plan-json", type=str, required=True, help="Path to a victim/shadow bank plan JSON.")
    parser.add_argument("--min-shadows", type=int, default=4)
    parser.add_argument("--allow-pipeline-mismatch", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    plan_path = Path(args.plan_json).expanduser().resolve()

    plan = _load_json(plan_path)
    dataset_name = str(plan["dataset_name"])
    base_config_path = Path(plan["base_config_path"]).expanduser().resolve()
    _ensure_exists(base_config_path, "base_config_path")
    base_cfg = _load_json(base_config_path)

    victim = plan["victim"]
    shadows = list(plan["shadows"])
    if len(shadows) < int(args.min_shadows):
        raise ValueError(f"Need at least {args.min_shadows} shadows, got {len(shadows)}")

    victim_name = str(victim["name"])
    victim_pipeline = str(victim["pipeline"])
    victim_ckpt = Path(victim["ckpt_path"]).expanduser().resolve()
    victim_model_id = int(victim["model_id"])
    victim_source_seeds = _norm_list(victim["source_seeds"])
    _ensure_exists(victim_ckpt, "victim ckpt")

    result_root = Path(plan.get("result_root", repo_root / "runs" / "mia_merge_bank")).expanduser().resolve()
    run_dir = result_root / dataset_name / victim_name
    run_dir.mkdir(parents=True, exist_ok=True)
    result_file = run_dir / "result.json"

    if args.skip_if_exists and result_file.exists():
        print(f"[skip] result exists: {result_file}")
        return

    victim_cfg = _clone_base_config(base_cfg, dataset_name=dataset_name, seed_id=victim_model_id)
    victim_cfg_path = run_dir / "configs" / f"victim_seed{victim_model_id}.json"
    _write_json(victim_cfg_path, victim_cfg)

    shadow_model_ids: List[int] = []
    shadow_ckpts: List[str] = []
    shadow_cfgs: List[str] = []
    shadow_meta: List[Dict[str, Any]] = []

    seen_ids = set()
    for sh in shadows:
        sh_name = str(sh["name"])
        sh_pipeline = str(sh["pipeline"])
        sh_ckpt = Path(sh["ckpt_path"]).expanduser().resolve()
        sh_model_id = int(sh["model_id"])
        sh_source_seeds = _norm_list(sh["source_seeds"])
        _ensure_exists(sh_ckpt, f"shadow ckpt {sh_name}")

        if sh_model_id in seen_ids:
            raise ValueError(f"Duplicate shadow model_id: {sh_model_id}")
        seen_ids.add(sh_model_id)

        ov = _overlap(victim_source_seeds, sh_source_seeds)
        if ov:
            raise ValueError(
                f"Shadow {sh_name} overlaps victim source_seeds. "
                f"victim={victim_source_seeds}, shadow={sh_source_seeds}, overlap={ov}"
            )

        if (not args.allow_pipeline_mismatch) and sh_pipeline != victim_pipeline:
            raise ValueError(
                f"Pipeline mismatch: victim={victim_pipeline}, shadow={sh_pipeline} for shadow={sh_name}"
            )

        sh_cfg = _clone_base_config(base_cfg, dataset_name=dataset_name, seed_id=sh_model_id)
        sh_cfg_path = run_dir / "configs" / f"shadow_seed{sh_model_id}.json"
        _write_json(sh_cfg_path, sh_cfg)

        shadow_model_ids.append(sh_model_id)
        shadow_ckpts.append(str(sh_ckpt))
        shadow_cfgs.append(str(sh_cfg_path))
        shadow_meta.append(
            {
                "name": sh_name,
                "pipeline": sh_pipeline,
                "model_id": sh_model_id,
                "source_seeds": sh_source_seeds,
                "ckpt_path": str(sh_ckpt),
                "config_path": str(sh_cfg_path),
            }
        )

    mia_script = repo_root / "mia_eval" / "core" / "mia_modi.py"
    _ensure_exists(mia_script, "mia_modi.py")

    cmd = [
        sys.executable,
        str(mia_script),
        "--device", str(plan.get("device", 0)),
        "--dataset_name", dataset_name,
        "--seed", str(plan.get("split_seed", 7)),
        "--victim_seed", str(victim_model_id),
        "--shadow_seeds", *[str(x) for x in shadow_model_ids],
        "--victim_ckpt_path", str(victim_ckpt),
        "--victim_config_path", str(victim_cfg_path),
        "--shadow_ckpt_paths", *shadow_ckpts,
        "--shadow_config_paths", *shadow_cfgs,
        "--attacks", str(plan.get("attacks", "threshold,lira,nn,samia")),
        "--forward_mode", str(plan.get("forward_mode", "standard")),
        "--tpr_fprs", str(plan.get("tpr_fprs", "0.1,1,5")),
        "--result_file", str(result_file),
    ]
    if bool(plan.get("save_scores", True)):
        cmd.append("--save_scores")
    if bool(plan.get("debug", False)):
        cmd.append("--debug")

    meta = {
        "dataset_name": dataset_name,
        "base_config_path": str(base_config_path),
        "victim": {
            "name": victim_name,
            "pipeline": victim_pipeline,
            "model_id": victim_model_id,
            "source_seeds": victim_source_seeds,
            "ckpt_path": str(victim_ckpt),
            "config_path": str(victim_cfg_path),
        },
        "shadows": shadow_meta,
        "command": cmd,
    }
    _write_json(run_dir / "plan.expanded.json", meta)

    print("[MIA-BANK] victim:", victim_name, "pipeline:", victim_pipeline, "shadow_count:", len(shadows))
    print("[MIA-BANK] result:", result_file)
    if args.dry_run:
        print("[MIA-BANK] dry-run only")
        return

    subprocess.run(cmd, cwd=str(repo_root), check=True)

    payload = _load_json(result_file)
    summary = {
        "result_file": str(result_file),
        "victim": meta["victim"],
        "shadow_count": len(shadow_meta),
        "metrics": _extract_metrics(payload),
    }
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
