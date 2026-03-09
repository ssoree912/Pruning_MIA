#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _to_ratio_acc(x: Any) -> Optional[float]:
    v = _as_float(x)
    if v is None:
        return None
    # Some MIA result files store test accuracy as percentage (e.g., 90.9).
    if v > 1.5:
        return v / 100.0
    return v


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _seed_group_from_values(source_seeds: Any = None, run_name: Optional[str] = None) -> str:
    nums: List[int] = []
    if source_seeds is not None:
        nums.extend(int(x) for x in re.findall(r"\d+", str(source_seeds)))
    if not nums and run_name:
        nums.extend(int(x) for x in re.findall(r"\d+", str(run_name)))
    if not nums:
        return ""
    uniq = sorted(set(nums))
    return "_".join(str(x) for x in uniq)


def _seed_sort_key(seed_group: Any) -> Tuple[int, ...]:
    vals = [int(x) for x in re.findall(r"\d+", str(seed_group or ""))]
    if not vals:
        return (10**9,)
    return tuple(vals)


def _method_order(method: Any) -> int:
    m = str(method or "")
    if m == "dense" or m.startswith("dense_") or m == "pipeline_dense":
        return -1
    if m == "scratch_retrain_baseline":
        return 0
    if m.startswith("unlearn_seed") or m == "unlearn_victim":
        return 1
    if m == "raw_unlearn":
        return 2
    if m.startswith("merge_"):
        merge_order = {
            "merge_bezier": 3,
            "merge_bezier_swa": 4,
            "merge_simplex": 5,
            "merge_simplex_soup": 6,
        }
        return merge_order.get(m, 7)
    order = {
        "raw_linear_best": 10,
        "perm_linear_best": 11,
        "bezier_best": 12,
        "bezier_swa_best": 13,
        "simplex_best": 14,
        "simplex_soup_best": 15,
        "simplex_swa_best": 16,
    }
    return order.get(m, 100)


def _sort_mia_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            _seed_sort_key(r.get("seed_group")),
            str(r.get("run_name", "")),
            str(r.get("attack", "")),
        ),
    )


def _sort_perf_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            _seed_sort_key(r.get("seed_group")),
            str(r.get("run_name", "")),
            str(r.get("source_type", "")),
            _method_order(r.get("method")),
            str(r.get("method", "")),
        ),
    )


def _has_glob_magic(s: str) -> bool:
    return any(ch in s for ch in "*?[]")


def _collect_from_dir(root: Path, kind: str) -> List[Path]:
    if kind == "mia":
        return [p.resolve() for p in root.rglob("result.json") if p.is_file()]
    if kind == "perf":
        # include summary.json + summary_seedXX.json etc.
        return [p.resolve() for p in root.rglob("summary*.json") if p.is_file()]
    raise ValueError(f"Unsupported kind: {kind}")


def _resolve_input_paths(inputs: Sequence[str], kind: str) -> List[Path]:
    out: List[Path] = []
    for raw in inputs:
        expanded = str(Path(raw).expanduser())
        matches: List[Path]

        if _has_glob_magic(expanded):
            matches = [Path(x) for x in glob.glob(expanded, recursive=True)]
        else:
            matches = [Path(expanded)]

        for m in matches:
            if m.is_dir():
                out.extend(_collect_from_dir(m, kind))
            elif m.is_file():
                out.append(m.resolve())

    return sorted(set(out))


def _is_aux_summary(path: Path) -> bool:
    # summary_seed42.json 같은 보조 파일은 같은 폴더의 summary.json과 중복일 수 있음.
    name = path.name
    if not name.startswith("summary_") or not name.endswith(".json"):
        return False
    canonical = path.parent / "summary.json"
    if not canonical.exists():
        return False
    # canonical summary가 connectivity/unlearn endpoint 정보를 담고 있을 때만
    # 보조 파일을 skip. retrain의 summary_seedXX.json은 보조가 아니라
    # 실제 per-seed 결과일 수 있으므로 여기서는 skip하지 않는다.
    try:
        payload = _load_json(canonical)
    except Exception:
        return False
    if isinstance(payload.get("endpoint_metrics"), dict) and payload.get("endpoint_metrics"):
        return True
    if isinstance(payload.get("endpoints"), dict) and payload.get("endpoints"):
        return True
    return False


def _auto_mia_results() -> List[Path]:
    return sorted(Path("runs").glob("mia_merge_bank/**/result.json"))


def _auto_perf_summaries() -> List[Path]:
    pats = [
        "runs/unlearning_connectivity_phase15*/**/summary.json",
        "runs/unlearning_df1/unlearn/**/summary.json",
        "runs/unlearning_df1/retrain/**/summary*.json",
        "runs/dense/**/summary_seed*.json",
    ]
    return _resolve_input_paths(pats, kind="perf")


def _attack_row(base: Dict[str, Any], attack_name: str, block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **base,
        "attack": attack_name,
        "accuracy": _as_float(block.get("accuracy")),
        "auc": _as_float(block.get("auc", block.get("auroc"))),
        "balanced_accuracy": _as_float(block.get("balanced_accuracy")),
        "advantage": _as_float(block.get("advantage")),
        "tpr_at_1fpr": _as_float(block.get("tpr_at_1fpr")),
        "implementation": block.get("implementation"),
        "per_example_reference_models": block.get("per_example_reference_models"),
    }


def _load_plan_expanded(result_path: Path) -> Dict[str, Any]:
    plan_path = result_path.parent / "plan.expanded.json"
    if not plan_path.exists():
        return {}
    try:
        return _load_json(plan_path)
    except Exception:
        return {}


def extract_mia_attack_rows(result_path: Path, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = payload.get("results", {}) if isinstance(payload, dict) else {}
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    plan_expanded = _load_plan_expanded(result_path)
    victim_meta = plan_expanded.get("victim", {}) if isinstance(plan_expanded, dict) else {}
    shadows_meta = plan_expanded.get("shadows", []) if isinstance(plan_expanded, dict) else []

    victim_seed = payload.get("victim_seed", config.get("victim_seed"))
    shadow_seeds = payload.get("shadow_seeds", config.get("shadow_seeds"))
    if isinstance(shadow_seeds, list):
        shadow_seeds_str = ",".join(str(s) for s in shadow_seeds)
    else:
        shadow_seeds_str = str(shadow_seeds) if shadow_seeds is not None else None

    base = {
        "result_file": str(result_path),
        "run_name": result_path.parent.name,
        "seed_group": _seed_group_from_values(victim_meta.get("source_seeds"), result_path.parent.name),
        "dataset": payload.get("dataset_name", config.get("dataset_name")),
        "victim_seed": victim_seed,
        "shadow_seeds": shadow_seeds_str,
        "victim_name": victim_meta.get("name"),
        "pipeline": victim_meta.get("pipeline"),
        "source_seeds": (
            ",".join(str(s) for s in victim_meta.get("source_seeds", []))
            if isinstance(victim_meta.get("source_seeds"), list)
            else None
        ),
        "shadow_count": len(shadows_meta) if isinstance(shadows_meta, list) else None,
        "victim_test_acc": _to_ratio_acc(payload.get("victim_test_acc")),
    }

    rows: List[Dict[str, Any]] = []

    threshold_attacks = results.get("threshold_attacks", {})
    if isinstance(threshold_attacks, dict):
        for thr_name, thr_block in threshold_attacks.items():
            if isinstance(thr_block, dict):
                rows.append(_attack_row(base, f"threshold_{thr_name}", thr_block))
    elif isinstance(results.get("confidence_extended"), dict):
        rows.append(_attack_row(base, "threshold_confidence", results["confidence_extended"]))

    for atk in ("lira", "nn", "nn_top3", "nn_cls", "samia"):
        block = results.get(atk)
        if isinstance(block, dict):
            rows.append(_attack_row(base, atk, block))

    return rows


def _classify_performance_row(method: str) -> Tuple[str, str]:
    m = (method or "").lower()
    if m == "scratch_retrain_baseline" or "scratch_retrain" in m:
        return "comparison", "retrain"
    if m.startswith("unlearn_seed") or m == "unlearn_victim" or "raw_unlearn" in m:
        return "comparison", "unlearn"
    if m == "dense" or m.startswith("dense_") or m == "pipeline_dense":
        return "comparison", "dense"
    if m.startswith("merge_"):
        return "comparison", "merge"
    return "candidate", ""


def _comparison_key_from_method(method: Any) -> Optional[str]:
    m = str(method or "").lower()
    if m == "dense" or m.startswith("dense_") or m == "pipeline_dense":
        return "dense"
    if m == "scratch_retrain_baseline" or "scratch_retrain" in m:
        return "retrain"
    if m.startswith("unlearn_seed") or m == "unlearn_victim" or "raw_unlearn" in m:
        return "unlearn"
    if m.startswith("merge_"):
        return m
    return None


def _json_cell(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return json.dumps(v, separators=(",", ":"))
    except Exception:
        return str(v)


def _perf_row(
    *,
    summary_path: Path,
    source_type: str,
    method: str,
    run_dir: Any,
    metrics: Dict[str, Any],
    selection_t: Optional[float] = None,
    selection_name: Optional[str] = None,
    selection_lambdas: Optional[str] = None,
    selected_candidates: Optional[str] = None,
) -> Dict[str, Any]:
    row_role, comparison_group = _classify_performance_row(method)
    return {
        "summary_file": str(summary_path),
        "source_type": source_type,
        "run_name": summary_path.parent.name,
        "seed_group": _seed_group_from_values(run_name=summary_path.parent.name),
        "run_dir": run_dir,
        "row_role": row_role,
        "comparison_group": comparison_group,
        "method": method,
        "selection_t": selection_t,
        "selection_name": selection_name,
        "selection_lambdas": selection_lambdas,
        "selected_candidates": selected_candidates,
        "test_acc": _to_ratio_acc(metrics.get("test_acc")),
        "retain_subset_test_acc": _to_ratio_acc(metrics.get("retain_test_acc")),
        "forget_subset_test_acc": _to_ratio_acc(metrics.get("forget_test_acc")),
    }


def _merge_metrics_from_connectivity_summary(
    pipeline: Optional[str],
    ckpt_path: Optional[Path],
) -> Optional[Dict[str, Any]]:
    if not pipeline or ckpt_path is None:
        return None
    summary_path = ckpt_path.parent / "summary.json"
    if not summary_path.exists():
        return None
    try:
        payload = _load_json(summary_path)
    except Exception:
        return None

    if pipeline == "merge_simplex_soup":
        block = payload.get("simplex_soup", {}).get("metrics")
        return block if isinstance(block, dict) else None
    if pipeline == "merge_bezier_swa":
        block = payload.get("bezier_swa", {}).get("best_by_selector")
        return block if isinstance(block, dict) else None
    if pipeline == "merge_simplex":
        block = payload.get("simplex", {}).get("best_sample")
        return block if isinstance(block, dict) else None
    if pipeline == "merge_bezier":
        block = payload.get("bezier", {}).get("best_by_selector")
        return block if isinstance(block, dict) else None
    return None


def _selection_meta_for_connectivity_method(
    method: str,
    payload: Dict[str, Any],
    block: Dict[str, Any],
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "selection_t": None,
        "selection_name": None,
        "selection_lambdas": None,
        "selected_candidates": None,
    }

    if method in {"raw_linear_best", "perm_linear_best", "bezier_best", "bezier_swa_best"}:
        meta["selection_t"] = _as_float(block.get("t"))
        return meta

    if method in {"simplex_best", "simplex_swa_best"}:
        meta["selection_name"] = block.get("name")
        meta["selection_lambdas"] = _json_cell(block.get("lambdas"))
        return meta

    if method == "simplex_soup_best":
        selected = payload.get("simplex_soup", {}).get("selected_candidates")
        if isinstance(selected, list) and selected:
            meta["selected_candidates"] = _json_cell(selected)
            if len(selected) == 1:
                meta["selection_name"] = str(selected[0])
                # if soup picked one candidate, try to recover lambdas from best samples.
                sb = payload.get("simplex", {}).get("best_sample", {})
                if isinstance(sb, dict) and sb.get("name") == meta["selection_name"]:
                    meta["selection_lambdas"] = _json_cell(sb.get("lambdas"))
                sswab = payload.get("simplex_swa", {}).get("best_sample", {})
                if (
                    meta["selection_lambdas"] is None
                    and isinstance(sswab, dict)
                    and sswab.get("name") == meta["selection_name"]
                ):
                    meta["selection_lambdas"] = _json_cell(sswab.get("lambdas"))
        return meta

    return meta


def _merge_selection_meta_from_connectivity_summary(
    pipeline: Optional[str],
    ckpt_path: Optional[Path],
) -> Dict[str, Any]:
    empty = {
        "selection_t": None,
        "selection_name": None,
        "selection_lambdas": None,
        "selected_candidates": None,
    }
    if not pipeline or ckpt_path is None:
        return empty

    summary_path = ckpt_path.parent / "summary.json"
    if not summary_path.exists():
        return empty

    try:
        payload = _load_json(summary_path)
    except Exception:
        return empty

    if pipeline == "merge_simplex_soup":
        block = payload.get("simplex_soup", {}).get("metrics")
        if isinstance(block, dict):
            return _selection_meta_for_connectivity_method("simplex_soup_best", payload, block)
    if pipeline == "merge_bezier_swa":
        block = payload.get("bezier_swa", {}).get("best_by_selector")
        if isinstance(block, dict):
            return _selection_meta_for_connectivity_method("bezier_swa_best", payload, block)
    if pipeline == "merge_simplex":
        block = payload.get("simplex", {}).get("best_sample")
        if isinstance(block, dict):
            return _selection_meta_for_connectivity_method("simplex_best", payload, block)
    if pipeline == "merge_bezier":
        block = payload.get("bezier", {}).get("best_by_selector")
        if isinstance(block, dict):
            return _selection_meta_for_connectivity_method("bezier_best", payload, block)
    return empty


def _nonmerge_metrics_from_train_summary(
    pipeline: Optional[str],
    ckpt_path: Optional[Path],
    model_id: Optional[int],
) -> Optional[Dict[str, Any]]:
    if pipeline not in {"scratch_retrain", "raw_unlearn"} or ckpt_path is None:
        return None
    run_dir = ckpt_path.parent
    candidate_summaries: List[Path] = []
    if model_id is not None:
        candidate_summaries.append(run_dir / f"summary_seed{int(model_id)}.json")
    candidate_summaries.append(run_dir / "summary.json")

    for sp in candidate_summaries:
        if not sp.exists():
            continue
        try:
            payload = _load_json(sp)
        except Exception:
            continue

        # scratch retrain baseline metrics
        scratch = payload.get("scratch_retrain_baseline", {})
        if isinstance(scratch, dict):
            m = scratch.get("metrics")
            if isinstance(m, dict) and m:
                return m

        # endpoint metrics (raw unlearn victim)
        endpoints = payload.get("endpoints", {})
        if isinstance(endpoints, dict):
            endpoint_map = endpoints.get("metrics", {})
            if isinstance(endpoint_map, dict):
                key = f"seed{int(model_id)}" if model_id is not None else None
                if key and isinstance(endpoint_map.get(key), dict):
                    return endpoint_map[key]
    return None


def extract_perf_row_from_mia_result(result_path: Path, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # run_mia_merge_bank.py always writes plan.expanded.json next to result.json
    plan_path = result_path.parent / "plan.expanded.json"
    plan = _load_json(plan_path) if plan_path.exists() else {}
    victim = plan.get("victim", {}) if isinstance(plan, dict) else {}

    pipeline = victim.get("pipeline")
    method = str(pipeline) if pipeline else "from_mia_result"
    run_dir = result_path.parent
    model_id = victim.get("model_id") if isinstance(victim, dict) else None

    test_acc = _to_ratio_acc(payload.get("victim_test_acc"))
    retain_test_acc = None
    forget_test_acc = None

    ckpt_path = None
    if isinstance(victim, dict) and victim.get("ckpt_path"):
        ckpt_path = Path(str(victim["ckpt_path"]))
    selection_meta = _merge_selection_meta_from_connectivity_summary(
        pipeline=str(pipeline) if pipeline is not None else None,
        ckpt_path=ckpt_path,
    )

    # For merged victims, try to recover richer metrics from connectivity summary.
    merged_metrics = _merge_metrics_from_connectivity_summary(
        pipeline=str(pipeline) if pipeline is not None else None,
        ckpt_path=ckpt_path,
    )
    if isinstance(merged_metrics, dict):
        test_acc = _as_float(merged_metrics.get("test_acc")) or test_acc
        retain_test_acc = _as_float(merged_metrics.get("retain_test_acc"))
        forget_test_acc = _as_float(merged_metrics.get("forget_test_acc"))
    else:
        # For raw_unlearn / scratch_retrain, pull subset metrics from train summary.
        train_metrics = _nonmerge_metrics_from_train_summary(
            pipeline=str(pipeline) if pipeline is not None else None,
            ckpt_path=ckpt_path,
            model_id=int(model_id) if model_id is not None else None,
        )
        if isinstance(train_metrics, dict):
            test_acc = _to_ratio_acc(train_metrics.get("test_acc")) or test_acc
            retain_test_acc = _to_ratio_acc(train_metrics.get("retain_test_acc"))
            forget_test_acc = _to_ratio_acc(train_metrics.get("forget_test_acc"))

    return {
        "summary_file": str(result_path),
        "source_type": "mia_result_perf",
        "run_name": result_path.parent.name,
        "seed_group": _seed_group_from_values(victim.get("source_seeds"), result_path.parent.name),
        "run_dir": str(run_dir),
        "row_role": _classify_performance_row(method)[0],
        "comparison_group": _classify_performance_row(method)[1],
        "method": method,
        "selection_t": selection_meta.get("selection_t"),
        "selection_name": selection_meta.get("selection_name"),
        "selection_lambdas": selection_meta.get("selection_lambdas"),
        "selected_candidates": selection_meta.get("selected_candidates"),
        "test_acc": test_acc,
        "retain_subset_test_acc": retain_test_acc,
        "forget_subset_test_acc": forget_test_acc,
    }


def _should_skip_mia_performance_row(row: Dict[str, Any]) -> bool:
    # Keep mia_merge_bank values in mia_attacks.csv only.
    return True


def _endpoint_seed_metrics(endpoints: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    out: List[Tuple[int, Dict[str, Any]]] = []
    endpoint_map = endpoints.get("metrics", {})
    if not isinstance(endpoint_map, dict):
        return out

    seen: Set[int] = set()

    for seed_key in ("seed_a", "seed_b"):
        seed_val = endpoints.get(seed_key)
        if seed_val is None:
            continue
        seed = _first_seed_like(seed_val)
        if seed is None or seed in seen:
            continue
        metrics = endpoint_map.get(f"seed{seed}")
        if not isinstance(metrics, dict) or not metrics:
            continue
        seen.add(seed)
        out.append((seed, metrics))

    if out:
        return out

    for key, metrics in endpoint_map.items():
        if not isinstance(metrics, dict) or not metrics:
            continue
        seed = _first_seed_like(key)
        if seed is None or seed in seen:
            continue
        seen.add(seed)
        out.append((seed, metrics))
    return sorted(out, key=lambda item: item[0])


def extract_performance_rows(summary_path: Path, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    run_dir = payload.get("run_dir", str(summary_path.parent))

    # connectivity/run_connectivity_experiment.py summary
    endpoint_metrics = payload.get("endpoint_metrics")
    if isinstance(endpoint_metrics, dict):
        method_blocks: List[Tuple[str, Any]] = [
            ("raw_linear_best", payload.get("raw_linear", {}).get("best_by_selector")),
            ("perm_linear_best", payload.get("perm_linear", {}).get("best_by_selector")),
            ("bezier_best", payload.get("bezier", {}).get("best_by_selector")),
            ("bezier_swa_best", payload.get("bezier_swa", {}).get("best_by_selector")),
            ("simplex_best", payload.get("simplex", {}).get("best_sample")),
            ("simplex_soup_best", payload.get("simplex_soup", {}).get("metrics")),
            ("simplex_swa_best", payload.get("simplex_swa", {}).get("best_sample")),
        ]

        for method, block in method_blocks:
            if isinstance(block, dict) and block:
                sel_meta = _selection_meta_for_connectivity_method(method, payload, block)
                rows.append(
                    _perf_row(
                        summary_path=summary_path,
                        source_type="connectivity_summary",
                        method=method,
                        run_dir=run_dir,
                        metrics=block,
                        selection_t=sel_meta.get("selection_t"),
                        selection_name=sel_meta.get("selection_name"),
                        selection_lambdas=sel_meta.get("selection_lambdas"),
                        selected_candidates=sel_meta.get("selected_candidates"),
                    )
                )
        if rows:
            return rows

    dense_baseline = payload.get("dense_baseline")
    if isinstance(dense_baseline, dict):
        dense_metrics = dense_baseline.get("metrics", {})
        if isinstance(dense_metrics, dict) and dense_metrics:
            dense_seed = _first_seed_like(dense_baseline.get("seed"), summary_path.name, summary_path.parent.name)
            method = f"dense_seed{int(dense_seed)}" if dense_seed is not None else "dense"
            rows.append(
                _perf_row(
                    summary_path=summary_path,
                    source_type="dense_summary",
                    method=method,
                    run_dir=run_dir,
                    metrics=dense_metrics,
                )
            )
            return rows

    # train.py summary (step1/step2/step3)
    endpoints = payload.get("endpoints")
    if isinstance(endpoints, dict):
        norm_path = str(summary_path).replace("\\", "/")
        in_unlearn_dir = "/unlearn/" in norm_path
        in_retrain_dir = "/retrain/" in norm_path

        endpoint_seed_rows = _endpoint_seed_metrics(endpoints)
        scratch_metrics = payload.get("scratch_retrain_baseline", {}).get("metrics", {})

        # Clean rule:
        # - unlearn 디렉터리에서는 unlearn row만.
        # - retrain 디렉터리에서는 scratch row만.
        # - 기타 위치에서는 둘 다.
        emit_unlearn = (not in_retrain_dir) or in_unlearn_dir
        emit_retrain = (not in_unlearn_dir) or in_retrain_dir

        if emit_unlearn:
            for seed, endpoint_seed_metrics in endpoint_seed_rows:
                rows.append(
                    _perf_row(
                        summary_path=summary_path,
                        source_type="unlearning_summary",
                        method=f"unlearn_seed{seed}",
                        run_dir=run_dir,
                        metrics=endpoint_seed_metrics,
                    )
                )
        if emit_retrain and isinstance(scratch_metrics, dict) and scratch_metrics:
            rows.append(
                _perf_row(
                    summary_path=summary_path,
                    source_type="unlearning_summary",
                    method="scratch_retrain_baseline",
                    run_dir=run_dir,
                    metrics=scratch_metrics,
                )
            )
        if rows:
            return rows

    # retrain 요약 중 일부는 endpoints 없이 scratch_retrain_baseline만 있는 포맷.
    scratch = payload.get("scratch_retrain_baseline", {})
    if isinstance(scratch, dict):
        scratch_metrics = scratch.get("metrics")
        if isinstance(scratch_metrics, dict) and scratch_metrics:
            rows.append(
                _perf_row(
                    summary_path=summary_path,
                    source_type="unlearning_summary",
                    method="scratch_retrain_baseline",
                    run_dir=run_dir,
                    metrics=scratch_metrics,
                )
            )
            return rows

    return rows


def _build_seed_compare_rows(perf_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build one row per seed_group for actual MIA victims.
    Columns are dynamically created per comparison key:
      dense, retrain, unlearn, merge_*.
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    for r in perf_rows:
        if str(r.get("source_type", "")) != "mia_result_perf":
            continue
        seed_group = str(r.get("seed_group") or "")
        if seed_group == "":
            seed_group = "_unknown"
        key = _comparison_key_from_method(r.get("method"))
        if key is None:
            continue

        row = grouped.setdefault(seed_group, {"seed_group": seed_group})
        test_acc = _as_float(r.get("test_acc"))
        prev_test_acc = _as_float(row.get(f"{key}_test_acc"))
        # If duplicate key exists for same seed, keep the better one.
        if prev_test_acc is not None and test_acc is not None and prev_test_acc > test_acc:
            continue

        row[f"{key}_run_name"] = r.get("run_name")
        row[f"{key}_method"] = r.get("method")
        row[f"{key}_test_acc"] = r.get("test_acc")
        row[f"{key}_retain_subset_test_acc"] = r.get("retain_subset_test_acc")
        row[f"{key}_forget_subset_test_acc"] = r.get("forget_subset_test_acc")

    return sorted(grouped.values(), key=lambda rr: _seed_sort_key(rr.get("seed_group")))


def _write_csv(
    path: Path,
    rows: Iterable[Dict[str, Any]],
    preferred_cols: Sequence[str],
    *,
    na_value: str = "",
) -> int:
    rows_list = list(rows)
    if not rows_list:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(preferred_cols)
        return 0

    all_cols = set()
    for r in rows_list:
        all_cols.update(r.keys())

    ordered: List[str] = []
    for c in preferred_cols:
        if c in all_cols and c not in ordered:
            ordered.append(c)
    for c in sorted(all_cols):
        if c not in ordered:
            ordered.append(c)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for r in rows_list:
            row_out: Dict[str, Any] = {}
            for c in ordered:
                v = r.get(c)
                if v is None or (isinstance(v, str) and v == ""):
                    row_out[c] = na_value
                else:
                    row_out[c] = v
            writer.writerow(row_out)
    return len(rows_list)


def _dedup_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[Tuple[str, str], ...]] = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = tuple(sorted((str(k), str(v)) for k, v in r.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export MIA result.json and performance summary.json into two CSV files."
    )
    parser.add_argument(
        "--mia-results",
        nargs="*",
        default=None,
        help=(
            "Glob/file/dir list for MIA result.json. "
            "If a directory is given, result.json is searched recursively. "
            "(default: runs/mia_merge_bank/**/result.json)"
        ),
    )
    parser.add_argument(
        "--perf-summaries",
        nargs="*",
        default=None,
        help=(
            "Glob/file/dir list for performance summary json. "
            "If a directory is given, summary*.json is searched recursively. "
            "(default scans connectivity/unlearning summary locations)"
        ),
    )
    parser.add_argument("--out-dir", type=str, default="runs/csv_exports")
    parser.add_argument("--mia-csv", type=str, default="mia_attacks.csv")
    parser.add_argument("--perf-csv", type=str, default="performance.csv")
    parser.add_argument("--perf-seed-csv", type=str, default="performance_by_seed.csv")
    parser.add_argument(
        "--na-value",
        type=str,
        default="-",
        help="Value to write for empty cells (default: '-')",
    )
    parser.add_argument(
        "--include-mia-performance",
        action="store_true",
        help="Also append performance rows inferred from mia result.json (disabled by default).",
    )
    parser.add_argument("--no-dedup", action="store_true", help="Disable row-level de-duplication")
    args = parser.parse_args()

    mia_files = _auto_mia_results() if args.mia_results is None else _resolve_input_paths(args.mia_results, kind="mia")
    perf_files = _auto_perf_summaries() if args.perf_summaries is None else _resolve_input_paths(args.perf_summaries, kind="perf")

    mia_rows: List[Dict[str, Any]] = []
    for p in mia_files:
        try:
            payload = _load_json(p)
            mia_rows.extend(extract_mia_attack_rows(p, payload))
        except Exception as e:
            mia_rows.append(
                {
                    "result_file": str(p),
                    "attack": "ERROR",
                    "error": str(e),
                }
            )

    perf_rows: List[Dict[str, Any]] = []
    for p in perf_files:
        if _is_aux_summary(p):
            continue
        try:
            payload = _load_json(p)
            perf_rows.extend(extract_performance_rows(p, payload))
        except Exception as e:
            perf_rows.append(
                {
                    "summary_file": str(p),
                    "source_type": "ERROR",
                    "method": "ERROR",
                    "error": str(e),
                }
            )

    # Optional: append performance rows inferred from mia result.json.
    if args.include_mia_performance:
        for p in mia_files:
            try:
                payload = _load_json(p)
                row = extract_perf_row_from_mia_result(p, payload)
                if row is not None:
                    if _should_skip_mia_performance_row(row):
                        continue
                    perf_rows.append(row)
            except Exception as e:
                perf_rows.append(
                    {
                        "summary_file": str(p),
                        "source_type": "ERROR",
                        "method": "ERROR",
                        "error": str(e),
                    }
                )

    if not args.no_dedup:
        mia_rows = _dedup_rows(mia_rows)
        perf_rows = _dedup_rows(perf_rows)

    mia_rows = _sort_mia_rows(mia_rows)
    perf_rows = _sort_perf_rows(perf_rows)

    out_dir = Path(args.out_dir).expanduser().resolve()
    mia_csv = out_dir / args.mia_csv
    perf_csv = out_dir / args.perf_csv
    perf_seed_csv = out_dir / args.perf_seed_csv

    mia_count = _write_csv(
        mia_csv,
        mia_rows,
        preferred_cols=[
            "result_file",
            "run_name",
            "seed_group",
            "dataset",
            "victim_name",
            "pipeline",
            "victim_seed",
            "source_seeds",
            "shadow_seeds",
            "shadow_count",
            "attack",
            "victim_test_acc",
            "accuracy",
            "auc",
            "balanced_accuracy",
            "advantage",
            "tpr_at_1fpr",
            "implementation",
            "per_example_reference_models",
            "error",
        ],
        na_value=args.na_value,
    )
    perf_count = _write_csv(
        perf_csv,
        perf_rows,
        preferred_cols=[
            "summary_file",
            "source_type",
            "run_name",
            "seed_group",
            "run_dir",
            "row_role",
            "comparison_group",
            "method",
            "selection_t",
            "selection_name",
            "selection_lambdas",
            "selected_candidates",
            "test_acc",
            "retain_subset_test_acc",
            "forget_subset_test_acc",
            "error",
        ],
        na_value=args.na_value,
    )

    perf_seed_rows = _build_seed_compare_rows(perf_rows)
    perf_seed_count = _write_csv(
        perf_seed_csv,
        perf_seed_rows,
        preferred_cols=[
            "seed_group",
            "dense_run_name",
            "dense_test_acc",
            "dense_retain_subset_test_acc",
            "dense_forget_subset_test_acc",
            "retrain_run_name",
            "retrain_test_acc",
            "retrain_retain_subset_test_acc",
            "retrain_forget_subset_test_acc",
            "unlearn_run_name",
            "unlearn_test_acc",
            "unlearn_retain_subset_test_acc",
            "unlearn_forget_subset_test_acc",
            "merge_bezier_run_name",
            "merge_bezier_test_acc",
            "merge_bezier_retain_subset_test_acc",
            "merge_bezier_forget_subset_test_acc",
            "merge_bezier_swa_run_name",
            "merge_bezier_swa_test_acc",
            "merge_bezier_swa_retain_subset_test_acc",
            "merge_bezier_swa_forget_subset_test_acc",
            "merge_simplex_run_name",
            "merge_simplex_test_acc",
            "merge_simplex_retain_subset_test_acc",
            "merge_simplex_forget_subset_test_acc",
            "merge_simplex_soup_run_name",
            "merge_simplex_soup_test_acc",
            "merge_simplex_soup_retain_subset_test_acc",
            "merge_simplex_soup_forget_subset_test_acc",
        ],
        na_value=args.na_value,
    )

    print(f"[done] mia csv : {mia_csv} (rows={mia_count})")
    print(f"[done] perf csv: {perf_csv} (rows={perf_count})")
    print(f"[done] perf seed csv: {perf_seed_csv} (rows={perf_seed_count})")
    print(f"[info] mia files scanned : {len(mia_files)}")
    print(f"[info] perf files scanned: {len(perf_files)}")


if __name__ == "__main__":
    main()
