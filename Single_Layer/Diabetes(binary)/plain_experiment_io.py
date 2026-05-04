"""Utilities for Quad4FHE plaintext experiments.

This module is intentionally small and dependency-light so it can be copied next
into any single-dataset notebook/script. It provides two things:

1. A tee context manager that writes all notebook/script stdout/stderr to a log
   file while still showing it on screen.
2. Direct JSON export from quad4fhe.ReplacementResult objects, avoiding fragile
   regex parsing of copied notebook output.
"""

from __future__ import annotations

import contextlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@contextlib.contextmanager
def tee_stdout_stderr(log_path: str | Path):
    """Duplicate stdout/stderr to ``log_path`` and the current console.

    Use around the whole experiment:

        with tee_stdout_stderr("results/my_run.txt"):
            main()
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8", buffering=1) as fh:
        sys.stdout = _TeeStream(old_stdout, fh)
        sys.stderr = _TeeStream(old_stderr, fh)
        try:
            print(f"[autosave] stdout/stderr log -> {log_path}")
            yield log_path
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def to_jsonable(obj: Any) -> Any:
    """Convert common scientific Python objects to strict JSON values."""
    # Local imports keep this helper usable even in minimal environments.
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    try:
        import pandas as pd
    except Exception:  # pragma: no cover
        pd = None
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None

    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            value = float(obj)
            return value if math.isfinite(value) else None
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return to_jsonable(obj.tolist())

    if torch is not None and hasattr(torch, "is_tensor") and torch.is_tensor(obj):
        return to_jsonable(obj.detach().cpu().numpy())

    if isinstance(obj, Path):
        return str(obj)

    if pd is not None:
        if isinstance(obj, pd.DataFrame):
            return to_jsonable(obj.to_dict(orient="records"))
        if isinstance(obj, pd.Series):
            return to_jsonable(obj.to_dict())

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    # Last resort: preserve readable values without breaking json.dump.
    return str(obj)


def save_json(obj: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(obj), fh, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"[autosave] JSON -> {path}")
    return path


def save_csv(rows: Iterable[Dict[str, Any]], path: str | Path) -> Path:
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    print(f"[autosave] CSV -> {path}")
    return path


def dataframe_records(df: Any) -> list:
    if df is None:
        return []
    try:
        return to_jsonable(df.to_dict(orient="records"))
    except Exception:
        return []


def dataframe_test_by_method(df: Any) -> Dict[str, Dict[str, Any]]:
    """Return {method: metrics} for rows with Split == 'test'."""
    out: Dict[str, Dict[str, Any]] = {}
    if df is None:
        return out
    try:
        sub = df[df["Split"] == "test"]
        for _, row in sub.iterrows():
            method = str(row.get("Method"))
            metrics = {str(k): to_jsonable(v) for k, v in row.to_dict().items()
                       if k not in ("Method", "Split")}
            out[method] = metrics
    except Exception:
        return out
    return out


def _metric_from_table(table: Dict[str, Dict[str, Any]], method: str, *keys: str) -> Any:
    row = table.get(method, {})
    for key in keys:
        if key in row:
            return row[key]
    return None


def quad_report_diagnostics(result: Any, split: str, n_expected: Optional[int] = None) -> Dict[str, Any]:
    """Extract agreement/mismatch diagnostics for fit/calibration or test split."""
    attr_candidates = []
    if split in ("fit", "calib", "calibration"):
        attr_candidates = ["fit_diagnostics", "calib_diagnostics", "calibration_diagnostics"]
        split_label = "fit"
    else:
        attr_candidates = ["test_diagnostics"]
        split_label = "test"

    diag = None
    for attr in attr_candidates:
        value = getattr(result, attr, None)
        if value is not None:
            diag = dict(value)
            break

    if diag is None:
        diag = {}
        df = getattr(result, "report_vs_pseudo", None)
        if df is not None:
            try:
                row = df[(df["Method"] == "Quad4FHE") & (df["Split"] == split_label)]
                if len(row) > 0:
                    diag["agreement"] = float(row.iloc[0]["Agreement"])
            except Exception:
                pass

    n_value = diag.get("n", diag.get("calib_n", diag.get("n_samples", n_expected)))
    agreement = diag.get("agreement", diag.get("calib_agreement", diag.get("fit_agreement")))
    mismatch = diag.get("mismatch_count", diag.get("calib_mismatch_count", diag.get("fit_mismatch_count")))

    if mismatch is None and agreement is not None and n_value is not None:
        mismatch = int(round((1.0 - float(agreement)) * int(n_value)))

    exact = diag.get("exact_preserved", diag.get("exact_preserved_on_calib", diag.get("fit_exact_preserved")))
    if exact is None and mismatch is not None:
        exact = bool(int(mismatch) == 0)

    return {
        "split": split_label,
        "n": to_jsonable(n_value),
        "agreement": to_jsonable(agreement),
        "mismatch_count": to_jsonable(mismatch),
        "exact_preserved": to_jsonable(exact),
    }


def quad_solver_diagnostics(result: Any) -> Dict[str, Any]:
    return to_jsonable(dict(getattr(result, "solver_diagnostics", None) or {}))


def build_quad4fhe_run_record(
    *,
    result: Any,
    key: str,
    hidden_dim: Optional[int],
    fit_n: Optional[int],
    test_n: Optional[int],
    pool_fraction: Optional[float] = None,
    rep_fit_size: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable record for one Quad4FHE run."""
    fit_diag = quad_report_diagnostics(result, "fit", n_expected=fit_n)
    test_diag = quad_report_diagnostics(result, "test", n_expected=test_n)
    solver_diag = quad_solver_diagnostics(result)

    report_truth_test = dataframe_test_by_method(getattr(result, "report_vs_truth", None))
    report_pseudo_test = dataframe_test_by_method(getattr(result, "report_vs_pseudo", None))

    calib_agreement = fit_diag.get("agreement")
    test_agreement = test_diag.get("agreement")
    gap = None
    if calib_agreement is not None and test_agreement is not None:
        gap = float(calib_agreement) - float(test_agreement)

    # Common keys requested by reviewers. Keep all solver diagnostics too.
    common_solver_keys = [
        "num_pairwise_constraints",
        "min_pairwise_margin",
        "normalized_min_pairwise_margin",
        "slack_positive_count",
        "sum_slack",
        "mean_slack",
        "max_slack",
        "pairwise_slack_positive_count",
        "pairwise_sum_slack",
        "pairwise_mean_slack",
        "pairwise_max_slack",
        "selected_C",
        "soft_C_grid",
        "soft_trace",
        "selected_mu",
        "mu_grid",
        "mu_p",
        "mu_n",
    ]

    quad = {
        "alpha": getattr(result, "alpha", None),
        "beta": getattr(result, "beta", None),
        "eta": getattr(result, "eta", None),
        "threshold": getattr(result, "threshold", None),
        "zero_threshold_realized": getattr(result, "zero_threshold_realized", None),
        "method_used": getattr(result, "method_used", None),
        "hard_feasible": getattr(result, "feasible", None),
        "empirical_margin": getattr(result, "empirical_margin", None),
        "normalized_margin": getattr(result, "normalized_margin", None),
        "quant_decimals": getattr(result, "quant_decimals", None),
        "constraint_version": getattr(result, "constraint_version", None),
        "he_artifact_dir": str(getattr(result, "he_export_dir", None)) if getattr(result, "he_export_dir", None) else None,
        "calib_n": fit_diag.get("n"),
        "calib_agreement": calib_agreement,
        "calib_mismatch_count": fit_diag.get("mismatch_count"),
        "exact_preserved_on_calib": fit_diag.get("exact_preserved"),
        "test_n": test_diag.get("n"),
        "test_agreement": test_agreement,
        "test_mismatch_count": test_diag.get("mismatch_count"),
        "calib_test_agreement_gap": gap,
        "test_top1_acc": _metric_from_table(report_truth_test, "Quad4FHE", "ACC", "Top1", "Top-1"),
        "test_top5_acc": _metric_from_table(report_truth_test, "Quad4FHE", "Top5", "Top-5"),
        "test_macro_f1": _metric_from_table(report_truth_test, "Quad4FHE", "MacroF1", "F1"),
        "solver_diagnostics": solver_diag,
    }
    for k in common_solver_keys:
        quad[k] = solver_diag.get(k)

    original = {
        "test_top1_acc": _metric_from_table(report_truth_test, "Original", "ACC", "Top1", "Top-1"),
        "test_top5_acc": _metric_from_table(report_truth_test, "Original", "Top5", "Top-5"),
    }

    record = {
        "key": key,
        "hidden_dim": hidden_dim,
        "pool_fraction": pool_fraction,
        "rep_fit_size": rep_fit_size,
        "original": original,
        "quad4fhe": quad,
        "report_vs_ground_truth_test": report_truth_test,
        "report_vs_pseudo_labels_test": report_pseudo_test,
        "report_vs_ground_truth_records": dataframe_records(getattr(result, "report_vs_truth", None)),
        "report_vs_pseudo_labels_records": dataframe_records(getattr(result, "report_vs_pseudo", None)),
    }
    if extra:
        record.update(to_jsonable(extra))
    return to_jsonable(record)


def make_experiment_payload(
    *,
    dataset: str,
    experiment: str,
    seed: int,
    dataset_info: Dict[str, Any],
    config: Dict[str, Any],
    source_script: Optional[str] = None,
    log_file: Optional[str | Path] = None,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "experiment": experiment,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_script": source_script,
        "log_file": str(log_file) if log_file is not None else None,
        "seed": seed,
        "dataset_info": to_jsonable(dataset_info),
        "config": to_jsonable(config),
        "runs": [],
    }


def write_combined_dataset_json(dataset: str, root_dir: str | Path = "results",
                                output_path: Optional[str | Path] = None) -> Optional[Path]:
    """Merge autosaved fulltrain/smallpool JSONs into one dataset-level JSON.

    This keeps compatibility with the older `<dataset>_results.json` convention.
    Missing halves are skipped, so it is safe to call after either notebook.
    """
    root = Path(root_dir) / dataset
    if output_path is None:
        output_path = root / f"{dataset}_results.json"
    output_path = Path(output_path)

    combined: Dict[str, Any] = {"dataset": dataset, "created_at": datetime.now().isoformat(timespec="seconds")}
    found = False
    for exp in ("fulltrain", "smallpool"):
        p = root / exp / f"{dataset}_{exp}_results.json"
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as fh:
            block = json.load(fh)
        combined[exp] = {
            "source_json": str(p),
            "source_script": block.get("source_script"),
            "log_file": block.get("log_file"),
            "dataset_info": block.get("dataset_info", {}),
            "config": block.get("config", {}),
            "runs": block.get("runs", []),
        }
        if exp == "fulltrain":
            combined[exp]["cross_hidden_dim_summary"] = block.get("cross_hidden_dim_summary", [])
        if exp == "smallpool":
            combined[exp]["cross_pool_tables"] = block.get("cross_pool_tables", {})
            combined[exp]["meta_table"] = block.get("meta_table", [])
        found = True

    if not found:
        print(f"[autosave] no fulltrain/smallpool JSON found under {root}; combined JSON not written")
        return None
    return save_json(combined, output_path)
