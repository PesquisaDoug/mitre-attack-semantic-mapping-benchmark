"""Compute 95% bootstrap confidence intervals for MRR@10 and nDCG@10.

Reads results/per_query_metrics.csv (produced by scripts/generate_results.py)
and writes results/bootstrap_ci.json. Uses a fixed random seed for
reproducibility. Confidence intervals for a single method resample per-query
metric values independently; improvement intervals (bi-encoder+cross-encoder
vs. bi-encoder, and bi-encoder vs. TF-IDF/BM25) resample the same bootstrap
query indices for both methods being compared (paired by query_id).

Usage:
    .venv\\Scripts\\python.exe -m scripts.bootstrap_ci
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_BOOT = 10000
METHODS = ["TF-IDF", "BM25", "MiniLM-BiEncoder", "MiniLM-BiEncoder+CrossEncoder"]
SUBSETS = ["all", "strict"]
METRICS = ["mrr@10", "ndcg@10"]
PAIRS = [
    ("MiniLM-BiEncoder+CrossEncoder", "MiniLM-BiEncoder"),
    ("MiniLM-BiEncoder", "TF-IDF"),
    ("MiniLM-BiEncoder", "BM25"),
]


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = values[idx].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def main() -> None:
    root = Path.cwd()
    results_dir = root / "results"
    df = pd.read_csv(results_dir / "per_query_metrics.csv")

    single_method_cis = []
    for subset in SUBSETS:
        sub_df = df[df["subset"] == subset]
        for metric in METRICS:
            for method in METHODS:
                vals = sub_df[sub_df["method"] == method].sort_values("query_id")[metric].values
                mean, lo, hi = bootstrap_ci(vals)
                single_method_cis.append(
                    {
                        "subset": subset,
                        "metric": metric,
                        "method": method,
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n_queries": int(len(vals)),
                    }
                )

    paired_diff_cis = []
    for subset in SUBSETS:
        sub_df = df[df["subset"] == subset]
        for metric in METRICS:
            pivot = sub_df.pivot(index="query_id", columns="method", values=metric).sort_index()
            n = len(pivot)
            rng = np.random.default_rng(SEED)
            idx = rng.integers(0, n, size=(N_BOOT, n))
            for a, b in PAIRS:
                diff = pivot[a].values - pivot[b].values
                boot_diffs = diff[idx].mean(axis=1)
                lo, hi = np.percentile(boot_diffs, [2.5, 97.5])
                paired_diff_cis.append(
                    {
                        "subset": subset,
                        "metric": metric,
                        "a": a,
                        "b": b,
                        "mean_diff": float(diff.mean()),
                        "ci_low": float(lo),
                        "ci_high": float(hi),
                        "n_queries": int(n),
                    }
                )

    payload = {
        "seed": SEED,
        "n_boot": N_BOOT,
        "paired_by_query": True,
        "single_method_cis": single_method_cis,
        "paired_diff_cis": paired_diff_cis,
    }
    (results_dir / "bootstrap_ci.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "bootstrap_ci", "seed": SEED, "n_boot": N_BOOT}, indent=2))


if __name__ == "__main__":
    main()
