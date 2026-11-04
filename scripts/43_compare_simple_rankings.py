#!/usr/bin/env python3
"""Reviewer round 2: compare optimized ranking with non-optimized rules.

Computes agreement, top-K overlap, internal score diagnostics and scoring-only
runtime on the identical frozen feature matrix. Does NOT claim independent
astronomical utility; downstream endpoints must be evaluated separately.
"""
from pathlib import Path
from time import perf_counter
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path.home() / "universe_revision" / "revision_v2"
OUT = ROOT / "outputs" / "reviewer_round2" / "simple_rules"
EPS = 1e-6

CATALOGS = {
    "gaia": {
        "matrix": ROOT / "data/processed/gaia_features_reference.csv",
        "feature_json": ROOT / "data/processed/gaia_feature_columns_reference.json",
        "id": "source_id",
        "weights": [
            ROOT / "outputs/baseline_reference/gaia/metrics/weights_gaia.csv",
            ROOT / "outputs/baseline_v2/gaia/metrics/weights_gaia.csv",
        ],
        "dominant": [
            "f_kinematic_interest", "f_photometric_color_rarity",
            "f_absolute_magnitude_rarity",
        ],
    },
    "sdss": {
        "matrix": ROOT / "data/processed/sdss_features_reference.csv",
        "feature_json": ROOT / "data/processed/sdss_feature_columns_reference.json",
        "id": "objid",
        "weights": [
            ROOT / "outputs/baseline_reference/sdss/metrics/weights_sdss.csv",
            ROOT / "outputs/baseline_v2/sdss/metrics/weights_sdss.csv",
        ],
        "dominant": [
            "f_photometric_quality", "f_redshift_precision", "f_isolation",
        ],
    },
}


def read_weights(path, features):
    tab = pd.read_csv(path)
    if not {"feature", "weight"}.issubset(tab.columns):
        raise ValueError(f"Malformed weight table: {path}")
    if tab["feature"].duplicated().any():
        raise ValueError(f"Duplicate feature weights: {path}")
    values = tab.set_index("feature")["weight"]
    if set(values.index) != set(features):
        raise ValueError(f"Weight feature mismatch: {path}")
    w = values.reindex(features).to_numpy(dtype=float)
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"Invalid weights: {path}")
    return w / w.sum()


def rank(scores, identifiers):
    # Stable object-ID tie breaker independent of CSV order.
    return np.lexsort((identifiers, -scores))


def run_catalog(catalog, top_k, override):
    cfg = CATALOGS[catalog]
    with open(cfg["feature_json"], encoding="utf-8") as fh:
        features = json.load(fh)
    weight_file = Path(override) if override else next(
        (p for p in cfg["weights"] if p.exists()), None
    )
    if weight_file is None:
        raise FileNotFoundError(
            f"No weights for {catalog}; supply --weights-{catalog}"
        )
    w_opt = read_weights(weight_file, features)
    data = pd.read_csv(cfg["matrix"], dtype={cfg["id"]: "string"})
    if data[cfg["id"]].isna().any() or data[cfg["id"]].duplicated().any():
        raise ValueError(f"Duplicate/missing {cfg['id']}")
    ids = data[cfg["id"]].astype(str).to_numpy()
    raw = data[features].to_numpy(dtype=float)
    if not np.isfinite(raw).all() or raw.min() < -1e-8 or raw.max() > 1 + 1e-8:
        raise ValueError("Features must be finite and within [0,1]")
    x = np.clip(raw, EPS, 1.0)
    logs = np.log(x)
    n, d = x.shape
    k = min(n, top_k)
    uniform = np.ones(d) / d
    fixed = np.array([float(f in cfg["dominant"]) for f in features])
    fixed /= fixed.sum()
    methods = {
        "optimized_geometric": ("geometric", w_opt),
        "equal_weight_geometric": ("geometric", uniform),
        "equal_weight_arithmetic": ("arithmetic", uniform),
        "fixed_three_geometric": ("geometric", fixed),
        "fixed_three_arithmetic": ("arithmetic", fixed),
    }
    for j, feature in enumerate(features):
        w = np.zeros(d)
        w[j] = 1.0
        methods["single_" + feature] = ("single_feature", w)

    scored = {}
    for method, (family, weights) in methods.items():
        def compute():
            if family == "geometric":
                return np.exp(logs @ weights)
            return x @ weights
        values = compute()
        samples = []
        for _ in range(5):
            t0 = perf_counter()
            compute()
            samples.append(1000 * (perf_counter() - t0))
        scored[method] = {
            "scores": values,
            "order": rank(values, ids),
            "scoring_ms": float(np.median(samples)),
            "family": family,
        }

    baseline = scored["optimized_geometric"]
    selected_baseline = set(ids[baseline["order"][:k]])
    reference_ranks = pd.Series(baseline["scores"]).rank(method="average")
    rows, selections = [], []
    for name, item in scored.items():
        idx = item["order"][:k]
        overlap = len(selected_baseline.intersection(ids[idx]))
        corr = spearmanr(
            reference_ranks,
            pd.Series(item["scores"]).rank(method="average"),
        ).statistic
        rows.append({
            "catalog": catalog,
            "method": name,
            "scoring_formula": item["family"],
            "n_targets": n,
            "top_k": k,
            "spearman_vs_optimized": float(corr),
            "top_k_overlap_count": overlap,
            "top_k_overlap_fraction": overlap / k,
            "top_k_jaccard": overlap / (2 * k - overlap),
            "top_k_mean_own_score": float(item["scores"][idx].mean()),
            "top_k_mean_common_optimized_score_INTERNAL_ONLY": float(
                baseline["scores"][idx].mean()
            ),
            "median_scoring_time_ms_5_repeats": item["scoring_ms"],
        })
        for r, pos in enumerate(idx, 1):
            selections.append({
                "catalog": catalog, "method": name, "rank": r,
                "object_id": ids[pos], "score": float(item["scores"][pos]),
            })

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{catalog}_simple_rules_metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(
        OUT / f"{catalog}_simple_rules_topk.csv", index=False
    )
    pd.DataFrame([
        {
            "catalog": catalog, "feature": feature,
            "optimized_weight": w_opt[i],
            "equal_weight": uniform[i], "fixed_three_weight": fixed[i],
        }
        for i, feature in enumerate(features)
    ]).to_csv(OUT / f"{catalog}_simple_rules_weights.csv", index=False)
    print(f"\\n{catalog.upper()} | n={n:,} | k={k:,}")
    print(f"Feature source: {cfg['matrix']}")
    print(f"Optimized weights: {weight_file}")
    print(df[[
        "method", "spearman_vs_optimized",
        "top_k_overlap_fraction", "top_k_jaccard",
        "median_scoring_time_ms_5_repeats",
    ]].to_string(index=False), flush=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", choices=["gaia", "sdss", "both"], default="both")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--weights-gaia", type=Path)
    parser.add_argument("--weights-sdss", type=Path)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    catalogs = ["gaia", "sdss"] if args.catalog == "both" else [args.catalog]
    tables = []
    for cat in catalogs:
        override = args.weights_gaia if cat == "gaia" else args.weights_sdss
        tables.append(run_catalog(cat, args.top_k, override))
    pd.concat(tables, ignore_index=True).to_csv(
        OUT / "simple_rules_all_metrics.csv", index=False
    )
    (OUT / "README.txt").write_text(
        "Reviewer round-2 descriptive simple-ranking comparators.\\n"
        "All methods operate on the same frozen feature matrix per catalog.\\n"
        "Ranking agreement, top-K overlap and common-optimized-score diagnostics\\n"
        "are INTERNAL comparisons, not independent downstream utility.\\n"
        "Timing measures scoring alone (five repetitions), excluding data acquisition,\\n"
        "feature engineering and evolutionary weight fitting.\\n"
        "For an astronomical-use claim, add an independent outcome endpoint and\\n"
        "full-pipeline computational cost in a separate experiment.\\n",
        encoding="utf-8",
    )
    print("\\nSaved to:", OUT, flush=True)


if __name__ == "__main__":
    main()
