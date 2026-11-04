#!/usr/bin/env python3
"""Reviewer round 2 / experiment 44: objective-function ablations.

Reoptimizes the SAME frozen reference-based feature matrices with alternate
entropy/concentration/diversity coefficients, plus a post-feature empirical-CDF
rank-normalization diagnostic.  Labels are not used to optimize weights.

Reporting notes:
- The percentile-rank case reranks the already-computed feature columns;
  it does not undo the original percentile clipping of raw measurements.
- Optimizer convergence, objective-induced structure, and external scientific
  utility are distinct questions. This script measures the first two, not
  the third. Runs with unconverged optimizers must be rerun with higher
  maxiter before scientific comparison.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

ROOT = Path.home() / "universe_revision" / "revision_v2"
OUT = ROOT / "outputs" / "reviewer_round2" / "objective_ablations"
EPS = 1e-6

CAT = {
    "gaia": {
        "matrix": ROOT / "data/processed/gaia_features_reference.csv",
        "features": ROOT / "data/processed/gaia_feature_columns_reference.json",
        "weights": ROOT / "outputs/baseline_reference/gaia/metrics/weights_gaia.csv",
        "id": "source_id",
    },
    "sdss": {
        "matrix": ROOT / "data/processed/sdss_features_reference.csv",
        "features": ROOT / "data/processed/sdss_feature_columns_reference.json",
        "weights": ROOT / "outputs/baseline_v2/sdss/metrics/weights_sdss.csv",
        "id": "objid",
    },
}

BASE = {"score": 0.55, "diversity": 0.20, "entropy": 0.20, "concentration": 0.25}
SCENARIOS = {
    "full": {},
    "no_entropy": {"entropy": 0.0},
    "no_concentration": {"concentration": 0.0},
    "no_entropy_no_concentration": {"entropy": 0.0, "concentration": 0.0},
    "half_entropy": {"entropy": 0.10},
    "double_entropy": {"entropy": 0.40},
    "half_concentration": {"concentration": 0.125},
    "double_concentration": {"concentration": 0.50},
    "half_diversity": {"diversity": 0.10},
    "double_diversity": {"diversity": 0.40},
    "full_percentile_ranks": {},
}
GROUPS = {
    "ablations": [
        "full", "no_entropy", "no_concentration",
        "no_entropy_no_concentration",
    ],
    "coefficients": [
        "full", "half_entropy", "double_entropy",
        "half_concentration", "double_concentration",
        "half_diversity", "double_diversity",
    ],
    "normalization": ["full", "full_percentile_ranks"],
    "all": list(SCENARIOS),
}


def normalized_weights(w):
    w = np.maximum(np.asarray(w, dtype=float), 0.0)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return np.ones(len(w), dtype=float) / len(w)
    return w / w.sum()


def canonical_weights(path, names):
    t = pd.read_csv(path)
    if not {"feature", "weight"}.issubset(t.columns):
        raise ValueError(f"Missing feature/weight columns: {path}")
    if t["feature"].duplicated().any() or set(t["feature"]) != set(names):
        raise ValueError(f"Invalid canonical weight names: {path}")
    return normalized_weights(t.set_index("feature").loc[names, "weight"].to_numpy())


def make_score(logx, w):
    return np.exp(logx @ w)


def components(logx, x, w, top_k, min_eff):
    scores = make_score(logx, w)
    ind = np.argpartition(scores, -top_k)[-top_k:]
    h = -np.sum(w * np.log(w + 1e-12)) / np.log(len(w))
    neff = 1.0 / np.sum(w * w)
    penalty = max(0.0, min_eff - neff) / min_eff
    return scores, {
        "mean_top_score": float(scores[ind].mean()),
        "mean_top_feature_std": float(np.std(x[ind], axis=0).mean()),
        "entropy": float(h),
        "effective_features": float(neff),
        "concentration_penalty": float(penalty),
    }


def objective_value(c, a):
    return (a["score"] * c["mean_top_score"]
            + a["diversity"] * c["mean_top_feature_std"]
            + a["entropy"] * c["entropy"]
            - a["concentration"] * c["concentration_penalty"])


def rank_normalized(x):
    # Empirical CDF of the already clipped normalized feature values.
    # Ties receive average ranks. Preserve the observed feature orientation.
    n = len(x)
    cols = []
    for j in range(x.shape[1]):
        ranks = pd.Series(x[:, j]).rank(method="average").to_numpy(float)
        cols.append(ranks / (n + 1))
    return np.column_stack(cols)


def stable_order(scores, identifiers):
    return np.lexsort((identifiers, -scores))


def run(catalog, args):
    conf = CAT[catalog]
    with open(conf["features"], encoding="utf-8") as f:
        features = json.load(f)
    if not conf["weights"].exists():
        raise FileNotFoundError(f"Missing canonical weights: {conf['weights']}")
    original_w = canonical_weights(conf["weights"], features)
    df = pd.read_csv(conf["matrix"], dtype={conf["id"]: "string"})
    ids = df[conf["id"]].astype(str).to_numpy()
    if pd.isna(df[conf["id"]]).any() or len(set(ids)) != len(ids):
        raise ValueError("Nonunique or missing identifiers in canonical features")
    raw = df[features].to_numpy(dtype=float)
    if not np.isfinite(raw).all() or np.min(raw) < -1e-8 or np.max(raw) > 1 + 1e-8:
        raise ValueError("Feature columns must be finite in [0,1]")
    x = np.clip(raw, EPS, 1.0)
    logx = np.log(x)
    rank_x = np.clip(rank_normalized(raw), EPS, 1.0)
    rank_logx = np.log(rank_x)
    k = min(args.top_k, len(df))
    baseline_scores = make_score(logx, original_w)
    baseline_order = stable_order(baseline_scores, ids)
    baseline_ids = set(ids[baseline_order[:k]])
    baseline_rank = pd.Series(baseline_scores).rank(method="average")

    print(f"\n{catalog.upper()} n={len(df):,} K={k:,} features={features}", flush=True)
    print(f"Canonical weights: {conf['weights']}", flush=True)
    print(dict(zip(features, original_w)), flush=True)

    metric_rows, weight_rows = [], []
    selected_scenarios = GROUPS[args.group]
    for scen in selected_scenarios:
        a = {**BASE, **SCENARIOS[scen]}
        rx, lx = (rank_x, rank_logx) if scen == "full_percentile_ranks" else (x, logx)
        print(f"  Starting {scen} (seed={args.seed})", flush=True)
        t0 = time.perf_counter()

        def neg_objective(w_raw):
            w = normalized_weights(w_raw)
            _, c = components(lx, rx, w, k, args.min_effective_features)
            return -objective_value(c, a)

        fit = differential_evolution(
            neg_objective,
            bounds=[(0.0, 1.0)] * len(features),
            seed=args.seed,
            maxiter=args.maxiter,
            popsize=args.popsize,
            tol=args.tol,
            polish=True,
            workers=1,
        )
        duration = time.perf_counter() - t0
        w = normalized_weights(fit.x)
        scores, comp = components(lx, rx, w, k, args.min_effective_features)
        ordered = stable_order(scores, ids)
        overlap = len(set(ids[ordered[:k]]) & baseline_ids)
        corr = spearmanr(
            pd.Series(scores).rank(method="average"), baseline_rank
        ).statistic
        cos = np.dot(w, original_w) / (np.linalg.norm(w) * np.linalg.norm(original_w))
        metric_rows.append({
            "catalog": catalog, "scenario": scen,
            "normalization": "empirical_rank_of_clipped_features"
                             if scen == "full_percentile_ranks" else "original_1_99",
            "seed": args.seed, "maxiter": args.maxiter, "popsize": args.popsize,
            "optimizer_success": bool(fit.success), "optimizer_message": str(fit.message),
            "optimizer_nit": int(fit.nit), "optimizer_nfev": int(fit.nfev),
            "optimizer_wall_seconds": float(duration),
            "coefficient_score": a["score"],
            "coefficient_diversity": a["diversity"],
            "coefficient_entropy": a["entropy"],
            "coefficient_concentration": a["concentration"],
            "objective_value": float(-fit.fun),
            **comp,
            "spearman_vs_canonical": float(corr),
            "top_k_overlap_vs_canonical": int(overlap),
            "top_k_jaccard_vs_canonical": float(overlap / (2 * k - overlap)),
            "weight_cosine_vs_canonical": float(cos),
            "weight_l1_vs_canonical": float(np.abs(w - original_w).sum()),
        })
        for name, weight in zip(features, w):
            weight_rows.append({
                "catalog": catalog, "scenario": scen,
                "feature": name, "weight": float(weight),
                "canonical_weight": float(original_w[features.index(name)]),
            })
        print(
            f"    obj={-fit.fun:.6f} neff={comp['effective_features']:.3f} "
            f"overlap={overlap}/{k} rho={corr:.5f} "
            f"success={fit.success} seconds={duration:.1f}", flush=True
        )

    OUT.mkdir(parents=True, exist_ok=True)
    metric = pd.DataFrame(metric_rows)
    metric.to_csv(OUT / f"{catalog}_{args.group}_metrics.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(
        OUT / f"{catalog}_{args.group}_weights.csv", index=False
    )
    return metric


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", choices=["gaia", "sdss", "both"], default="sdss")
    p.add_argument("--group", choices=list(GROUPS), default="ablations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=1000)
    p.add_argument("--min-effective-features", type=float, default=3)
    p.add_argument("--maxiter", type=int, default=180)
    p.add_argument("--popsize", type=int, default=15)
    p.add_argument("--tol", type=float, default=1e-6)
    a = p.parse_args()
    if a.top_k < 1 or a.maxiter < 1 or a.popsize < 2:
        p.error("Invalid top-k, maxiter, or popsize")
    cs = ["gaia", "sdss"] if a.catalog == "both" else [a.catalog]
    frames = [run(c, a) for c in cs]
    OUT.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUT / f"combined_{a.group}_metrics.csv", index=False)
    metadata = {
        "experiment": 44, "group": a.group, "catalog": a.catalog,
        "seed": a.seed, "top_k": a.top_k,
        "min_effective_features": a.min_effective_features,
        "maxiter": a.maxiter, "popsize": a.popsize, "tol": a.tol,
        "baseline_coefficients": BASE,
        "scenarios": selected if (selected := GROUPS[a.group]) else [],
        "caution": "Convergence and objective-induced weight structure, not external utility",
        "rank_normalization_caution": "Empirical ranks of already clipped feature values",
    }
    (OUT / f"experiment44_{a.group}_run_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nSaved:", OUT, flush=True)
    print(combined[[
        "catalog", "scenario", "optimizer_success",
        "effective_features", "top_k_overlap_vs_canonical",
        "spearman_vs_canonical", "weight_cosine_vs_canonical",
        "optimizer_wall_seconds",
    ]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
