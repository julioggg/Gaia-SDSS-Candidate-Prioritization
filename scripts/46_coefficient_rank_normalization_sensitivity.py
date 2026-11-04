#!/usr/bin/env python3
"""Experiment 46. Reviewer R1: coefficient sensitivity and percentile-rank normalization.

Same 50,000 SDSS DR17 frozen reference-based feature rows and six features as
the published baseline. Reoptimizes with alternate coefficients, and tests
empirical ranks of ALREADY clipped feature columns. This rank diagnostic does
not reconstruct feature information suppressed by the original clipping.

For each setting, run multiple seeds and preserve results separately.
SciPy optimizer success is a stopping-criterion flag, not proof of global
optimality; report between-seed stability and independent validation separately.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

ROOT = Path.home() / "universe_revision" / "revision_v2"
DATA = ROOT / "data/processed/sdss_features_reference.csv"
FEATURE_JSON = ROOT / "data/processed/sdss_feature_columns_reference.json"
WEIGHTS = ROOT / "outputs/baseline_v2/sdss/metrics/weights_sdss.csv"
OUT = ROOT / "outputs/reviewer_round2/coefficient_rank_sensitivity"
EPS = 1e-6
BASE = {"score": 0.55, "diversity": 0.20, "entropy": 0.20, "concentration": 0.25}
CHANGES = {
    "original": {},
    "entropy_half": {"entropy": 0.10},
    "entropy_double": {"entropy": 0.40},
    "concentration_half": {"concentration": 0.125},
    "concentration_double": {"concentration": 0.50},
    "diversity_half": {"diversity": 0.10},
    "diversity_double": {"diversity": 0.40},
    "score_half": {"score": 0.275},
    "score_double": {"score": 1.10},
    "empirical_percentile_rank": {},
}


def norm(w):
    w = np.maximum(np.asarray(w, float), 0.0)
    return (w / w.sum()) if w.sum() > 0 else np.ones(len(w)) / len(w)


def score(X_log, w):
    return np.exp(X_log @ w)


def order(s, ids):
    return np.lexsort((ids, -s))


def components(X, X_log, w, k, min_eff):
    s = score(X_log, w)
    top_idx = np.argpartition(s, -k)[-k:]
    neff = float(1.0 / np.sum(w ** 2))
    return s, {
        "mean_score": float(np.mean(s[top_idx])),
        "diversity": float(np.std(X[top_idx], axis=0).mean()),
        "entropy": float(-np.sum(w * np.log(w + 1e-12)) / np.log(len(w))),
        "neff": neff,
        "concentration_penalty": float(max(0.0, min_eff - neff) / min_eff),
    }


def evaluate(c, weights):
    return (weights["score"] * c["mean_score"]
            + weights["diversity"] * c["diversity"]
            + weights["entropy"] * c["entropy"]
            - weights["concentration"] * c["concentration_penalty"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,7,123")
    p.add_argument("--settings", default="all")
    p.add_argument("--maxiter", type=int, default=500)
    p.add_argument("--popsize", type=int, default=20)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--top-k", type=int, default=1000)
    p.add_argument("--min-eff", type=float, default=3.0)
    a = p.parse_args()
    seeds = [int(v.strip()) for v in a.seeds.split(",") if v.strip()]
    settings = list(CHANGES) if a.settings == "all" else [
        v.strip() for v in a.settings.split(",") if v.strip()
    ]
    if not seeds or len(seeds) != len(set(seeds)):
        p.error("Require nonempty, unique seeds")
    if not settings or any(v not in CHANGES for v in settings):
        p.error("Invalid --settings")
    if a.maxiter < 1 or a.popsize < 2 or a.top_k < 1:
        p.error("Invalid optimizer budget or top-K")

    with FEATURE_JSON.open(encoding="utf8") as f:
        features = json.load(f)
    df = pd.read_csv(DATA, dtype={"objid": "string"})
    ids = df["objid"].astype(str).to_numpy()
    if df["objid"].isna().any() or len(set(ids)) != len(df):
        raise ValueError("Missing or duplicated SDSS objid")
    x_orig = df[features].to_numpy(float)
    if not np.isfinite(x_orig).all() or np.min(x_orig) < -1e-8 or np.max(x_orig) > 1 + 1e-8:
        raise ValueError("Nonfinite or out-of-domain feature values")
    x_orig = np.clip(x_orig, EPS, 1.0)
    log_orig = np.log(x_orig)
    canonical_df = pd.read_csv(WEIGHTS)
    if canonical_df["feature"].duplicated().any() or set(canonical_df["feature"]) != set(features):
        raise ValueError("Canonical feature mismatch")
    canonical = norm(canonical_df.set_index("feature").loc[features, "weight"].to_numpy(float))
    k = min(len(df), a.top_k)
    baseline_score = score(log_orig, canonical)
    baseline_order = order(baseline_score, ids)
    baseline_top = set(ids[baseline_order[:k]])
    baseline_rank = pd.Series(baseline_score).rank(method="average")
    x_pct = pd.concat(
        [pd.Series(x_orig[:, j]).rank(method="average") / (len(df) + 1)
         for j in range(len(features))], axis=1
    ).to_numpy(float)
    x_pct = np.clip(x_pct, EPS, 1.0)
    log_pct = np.log(x_pct)

    OUT.mkdir(parents=True, exist_ok=True)
    rows, weights_rows, topids_by_setting = [], [], {}
    for setting in settings:
        coefficients = {**BASE, **CHANGES[setting]}
        X, logX = ((x_pct, log_pct) if setting == "empirical_percentile_rank"
                   else (x_orig, log_orig))
        for seed in seeds:
            print(f"Running {setting} seed={seed}", flush=True)

            def target(raw):
                w = norm(raw)
                _, comp = components(X, logX, w, k, a.min_eff)
                return -evaluate(comp, coefficients)

            started = time.perf_counter()
            res = differential_evolution(
                target, bounds=[(0.0, 1.0)] * len(features),
                seed=seed, maxiter=a.maxiter, popsize=a.popsize,
                tol=a.tol, polish=True, workers=1,
            )
            seconds = time.perf_counter() - started
            w = norm(res.x)
            scores, comp = components(X, logX, w, k, a.min_eff)
            chosen_order = order(scores, ids)
            topids = ids[chosen_order[:k]]
            overlap = len(set(topids) & baseline_top)
            rho = float(spearmanr(pd.Series(scores).rank(method="average"), baseline_rank).statistic)
            topids_by_setting[(setting, seed)] = set(topids)
            pd.DataFrame({
                "rank": np.arange(1, k + 1), "objid": topids,
                "score": scores[chosen_order[:k]]
            }).to_csv(OUT / f"topk_{setting}_seed{seed}.csv", index=False)
            row = {
                "setting": setting, "seed": seed,
                "normalization": "rank_of_originally_clipped_features"
                 if setting == "empirical_percentile_rank" else "original_1_99",
                "optimizer_success": bool(res.success),
                "optimizer_message": str(res.message),
                "iterations": int(res.nit), "nfev": int(res.nfev),
                "wall_seconds": float(seconds),
                "objective": float(-res.fun),
                **comp,
                "spearman_vs_canonical": rho,
                "topk_overlap_vs_canonical": overlap,
                "topk_jaccard_vs_canonical": overlap / (2 * k - overlap),
                "weight_cosine_vs_canonical": float(
                    np.dot(w, canonical) / (np.linalg.norm(w) * np.linalg.norm(canonical))
                ),
                **{f"coef_{key}": val for key, val in coefficients.items()},
            }
            rows.append(row)
            weights_rows.extend({
                "setting": setting, "seed": seed,
                "feature": name, "weight": float(value),
                "canonical_weight": float(canonical[j]),
            } for j, (name, value) in enumerate(zip(features, w)))
            pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
            pd.DataFrame(weights_rows).to_csv(OUT / "weights.csv", index=False)
            print(f"  success={res.success} obj={-res.fun:.6f} Neff={comp['neff']:.4f}"
                  f" overlap={overlap}/{k} rho={rho:.6f} time={seconds:.1f}s", flush=True)

    stability = []
    for setting in settings:
        for i, seed_a in enumerate(seeds):
            for seed_b in seeds[i + 1:]:
                A = topids_by_setting[(setting, seed_a)]
                B = topids_by_setting[(setting, seed_b)]
                common = len(A & B)
                stability.append({
                    "setting": setting, "seed_a": seed_a, "seed_b": seed_b,
                    "topk_overlap": common, "topk_jaccard": common / (2 * k - common)
                })
    pd.DataFrame(stability).to_csv(OUT / "within_setting_topk_stability.csv", index=False)
    (OUT / "metadata.json").write_text(json.dumps({
        "experiment": 46, "data": str(DATA), "weights": str(WEIGHTS),
        "feature_columns": features, "seed_list": seeds, "settings": settings,
        "k": k, "maxiter": a.maxiter, "popsize": a.popsize,
        "tol": a.tol, "min_eff": a.min_eff,
        "original_objective": BASE, "coefficient_modifications": CHANGES,
        "caveats": [
            "rank normalization is applied to feature columns AFTER original raw-data clipping",
            "different objective values across settings are not commensurate",
            "optimizer_success is not a global optimality certificate",
            "ranking agreement is not external astronomical utility",
        ],
    }, indent=2), encoding="utf8")
    print("\nSUMMARY", flush=True)
    print(pd.DataFrame(rows)[[
        "setting", "seed", "optimizer_success", "neff",
        "topk_overlap_vs_canonical", "spearman_vs_canonical",
        "wall_seconds"
    ]].to_string(index=False), flush=True)
    print("\nFiles:", OUT, flush=True)


if __name__ == "__main__":
    main()
