#!/usr/bin/env python3
"""Experiment 45: multiseed confirmation of reviewer-2 SDSS penalty ablations.

Reoptimizes the unchanged reference-feature matrix for four objective variants.
Writes a checkpoint CSV after EVERY run and preserves ID-keyed top-K lists.
Flag optimizer_success=False runs as unconverged; agreement across seeds may
still be reported as empirical but does not prove mathematical convergence.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

ROOT = Path.home() / "universe_revision" / "revision_v2"
DATA = ROOT / "data/processed/sdss_features_reference.csv"
FEATURE_JSON = ROOT / "data/processed/sdss_feature_columns_reference.json"
WEIGHTS = ROOT / "outputs/baseline_v2/sdss/metrics/weights_sdss.csv"
OUT = ROOT / "outputs/reviewer_round2/objective_convergence"
EPS = 1e-6
COEFFS = {
    "full": (0.55, 0.20, 0.20, 0.25),
    "no_entropy": (0.55, 0.20, 0.0, 0.25),
    "no_concentration": (0.55, 0.20, 0.20, 0.0),
    "no_entropy_no_concentration": (0.55, 0.20, 0.0, 0.0),
}


def norm(w):
    w = np.maximum(np.asarray(w, dtype=float), 0.0)
    if w.sum() <= 0:
        return np.full(len(w), 1.0 / len(w))
    return w / w.sum()


def rank_order(scores, ids):
    return np.lexsort((ids, -scores))


def objective_from_weights(logx, x, w, k, min_eff, coeffs):
    score = np.exp(logx @ w)
    indices = np.argpartition(score, -k)[-k:]
    mean_score = float(score[indices].mean())
    diversity = float(np.std(x[indices], axis=0).mean())
    entropy = float(-np.sum(w * np.log(w + 1e-12)) / np.log(len(w)))
    neff = float(1 / np.sum(w ** 2))
    concentration = max(0.0, min_eff - neff) / min_eff
    objective = (
        coeffs[0] * mean_score
        + coeffs[1] * diversity
        + coeffs[2] * entropy
        - coeffs[3] * concentration
    )
    return objective, score, (mean_score, diversity, entropy, neff, concentration)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,7,123")
    p.add_argument("--scenarios", default="full,no_entropy,no_concentration,no_entropy_no_concentration")
    p.add_argument("--top-k", type=int, default=1000)
    p.add_argument("--maxiter", type=int, default=500)
    p.add_argument("--popsize", type=int, default=20)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--min-eff", type=float, default=3.0)
    args = p.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    if not seeds or not scenarios or len(seeds) != len(set(seeds)):
        p.error("Supply unique nonempty seed and scenario lists")
    if any(s not in COEFFS for s in scenarios):
        p.error(f"Unknown scenario; choose among {list(COEFFS)}")
    if args.top_k <= 0 or args.maxiter <= 0 or args.popsize <= 1:
        p.error("Invalid top-k or optimizer budget")

    with FEATURE_JSON.open(encoding="utf-8") as f:
        features = json.load(f)
    df = pd.read_csv(DATA, dtype={"objid": "string"})
    ids = df["objid"].astype(str).to_numpy()
    if df["objid"].isna().any() or len(set(ids)) != len(ids):
        raise ValueError("Object identifiers missing or duplicated")
    x = df[features].to_numpy(float)
    if not np.isfinite(x).all() or x.min() < -1e-8 or x.max() > 1 + 1e-8:
        raise ValueError("Features must be finite within [0,1]")
    x = np.clip(x, EPS, 1.0)
    logx = np.log(x)
    weights_table = pd.read_csv(WEIGHTS).set_index("feature")
    if set(weights_table.index) != set(features):
        raise ValueError("Reference weights and feature names do not match")
    canonical_w = norm(weights_table.loc[features, "weight"].to_numpy(float))
    k = min(args.top_k, len(df))
    baseline_score = np.exp(logx @ canonical_w)
    baseline_top = set(ids[rank_order(baseline_score, ids)[:k]])
    baseline_ranks = pd.Series(baseline_score).rank(method="average")

    OUT.mkdir(parents=True, exist_ok=True)
    rows, weight_rows, bestlists, scorelists = [], [], {}, {}
    for scenario in scenarios:
        coeffs = COEFFS[scenario]
        for seed in seeds:
            print(f"\nSCENARIO {scenario} | seed={seed} | maxiter={args.maxiter}",
                  flush=True)

            def negobjective(raw):
                w = norm(raw)
                value, _, _ = objective_from_weights(
                    logx, x, w, k, args.min_eff, coeffs)
                return -value

            started = perf_counter()
            result = differential_evolution(
                negobjective, bounds=[(0.0, 1.0)] * len(features),
                seed=seed, maxiter=args.maxiter, popsize=args.popsize,
                tol=args.tol, polish=True, workers=1,
            )
            wall = perf_counter() - started
            w = norm(result.x)
            score_value, scores, comps = objective_from_weights(
                logx, x, w, k, args.min_eff, coeffs)
            ordered = rank_order(scores, ids)
            topids = ids[ordered[:k]]
            overlap = len(set(topids) & baseline_top)
            rho = spearmanr(
                pd.Series(scores).rank(method="average"),
                baseline_ranks,
            ).statistic
            key = f"{scenario}_seed{seed}"
            bestlists[key] = set(topids)
            scorelists[key] = scores
            pd.DataFrame({
                "rank": np.arange(1, k + 1),
                "objid": topids,
                "score": scores[ordered[:k]],
            }).to_csv(OUT / f"{key}_topk.csv", index=False)

            row = {
                "scenario": scenario, "seed": seed,
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "n_iterations": int(result.nit),
                "n_function_evaluations": int(result.nfev),
                "wall_seconds": wall, "objective": score_value,
                "mean_top_score": comps[0],
                "mean_top_feature_std": comps[1],
                "normalized_weight_entropy": comps[2],
                "effective_feature_number": comps[3],
                "concentration_penalty": comps[4],
                "topk_overlap_vs_baseline": overlap,
                "jaccard_vs_baseline": overlap / (2 * k - overlap),
                "spearman_vs_baseline": float(rho),
                "maxiter_budget": args.maxiter, "popsize": args.popsize,
                "relative_tolerance": args.tol,
            }
            rows.append(row)
            weight_rows.extend({
                "scenario": scenario, "seed": seed,
                "feature": feature, "weight": float(weight),
                "baseline_weight": float(canonical_w[i]),
            } for i, (feature, weight) in enumerate(zip(features, w)))
            pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
            pd.DataFrame(weight_rows).to_csv(OUT / "weights.csv", index=False)
            print(f"success={result.success}; objective={score_value:.6f}; "
                  f"Neff={comps[3]:.4f}; overlap={overlap}/{k}; "
                  f"rho={rho:.6f}; time={wall:.1f}s", flush=True)

    within = []
    for scenario in scenarios:
        for i, seed1 in enumerate(seeds):
            for seed2 in seeds[i + 1:]:
                key1 = f"{scenario}_seed{seed1}"
                key2 = f"{scenario}_seed{seed2}"
                overlap = len(bestlists[key1] & bestlists[key2])
                rho = spearmanr(
                    pd.Series(scorelists[key1]).rank(method="average"),
                    pd.Series(scorelists[key2]).rank(method="average"),
                ).statistic
                within.append({
                    "scenario": scenario,
                    "seed_a": seed1, "seed_b": seed2,
                    "topk_overlap": overlap,
                    "topk_jaccard": overlap / (2 * k - overlap),
                    "rank_spearman": float(rho),
                })
    pd.DataFrame(within).to_csv(OUT / "within_scenario_seed_agreement.csv", index=False)
    (OUT / "run_metadata.json").write_text(json.dumps({
        "experiment": 45, "matrix_path": str(DATA),
        "feature_json": str(FEATURE_JSON),
        "canonical_weights_file": str(WEIGHTS),
        "top_k": k, "seeds": seeds, "scenarios": scenarios,
        "maxiter": args.maxiter, "popsize": args.popsize,
        "tol": args.tol, "min_eff": args.min_eff,
        "coefficients": COEFFS,
        "note": "Do not report runs with optimizer_success=False as converged",
    }, indent=2), encoding="utf-8")
    print("\nRUN SUMMARY\n", pd.DataFrame(rows).to_string(index=False), flush=True)
    print("\nWITHIN SCENARIO AGREEMENT\n",
          pd.DataFrame(within).to_string(index=False), flush=True)
    print("\nSaved to", OUT, flush=True)


if __name__ == "__main__":
    main()
