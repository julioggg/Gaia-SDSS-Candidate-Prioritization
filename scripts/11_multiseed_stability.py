from pathlib import Path
import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

from scipy.optimize import differential_evolution
from scipy.stats import spearmanr


parser = argparse.ArgumentParser()
parser.add_argument("--catalog", choices=["gaia", "sdss"], required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--n_seeds", type=int, default=30)
args = parser.parse_args()


catalog = args.catalog

with open(args.config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


feature_file = Path(cfg["outputs"]["processed_csv"])

feature_json_candidates = [
    Path(f"data/processed/{catalog}_feature_columns_reference.json"),
    Path(f"data/processed/{catalog}_feature_columns.json"),
]

feature_file_json = None

for p in feature_json_candidates:
    if p.exists():
        feature_file_json = p
        break

if feature_file_json is None:
    raise FileNotFoundError(
        f"No feature-columns JSON found for {catalog}"
    )


with open(feature_file_json, "r", encoding="utf-8") as f:
    feature_cols = json.load(f)


df = pd.read_csv(feature_file)

X = (
    df[feature_cols]
    .to_numpy(dtype=float)
)

top_k = int(cfg["optimization"]["top_k"])

min_eff = float(
    cfg["optimization"]["min_effective_features"]
)

n_features = len(feature_cols)


def normalize_weights(w):
    w = np.clip(
        np.asarray(w, dtype=float),
        1e-12,
        None
    )
    return w / np.sum(w)


def weighted_geometric_score(X, w):
    w = normalize_weights(w)

    eps = 1e-12

    return np.exp(
        np.sum(
            w * np.log(np.clip(X, eps, 1.0)),
            axis=1
        )
    )


def entropy_of_weights(w):
    w = normalize_weights(w)

    return (
        -np.sum(w * np.log(w + 1e-12))
        / np.log(len(w))
    )


def effective_number_of_features(w):
    w = normalize_weights(w)
    return 1.0 / np.sum(w ** 2)


def objective(w_raw):
    w = normalize_weights(w_raw)

    score = weighted_geometric_score(X, w)

    k = min(top_k, len(score))

    idx = np.argpartition(
        score,
        -k
    )[-k:]

    top_X = X[idx]
    top_scores = score[idx]

    mean_score = np.mean(top_scores)

    diversity_top = np.mean(
        np.std(top_X, axis=0)
    )

    entropy = entropy_of_weights(w)

    eff_feats = effective_number_of_features(w)

    penalty_concentration = (
        max(
            0.0,
            min_eff - eff_feats
        )
        / min_eff
    )

    value = (
        0.55 * mean_score
        + 0.20 * diversity_top
        + 0.20 * entropy
        - 0.25 * penalty_concentration
    )

    return -value


# ==========================================================
# Seed 42 reference
# ==========================================================

baseline_dir = (
    Path("outputs")
    / "baseline_v2"
    / catalog
    / "metrics"
)

baseline_weights = pd.read_csv(
    baseline_dir / f"weights_{catalog}.csv"
)

weight_map = dict(
    zip(
        baseline_weights["feature"],
        baseline_weights["weight"]
    )
)

w42 = np.array(
    [weight_map[c] for c in feature_cols],
    dtype=float
)

score42 = weighted_geometric_score(
    X,
    w42
)

rank42 = pd.Series(
    score42
).rank(
    method="average"
).to_numpy()

idx42 = np.argpartition(
    score42,
    -top_k
)[-top_k:]

set42 = set(
    idx42.tolist()
)


# ==========================================================
# Multi-seed experiment
# ==========================================================

seeds = list(range(args.n_seeds))

# Ensure seed 42 is explicitly included.
if 42 not in seeds:
    seeds.append(42)

rows = []
weight_rows = []


print("=" * 76)
print(f"MULTI-SEED STABILITY: {catalog.upper()}")
print("=" * 76)
print("Rows:", len(df))
print("Features:", feature_cols)
print("Top K:", top_k)
print("Seeds:", len(seeds))
print()


bounds = [(0, 1)] * n_features


for run_i, seed in enumerate(seeds, start=1):

    print(
        f"[{run_i:02d}/{len(seeds):02d}] "
        f"seed={seed}"
    )

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=120,
        popsize=15,
        tol=1e-6,
        polish=True,
        workers=1
    )

    w = normalize_weights(
        result.x
    )

    score = weighted_geometric_score(
        X,
        w
    )

    idx = np.argpartition(
        score,
        -top_k
    )[-top_k:]

    selected = set(
        idx.tolist()
    )

    intersection = len(
        selected & set42
    )

    union = len(
        selected | set42
    )

    jaccard = (
        intersection / union
        if union
        else np.nan
    )

    rank = pd.Series(
        score
    ).rank(
        method="average"
    ).to_numpy()

    rho = spearmanr(
        rank42,
        rank
    ).statistic

    rows.append({
        "seed": seed,
        "objective": float(-result.fun),
        "mean_top_score": float(
            np.mean(score[idx])
        ),
        "effective_features":
            float(
                effective_number_of_features(w)
            ),
        "entropy":
            float(
                entropy_of_weights(w)
            ),
        "topk_overlap_n":
            intersection,
        "topk_jaccard_vs_seed42":
            float(jaccard),
        "rank_spearman_vs_seed42":
            float(rho),
        "optimizer_success":
            bool(result.success),
        "nfev":
            int(result.nfev),
    })

    for feature, weight in zip(
        feature_cols,
        w
    ):
        weight_rows.append({
            "seed": seed,
            "feature": feature,
            "weight": float(weight),
        })


# ==========================================================
# Save
# ==========================================================

OUT = (
    Path("outputs")
    / "stability"
    / catalog
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

results = pd.DataFrame(rows)
weights = pd.DataFrame(weight_rows)

results.to_csv(
    OUT / "multiseed_metrics.csv",
    index=False
)

weights.to_csv(
    OUT / "multiseed_weights.csv",
    index=False
)


summary_weights = (
    weights
    .groupby("feature")["weight"]
    .agg([
        "mean",
        "std",
        "min",
        "median",
        "max"
    ])
    .sort_values(
        "mean",
        ascending=False
    )
)

summary_weights.to_csv(
    OUT / "weight_stability_summary.csv"
)


summary_metrics = {
    "catalog": catalog,
    "n_runs": int(len(results)),
    "objective_mean":
        float(results["objective"].mean()),
    "objective_std":
        float(results["objective"].std()),
    "neff_mean":
        float(results["effective_features"].mean()),
    "neff_std":
        float(results["effective_features"].std()),
    "entropy_mean":
        float(results["entropy"].mean()),
    "entropy_std":
        float(results["entropy"].std()),
    "topk_jaccard_mean":
        float(
            results[
                "topk_jaccard_vs_seed42"
            ].mean()
        ),
    "topk_jaccard_min":
        float(
            results[
                "topk_jaccard_vs_seed42"
            ].min()
        ),
    "rank_spearman_mean":
        float(
            results[
                "rank_spearman_vs_seed42"
            ].mean()
        ),
    "rank_spearman_min":
        float(
            results[
                "rank_spearman_vs_seed42"
            ].min()
        ),
}


with open(
    OUT / "multiseed_summary.json",
    "w"
) as f:
    json.dump(
        summary_metrics,
        f,
        indent=2
    )


print()
print("=" * 76)
print("SUMMARY")
print("=" * 76)

print()
print("Metrics:")
print(results.describe().to_string())

print()
print("Weight stability:")
print(summary_weights.to_string())

print()
print("Saved to:", OUT)
print("DONE.")
