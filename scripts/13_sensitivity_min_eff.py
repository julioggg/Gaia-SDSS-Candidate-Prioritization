from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import yaml

from scipy.optimize import differential_evolution
from scipy.stats import spearmanr


parser = argparse.ArgumentParser()
parser.add_argument("--catalog", choices=["gaia", "sdss"], required=True)
parser.add_argument("--config", required=True)
args = parser.parse_args()

catalog = args.catalog

with open(args.config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

df = pd.read_csv(cfg["outputs"]["processed_csv"])

feature_json_candidates = [
    Path(f"data/processed/{catalog}_feature_columns_reference.json"),
    Path(f"data/processed/{catalog}_feature_columns.json"),
]

feature_json = next(
    p for p in feature_json_candidates if p.exists()
)

with open(feature_json, "r", encoding="utf-8") as f:
    feature_cols = json.load(f)

X = df[feature_cols].to_numpy(float)

top_k = int(cfg["optimization"]["top_k"])
n_features = len(feature_cols)

MIN_EFF_VALUES = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
SEEDS = [0, 1, 2, 3, 4]


def normalize_weights(w):
    w = np.clip(np.asarray(w, float), 1e-12, None)
    return w / w.sum()


def weighted_geometric_score(X, w):
    w = normalize_weights(w)
    return np.exp(
        np.sum(
            w * np.log(np.clip(X, 1e-12, 1.0)),
            axis=1
        )
    )


def entropy_of_weights(w):
    w = normalize_weights(w)
    return (
        -np.sum(w*np.log(w + 1e-12))
        / np.log(len(w))
    )


def effective_number_of_features(w):
    w = normalize_weights(w)
    return 1.0 / np.sum(w**2)


# Reference baseline v2
baseline_dir = (
    Path("outputs")
    / "baseline_v2"
    / catalog
    / "metrics"
)

bw = pd.read_csv(
    baseline_dir / f"weights_{catalog}.csv"
)

wm = dict(zip(bw["feature"], bw["weight"]))
w_ref = np.array([wm[c] for c in feature_cols])

score_ref = weighted_geometric_score(X, w_ref)

idx_ref = np.argpartition(
    score_ref,
    -top_k
)[-top_k:]

set_ref = set(idx_ref.tolist())

rank_ref = pd.Series(score_ref).rank(
    method="average"
).to_numpy()


rows = []
weight_rows = []

for min_eff in MIN_EFF_VALUES:

    print()
    print("=" * 78)
    print(f"MIN EFFECTIVE FEATURES = {min_eff}")
    print("=" * 78)

    def objective(w_raw):
        w = normalize_weights(w_raw)

        score = weighted_geometric_score(X, w)

        idx = np.argpartition(
            score,
            -top_k
        )[-top_k:]

        top_X = X[idx]
        top_scores = score[idx]

        mean_score = np.mean(top_scores)
        diversity_top = np.mean(
            np.std(top_X, axis=0)
        )

        entropy = entropy_of_weights(w)
        eff = effective_number_of_features(w)

        penalty = max(
            0.0,
            min_eff - eff
        ) / min_eff

        value = (
            0.55 * mean_score
            + 0.20 * diversity_top
            + 0.20 * entropy
            - 0.25 * penalty
        )

        return -value

    for seed in SEEDS:

        print(f"  seed={seed}")

        result = differential_evolution(
            objective,
            bounds=[(0, 1)] * n_features,
            seed=seed,
            maxiter=180,
            popsize=20,
            tol=1e-7,
            polish=True,
            workers=1
        )

        w = normalize_weights(result.x)

        score = weighted_geometric_score(
            X,
            w
        )

        idx = np.argpartition(
            score,
            -top_k
        )[-top_k:]

        selected = set(idx.tolist())

        jaccard = (
            len(selected & set_ref)
            / len(selected | set_ref)
        )

        rank = pd.Series(score).rank(
            method="average"
        ).to_numpy()

        rho = spearmanr(
            rank_ref,
            rank
        ).statistic

        rows.append({
            "min_effective_features": min_eff,
            "seed": seed,
            "objective": float(-result.fun),
            "mean_top_score": float(
                np.mean(score[idx])
            ),
            "effective_features": float(
                effective_number_of_features(w)
            ),
            "entropy": float(
                entropy_of_weights(w)
            ),
            "jaccard_vs_baseline_v2": float(
                jaccard
            ),
            "spearman_vs_baseline_v2": float(
                rho
            ),
            "nfev": int(result.nfev),
        })

        for feature, weight in zip(
            feature_cols,
            w
        ):
            weight_rows.append({
                "min_effective_features": min_eff,
                "seed": seed,
                "feature": feature,
                "weight": float(weight),
            })


OUT = (
    Path("outputs")
    / "sensitivity"
    / catalog
    / "min_effective_features"
)

OUT.mkdir(parents=True, exist_ok=True)

metrics = pd.DataFrame(rows)
weights = pd.DataFrame(weight_rows)

metrics.to_csv(
    OUT / "metrics.csv",
    index=False
)

weights.to_csv(
    OUT / "weights.csv",
    index=False
)


summary = (
    metrics
    .groupby("min_effective_features")
    .agg(
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        top_score_mean=("mean_top_score", "mean"),
        neff_mean=("effective_features", "mean"),
        neff_std=("effective_features", "std"),
        entropy_mean=("entropy", "mean"),
        jaccard_mean=("jaccard_vs_baseline_v2", "mean"),
        jaccard_min=("jaccard_vs_baseline_v2", "min"),
        spearman_mean=("spearman_vs_baseline_v2", "mean"),
        spearman_min=("spearman_vs_baseline_v2", "min"),
    )
)

summary.to_csv(
    OUT / "summary.csv"
)


weight_summary = (
    weights
    .groupby([
        "min_effective_features",
        "feature"
    ])["weight"]
    .agg([
        "mean",
        "std",
        "min",
        "max"
    ])
)

weight_summary.to_csv(
    OUT / "weight_summary.csv"
)

print()
print("=" * 90)
print("SENSITIVITY SUMMARY")
print("=" * 90)
print(summary.to_string())

print()
print("=" * 90)
print("WEIGHT SUMMARY")
print("=" * 90)
print(weight_summary.to_string())

print()
print("Saved to:", OUT)
