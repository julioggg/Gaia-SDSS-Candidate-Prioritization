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
    p for p in feature_json_candidates
    if p.exists()
)

with open(feature_json, "r", encoding="utf-8") as f:
    feature_cols = json.load(f)

X = df[feature_cols].to_numpy(float)

min_eff = float(
    cfg["optimization"]["min_effective_features"]
)

n_features = len(feature_cols)

TOP_K_VALUES = [250, 500, 1000, 1500, 2500]
SEEDS = [0, 1, 2]


def normalize_weights(w):
    w = np.clip(
        np.asarray(w, dtype=float),
        1e-12,
        None
    )
    return w / np.sum(w)


def weighted_geometric_score(X, w):
    w = normalize_weights(w)

    return np.exp(
        np.sum(
            w * np.log(
                np.clip(X, 1e-12, 1.0)
            ),
            axis=1
        )
    )


def entropy_of_weights(w):
    w = normalize_weights(w)

    return (
        -np.sum(
            w * np.log(w + 1e-12)
        )
        / np.log(len(w))
    )


def effective_number_of_features(w):
    w = normalize_weights(w)
    return 1.0 / np.sum(w**2)


# ==========================================================
# Baseline K=1000 reference
# ==========================================================

baseline_dir = (
    Path("outputs")
    / "baseline_reference"
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

w_ref = np.array(
    [weight_map[c] for c in feature_cols],
    dtype=float
)

score_ref = weighted_geometric_score(
    X,
    w_ref
)

rank_ref = pd.Series(
    score_ref
).rank(
    method="average"
).to_numpy()

baseline_top_k = int(
    cfg["optimization"]["top_k"]
)

idx_ref_1000 = np.argpartition(
    score_ref,
    -baseline_top_k
)[-baseline_top_k:]

set_ref_1000 = set(
    idx_ref_1000.tolist()
)


rows = []
weight_rows = []


for top_k in TOP_K_VALUES:

    print()
    print("=" * 78)
    print(f"TOP K = {top_k}")
    print("=" * 78)

    def objective(w_raw):
        w = normalize_weights(w_raw)

        score = weighted_geometric_score(
            X,
            w
        )

        k = min(
            top_k,
            len(score)
        )

        idx = np.argpartition(
            score,
            -k
        )[-k:]

        top_X = X[idx]
        top_scores = score[idx]

        mean_score = np.mean(
            top_scores
        )

        diversity_top = np.mean(
            np.std(
                top_X,
                axis=0
            )
        )

        entropy = entropy_of_weights(
            w
        )

        eff_feats = (
            effective_number_of_features(
                w
            )
        )

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

        rank = pd.Series(
            score
        ).rank(
            method="average"
        ).to_numpy()

        rho = spearmanr(
            rank_ref,
            rank
        ).statistic

        # Compare all configurations to the
        # baseline top-1000 selection.
        overlap_1000 = len(
            selected & set_ref_1000
        )

        # Fraction of the current selection
        # that is also in baseline top-1000.
        precision_like = (
            overlap_1000 / top_k
        )

        # Fraction of baseline top-1000
        # recovered by this selection.
        recall_like = (
            overlap_1000
            / baseline_top_k
        )

        union = len(
            selected | set_ref_1000
        )

        jaccard = (
            overlap_1000 / union
        )

        rows.append({
            "top_k": top_k,
            "seed": seed,
            "objective": float(
                -result.fun
            ),
            "mean_top_score": float(
                np.mean(score[idx])
            ),
            "effective_features": float(
                effective_number_of_features(
                    w
                )
            ),
            "entropy": float(
                entropy_of_weights(w)
            ),
            "rank_spearman_vs_k1000": float(
                rho
            ),
            "overlap_with_baseline_top1000":
                overlap_1000,
            "fraction_current_in_baseline_top1000":
                float(precision_like),
            "fraction_baseline_top1000_recovered":
                float(recall_like),
            "jaccard_vs_baseline_top1000":
                float(jaccard),
            "nfev": int(
                result.nfev
            ),
        })

        for feature, weight in zip(
            feature_cols,
            w
        ):
            weight_rows.append({
                "top_k": top_k,
                "seed": seed,
                "feature": feature,
                "weight": float(weight),
            })


OUT = (
    Path("outputs")
    / "sensitivity_reference"
    / catalog
    / "top_k"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

metrics = pd.DataFrame(
    rows
)

weights = pd.DataFrame(
    weight_rows
)

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
    .groupby("top_k")
    .agg(
        objective_mean=(
            "objective",
            "mean"
        ),
        objective_std=(
            "objective",
            "std"
        ),
        mean_top_score=(
            "mean_top_score",
            "mean"
        ),
        neff_mean=(
            "effective_features",
            "mean"
        ),
        neff_std=(
            "effective_features",
            "std"
        ),
        entropy_mean=(
            "entropy",
            "mean"
        ),
        spearman_mean=(
            "rank_spearman_vs_k1000",
            "mean"
        ),
        spearman_min=(
            "rank_spearman_vs_k1000",
            "min"
        ),
        overlap_mean=(
            "overlap_with_baseline_top1000",
            "mean"
        ),
        current_in_baseline_mean=(
            "fraction_current_in_baseline_top1000",
            "mean"
        ),
        baseline_recovered_mean=(
            "fraction_baseline_top1000_recovered",
            "mean"
        ),
        jaccard_mean=(
            "jaccard_vs_baseline_top1000",
            "mean"
        ),
    )
)

summary.to_csv(
    OUT / "summary.csv"
)


weight_summary = (
    weights
    .groupby([
        "top_k",
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
print("=" * 100)
print("TOP-K SENSITIVITY SUMMARY")
print("=" * 100)
print(
    summary.to_string()
)

print()
print("=" * 100)
print("WEIGHT SUMMARY")
print("=" * 100)
print(
    weight_summary.to_string()
)

print()
print("Saved to:", OUT)
