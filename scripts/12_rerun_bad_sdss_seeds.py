from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from scipy.optimize import differential_evolution
from scipy.stats import spearmanr


CONFIG = "configs/sdss_baseline_v2.yaml"
BAD_SEEDS = [5, 22, 24]

with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

df = pd.read_csv(cfg["outputs"]["processed_csv"])

with open(
    "data/processed/sdss_feature_columns_reference.json",
    "r",
    encoding="utf-8"
) as f:
    feature_cols = json.load(f)

X = df[feature_cols].to_numpy(float)

top_k = int(cfg["optimization"]["top_k"])
min_eff = float(cfg["optimization"]["min_effective_features"])
n_features = len(feature_cols)


def normalize_weights(w):
    w = np.clip(np.asarray(w, float), 1e-12, None)
    return w / w.sum()


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
    return -np.sum(w*np.log(w + 1e-12)) / np.log(len(w))


def effective_number_of_features(w):
    w = normalize_weights(w)
    return 1.0 / np.sum(w**2)


def objective(w_raw):
    w = normalize_weights(w_raw)
    score = weighted_geometric_score(X, w)

    k = min(top_k, len(score))
    idx = np.argpartition(score, -k)[-k:]

    top_X = X[idx]
    top_scores = score[idx]

    mean_score = np.mean(top_scores)
    diversity_top = np.mean(np.std(top_X, axis=0))
    entropy = entropy_of_weights(w)
    eff_feats = effective_number_of_features(w)

    penalty_concentration = max(
        0.0,
        min_eff - eff_feats
    ) / min_eff

    value = (
        0.55 * mean_score
        + 0.20 * diversity_top
        + 0.20 * entropy
        - 0.25 * penalty_concentration
    )

    return -value


# Baseline seed 42
weights42 = pd.read_csv(
    "outputs/baseline_v2/sdss/metrics/weights_sdss.csv"
)

map42 = dict(zip(weights42["feature"], weights42["weight"]))
w42 = np.array([map42[c] for c in feature_cols])

score42 = weighted_geometric_score(X, w42)

idx42 = np.argpartition(
    score42,
    -top_k
)[-top_k:]

set42 = set(idx42.tolist())

rank42 = pd.Series(score42).rank(
    method="average"
).to_numpy()


OUT = Path("outputs/stability/sdss/bad_seed_reruns")
OUT.mkdir(parents=True, exist_ok=True)

rows = []
weight_rows = []

for seed in BAD_SEEDS:

    print("="*72)
    print("RERUN SEED:", seed)
    print("="*72)

    result = differential_evolution(
        objective,
        bounds=[(0, 1)] * n_features,
        seed=seed,
        maxiter=300,
        popsize=30,
        tol=1e-8,
        polish=True,
        workers=1,
        updating="immediate"
    )

    w = normalize_weights(result.x)
    score = weighted_geometric_score(X, w)

    idx = np.argpartition(
        score,
        -top_k
    )[-top_k:]

    selected = set(idx.tolist())

    intersection = len(selected & set42)
    union = len(selected | set42)

    jaccard = intersection / union

    rank = pd.Series(score).rank(
        method="average"
    ).to_numpy()

    rho = spearmanr(
        rank42,
        rank
    ).statistic

    row = {
        "seed": seed,
        "objective": float(-result.fun),
        "mean_top_score": float(np.mean(score[idx])),
        "effective_features": float(
            effective_number_of_features(w)
        ),
        "entropy": float(
            entropy_of_weights(w)
        ),
        "topk_overlap_n": intersection,
        "topk_jaccard_vs_seed42": float(jaccard),
        "rank_spearman_vs_seed42": float(rho),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
    }

    rows.append(row)

    for f, weight in zip(feature_cols, w):
        weight_rows.append({
            "seed": seed,
            "feature": f,
            "weight": float(weight)
        })

    print("objective:", row["objective"])
    print("mean_top_score:", row["mean_top_score"])
    print("N_eff:", row["effective_features"])
    print("entropy:", row["entropy"])
    print("Jaccard vs seed42:", row["topk_jaccard_vs_seed42"])
    print("Spearman vs seed42:", row["rank_spearman_vs_seed42"])
    print("nfev:", row["nfev"])
    print()
    print(
        pd.DataFrame({
            "feature": feature_cols,
            "weight": w
        })
        .sort_values("weight", ascending=False)
        .to_string(index=False)
    )
    print()

pd.DataFrame(rows).to_csv(
    OUT / "rerun_metrics.csv",
    index=False
)

pd.DataFrame(weight_rows).to_csv(
    OUT / "rerun_weights.csv",
    index=False
)

print("Saved to:", OUT)
