from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from scipy.optimize import differential_evolution
from scipy.stats import spearmanr


CONFIG = "configs/sdss_baseline_v2.yaml"
SEEDS = [0, 1, 2]

with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

base = pd.read_csv(
    cfg["outputs"]["processed_csv"]
)

with open(
    "data/processed/sdss_feature_columns_reference.json",
    "r",
    encoding="utf-8"
) as f:
    feature_cols = json.load(f)


def robust_01_custom(s, qlow, qhigh):
    x = pd.to_numeric(
        s,
        errors="coerce"
    )

    x = x.fillna(
        x.median()
    )

    lo = x.quantile(qlow)
    hi = x.quantile(qhigh)

    x = x.clip(
        lo,
        hi
    )

    if hi <= lo:
        return pd.Series(
            np.zeros(len(x)),
            index=x.index
        )

    return (
        (x - lo)
        / (hi - lo)
    )


def inverse_01_custom(s, qlow, qhigh):
    return (
        1.0
        - robust_01_custom(
            s,
            qlow,
            qhigh
        )
    )


# ==========================================================
# Build 0.5-99.5% features
# ==========================================================

qlow = 0.005
qhigh = 0.995

feat = base.copy()

feat["f_photometric_quality"] = inverse_01_custom(
    feat["mean_phot_error"],
    qlow,
    qhigh
)

feat["f_color_rarity"] = robust_01_custom(
    feat["color_distance"],
    qlow,
    qhigh
)

feat["f_redshift_interest"] = robust_01_custom(
    feat["log_redshift"],
    qlow,
    qhigh
)

feat["f_redshift_precision"] = inverse_01_custom(
    feat["redshift_error"],
    qlow,
    qhigh
)

feat["f_local_density"] = (
    0.50 * robust_01_custom(
        feat["local_density_knn"],
        qlow,
        qhigh
    )
    + 0.25 * robust_01_custom(
        feat["neighbor_count_radius"],
        qlow,
        qhigh
    )
    + 0.25 * robust_01_custom(
        feat["redshift_neighbor_count"],
        qlow,
        qhigh
    )
)

feat["f_isolation"] = robust_01_custom(
    feat["nn_sep_arcsec"],
    qlow,
    qhigh
)

X = feat[
    feature_cols
].to_numpy(float)

top_k = int(
    cfg["optimization"]["top_k"]
)

min_eff = float(
    cfg["optimization"]["min_effective_features"]
)

n_features = len(
    feature_cols
)


def normalize_weights(w):
    w = np.clip(
        np.asarray(w, float),
        1e-12,
        None
    )
    return w / w.sum()


def weighted_geometric_score(X, w):
    w = normalize_weights(w)

    return np.exp(
        np.sum(
            w
            * np.log(
                np.clip(
                    X,
                    1e-12,
                    1.0
                )
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


def objective(w_raw):
    w = normalize_weights(w_raw)

    score = weighted_geometric_score(
        X,
        w
    )

    idx = np.argpartition(
        score,
        -top_k
    )[-top_k:]

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


# ==========================================================
# Reference = original 1-99 baseline
# ==========================================================

weights_ref = pd.read_csv(
    "outputs/baseline_v2/sdss/metrics/weights_sdss.csv"
)

wm = dict(
    zip(
        weights_ref["feature"],
        weights_ref["weight"]
    )
)

w_ref = np.array(
    [wm[c] for c in feature_cols]
)

X_ref = base[
    feature_cols
].to_numpy(float)

score_ref = weighted_geometric_score(
    X_ref,
    w_ref
)

rank_ref = pd.Series(
    score_ref
).rank(
    method="average"
).to_numpy()

idx_ref = np.argpartition(
    score_ref,
    -top_k
)[-top_k:]

set_ref = set(
    idx_ref.tolist()
)


OUT = Path(
    "outputs/sensitivity/sdss/clipping/rerun_005"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

rows = []
weight_rows = []


for seed in SEEDS:

    print()
    print("=" * 76)
    print(
        f"0.5-99.5 HIGH-BUDGET RERUN | seed={seed}"
    )
    print("=" * 76)

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
        selected & set_ref
    )

    union = len(
        selected | set_ref
    )

    jaccard = (
        intersection / union
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

    row = {
        "seed": seed,
        "objective": float(
            -result.fun
        ),
        "mean_top_score": float(
            np.mean(score[idx])
        ),
        "effective_features": float(
            effective_number_of_features(w)
        ),
        "entropy": float(
            entropy_of_weights(w)
        ),
        "topk_overlap_n": intersection,
        "jaccard_vs_baseline_1_99":
            float(jaccard),
        "spearman_vs_baseline_1_99":
            float(rho),
        "success": bool(
            result.success
        ),
        "nfev": int(
            result.nfev
        ),
        "nit": int(
            result.nit
        ),
    }

    rows.append(row)

    for feature, weight in zip(
        feature_cols,
        w
    ):
        weight_rows.append({
            "seed": seed,
            "feature": feature,
            "weight": float(weight)
        })

    print()
    print("objective:", row["objective"])
    print(
        "mean_top_score:",
        row["mean_top_score"]
    )
    print(
        "N_eff:",
        row["effective_features"]
    )
    print(
        "entropy:",
        row["entropy"]
    )
    print(
        "Jaccard vs baseline:",
        row["jaccard_vs_baseline_1_99"]
    )
    print(
        "Spearman vs baseline:",
        row["spearman_vs_baseline_1_99"]
    )
    print(
        "nfev:",
        row["nfev"]
    )

    print()
    print(
        pd.DataFrame({
            "feature": feature_cols,
            "weight": w
        })
        .sort_values(
            "weight",
            ascending=False
        )
        .to_string(
            index=False
        )
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


print()
print("=" * 90)
print("FINAL HIGH-BUDGET SUMMARY")
print("=" * 90)

print(
    metrics.to_string(
        index=False
    )
)

print()
print(
    weights
    .groupby("feature")["weight"]
    .agg([
        "mean",
        "std",
        "min",
        "max"
    ])
    .sort_values(
        "mean",
        ascending=False
    )
    .to_string()
)

print()
print("Saved to:", OUT)
