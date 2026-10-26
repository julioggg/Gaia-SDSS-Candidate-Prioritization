from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from scipy.optimize import differential_evolution
from scipy.stats import spearmanr


CONFIG = "configs/sdss_baseline_v2.yaml"

with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

raw = pd.read_csv(
    cfg["outputs"]["raw_csv"],
    dtype={
        "objid": str,
        "specobjid": str,
        "htmid": str,
    }
)

reference = pd.read_csv(
    cfg["outputs"]["reference_csv"],
    dtype={
        "objid": str,
        "specobjid": str,
        "class": str,
    }
)

baseline_features = pd.read_csv(
    cfg["outputs"]["processed_csv"]
)

with open(
    "data/processed/sdss_feature_columns_reference.json",
    "r",
    encoding="utf-8"
) as f:
    feature_cols = json.load(f)


# ==========================================================
# We reuse already-computed physical variables and only
# rebuild normalized feature columns using different
# clipping percentiles.
# ==========================================================

base = baseline_features.copy()

required = [
    "mean_phot_error",
    "color_distance",
    "log_redshift",
    "redshift_error",
    "local_density_knn",
    "neighbor_count_radius",
    "redshift_neighbor_count",
    "nn_sep_arcsec",
]

missing = [
    c for c in required
    if c not in base.columns
]

if missing:
    raise RuntimeError(
        f"Missing columns required for clipping sensitivity: {missing}"
    )


def robust_01_custom(s, qlow, qhigh):
    x = pd.to_numeric(s, errors="coerce")

    med = x.median()
    x = x.fillna(med)

    lo = x.quantile(qlow)
    hi = x.quantile(qhigh)

    x = x.clip(lo, hi)

    if hi <= lo:
        return pd.Series(
            np.zeros(len(x)),
            index=x.index
        )

    return (x - lo) / (hi - lo)


def inverse_01_custom(s, qlow, qhigh):
    return 1.0 - robust_01_custom(
        s,
        qlow,
        qhigh
    )


def build_features(df, qlow, qhigh):
    out = df.copy()

    out["f_photometric_quality"] = inverse_01_custom(
        out["mean_phot_error"],
        qlow,
        qhigh
    )

    out["f_color_rarity"] = robust_01_custom(
        out["color_distance"],
        qlow,
        qhigh
    )

    out["f_redshift_interest"] = robust_01_custom(
        out["log_redshift"],
        qlow,
        qhigh
    )

    out["f_redshift_precision"] = inverse_01_custom(
        out["redshift_error"],
        qlow,
        qhigh
    )

    out["f_local_density"] = (
        0.50 * robust_01_custom(
            out["local_density_knn"],
            qlow,
            qhigh
        )
        + 0.25 * robust_01_custom(
            out["neighbor_count_radius"],
            qlow,
            qhigh
        )
        + 0.25 * robust_01_custom(
            out["redshift_neighbor_count"],
            qlow,
            qhigh
        )
    )

    out["f_isolation"] = robust_01_custom(
        out["nn_sep_arcsec"],
        qlow,
        qhigh
    )

    return out


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


top_k = int(cfg["optimization"]["top_k"])
min_eff = float(
    cfg["optimization"]["min_effective_features"]
)

n_features = len(feature_cols)

CLIPS = [
    (0.005, 0.995, "0.5-99.5"),
    (0.010, 0.990, "1-99"),
    (0.025, 0.975, "2.5-97.5"),
    (0.050, 0.950, "5-95"),
]

SEEDS = [0, 1, 2]


# ==========================================================
# Baseline original 1-99
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

X_ref = base[feature_cols].to_numpy(float)

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

set_ref = set(idx_ref.tolist())


rows = []
weight_rows = []


for qlow, qhigh, label in CLIPS:

    print()
    print("=" * 80)
    print(f"CLIPPING: {label}")
    print("=" * 80)

    feat = build_features(
        base,
        qlow,
        qhigh
    )

    X = feat[
        feature_cols
    ].to_numpy(float)

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

        jaccard = (
            len(selected & set_ref)
            / len(selected | set_ref)
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

        rows.append({
            "clipping": label,
            "qlow": qlow,
            "qhigh": qhigh,
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
            "jaccard_vs_baseline_1_99":
                float(jaccard),
            "spearman_vs_baseline_1_99":
                float(rho),
            "nfev": int(
                result.nfev
            ),
        })

        for feature, weight in zip(
            feature_cols,
            w
        ):
            weight_rows.append({
                "clipping": label,
                "seed": seed,
                "feature": feature,
                "weight": float(weight),
            })


OUT = (
    Path("outputs")
    / "sensitivity"
    / "sdss"
    / "clipping"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

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
    .groupby("clipping")
    .agg(
        objective_mean=(
            "objective",
            "mean"
        ),
        objective_std=(
            "objective",
            "std"
        ),
        top_score_mean=(
            "mean_top_score",
            "mean"
        ),
        neff_mean=(
            "effective_features",
            "mean"
        ),
        entropy_mean=(
            "entropy",
            "mean"
        ),
        jaccard_mean=(
            "jaccard_vs_baseline_1_99",
            "mean"
        ),
        jaccard_min=(
            "jaccard_vs_baseline_1_99",
            "min"
        ),
        spearman_mean=(
            "spearman_vs_baseline_1_99",
            "mean"
        ),
        spearman_min=(
            "spearman_vs_baseline_1_99",
            "min"
        ),
    )
)

summary.to_csv(
    OUT / "summary.csv"
)


weight_summary = (
    weights
    .groupby([
        "clipping",
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
print("CLIPPING SENSITIVITY SUMMARY")
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
