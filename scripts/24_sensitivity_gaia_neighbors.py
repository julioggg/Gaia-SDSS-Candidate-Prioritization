from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from sklearn.neighbors import BallTree
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr


CONFIG = "configs/gaia_config_reference.yaml"

with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

targets_base = pd.read_csv(
    cfg["outputs"]["processed_csv"],
    dtype={"source_id": str}
)

reference = pd.read_csv(
    cfg["outputs"]["reference_csv"],
    dtype={"source_id": str}
)

with open(
    "data/processed/gaia_feature_columns_reference.json",
    "r",
    encoding="utf-8"
) as f:
    feature_cols = json.load(f)


ARCSEC_PER_RAD = (180.0 / np.pi) * 3600.0

MIN_PARALLAX_SNR = float(
    cfg["quality_filters"]["min_parallax_snr"]
)

MAX_RUWE = float(
    cfg["quality_filters"]["max_ruwe"]
)

MAX_VTAN = float(
    cfg["quality_filters"]["max_vtan_kms"]
)

TOP_K = int(
    cfg["optimization"]["top_k"]
)

MIN_EFF = float(
    cfg["optimization"]["min_effective_features"]
)

SEEDS = [0, 1, 2]

CONFIGS = [
    (60.0, 5, "r60_k5"),
    (120.0, 5, "r120_k5"),
    (240.0, 5, "r240_k5"),
    (120.0, 3, "r120_k3"),
    (120.0, 10, "r120_k10"),
]


def prepare_reference(df):
    df = df.copy()

    numeric = [
        "ra", "dec",
        "parallax", "parallax_error",
        "pmra", "pmdec",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "bp_rp", "ruwe",
    ]

    for c in numeric:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "ra", "dec",
            "parallax", "parallax_error",
            "phot_g_mean_mag",
            "bp_rp", "ruwe",
            "pmra", "pmdec",
        ]
    )

    df = df[
        (df["parallax"] > 0)
        &
        (df["parallax_error"] > 0)
    ].copy()

    df["pm_total"] = np.sqrt(
        df["pmra"]**2
        +
        df["pmdec"]**2
    )

    df["vtan_kms"] = (
        4.74047
        * df["pm_total"]
        / df["parallax"]
    )

    df["parallax_snr"] = (
        df["parallax"]
        / df["parallax_error"]
    )

    df = df[
        (df["parallax_snr"] >= MIN_PARALLAX_SNR)
        &
        (df["ruwe"] < MAX_RUWE)
        &
        (df["vtan_kms"] >= 0)
        &
        (df["vtan_kms"] <= MAX_VTAN)
    ].copy()

    return (
        df
        .drop_duplicates(
            subset=["source_id"]
        )
        .reset_index(drop=True)
    )


def robust_01(s):
    x = pd.to_numeric(
        s,
        errors="coerce"
    )

    x = x.fillna(x.median())

    lo = x.quantile(0.01)
    hi = x.quantile(0.99)

    x = x.clip(lo, hi)

    if hi <= lo:
        return pd.Series(
            np.zeros(len(x)),
            index=x.index
        )

    return (x - lo) / (hi - lo)


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
        -np.sum(w*np.log(w + 1e-12))
        / np.log(len(w))
    )


def effective_number_of_features(w):
    w = normalize_weights(w)
    return 1.0 / np.sum(w**2)


reference = prepare_reference(reference)

ref_coord = np.column_stack([
    np.deg2rad(
        reference["dec"].to_numpy(float)
    ),
    np.deg2rad(
        reference["ra"].to_numpy(float)
    ),
])

target_coord = np.column_stack([
    np.deg2rad(
        targets_base["dec"].to_numpy(float)
    ),
    np.deg2rad(
        targets_base["ra"].to_numpy(float)
    ),
])

ref_ids = reference["source_id"].to_numpy()
target_ids = targets_base["source_id"].to_numpy()

tree = BallTree(
    ref_coord,
    metric="haversine"
)


# ==========================================================
# Baseline reference ranking
# ==========================================================

bw = pd.read_csv(
    "outputs/baseline_reference/gaia/metrics/weights_gaia.csv"
)

wm = dict(
    zip(
        bw["feature"],
        bw["weight"]
    )
)

w_ref = np.array(
    [wm[c] for c in feature_cols]
)

X_ref = targets_base[
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
    -TOP_K
)[-TOP_K:]

set_ref = set(idx_ref.tolist())


rows = []
weight_rows = []
neighbor_rows = []


for radius_arcsec, kth, label in CONFIGS:

    print()
    print("="*82)
    print(
        f"GAIA NEIGHBOR CONFIG: "
        f"radius={radius_arcsec}\" | k={kth}"
    )
    print("="*82)

    df = targets_base.copy()

    query_k = max(
        kth + 3,
        13
    )

    dist, ind = tree.query(
        target_coord,
        k=query_k
    )

    nn_sep = np.empty(
        len(df)
    )

    kth_sep = np.empty(
        len(df)
    )

    for i in range(len(df)):

        idx = ind[i]
        dd = dist[i]

        keep = (
            ref_ids[idx]
            != target_ids[i]
        )

        dd = dd[keep]

        if len(dd) < kth:
            raise RuntimeError(
                f"Not enough neighbors row={i}"
            )

        nn_sep[i] = (
            dd[0]
            * ARCSEC_PER_RAD
        )

        kth_sep[i] = (
            dd[kth - 1]
            * ARCSEC_PER_RAD
        )

    df["nn_sep_arcsec"] = nn_sep
    df["kth_sep_arcsec"] = kth_sep

    kth_arcmin = (
        df["kth_sep_arcsec"]
        / 60.0
    )

    df["local_density_knn"] = (
        kth
        /
        (
            np.pi
            * np.maximum(
                kth_arcmin,
                1e-12
            )**2
        )
    )

    radius_rad = (
        radius_arcsec
        / ARCSEC_PER_RAD
    )

    neighbor_lists = tree.query_radius(
        target_coord,
        r=radius_rad
    )

    counts = np.empty(
        len(df),
        dtype=int
    )

    for i, idx in enumerate(
        neighbor_lists
    ):
        counts[i] = np.sum(
            ref_ids[idx]
            != target_ids[i]
        )

    df["neighbor_count_radius"] = counts

    # Rebuild only the features affected
    # by neighborhood definitions.
    df["f_local_density"] = (
        0.70
        * robust_01(
            df["local_density_knn"]
        )
        +
        0.30
        * robust_01(
            df["neighbor_count_radius"]
        )
    )

    df["f_isolation"] = robust_01(
        df["nn_sep_arcsec"]
    )

    X = df[
        feature_cols
    ].to_numpy(float)

    neighbor_rows.append({
        "configuration": label,
        "radius_arcsec": radius_arcsec,
        "kth_neighbor": kth,
        "percent_with_neighbor": float(
            100 * np.mean(counts > 0)
        ),
        "median_nn_arcsec": float(
            np.median(nn_sep)
        ),
        "median_kth_arcsec": float(
            np.median(kth_sep)
        ),
        "median_neighbor_count": float(
            np.median(counts)
        ),
        "median_local_density": float(
            np.median(
                df["local_density_knn"]
            )
        ),
    })

    def objective(w_raw):

        w = normalize_weights(w_raw)

        score = weighted_geometric_score(
            X,
            w
        )

        idx = np.argpartition(
            score,
            -TOP_K
        )[-TOP_K:]

        top_X = X[idx]
        top_scores = score[idx]

        mean_score = np.mean(
            top_scores
        )

        diversity = np.mean(
            np.std(
                top_X,
                axis=0
            )
        )

        entropy = entropy_of_weights(w)

        neff = effective_number_of_features(w)

        penalty = max(
            0.0,
            MIN_EFF - neff
        ) / MIN_EFF

        return -(
            0.55 * mean_score
            + 0.20 * diversity
            + 0.20 * entropy
            - 0.25 * penalty
        )

    for seed in SEEDS:

        print(f"  seed={seed}")

        result = differential_evolution(
            objective,
            bounds=[
                (0, 1)
            ] * len(feature_cols),
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
            -TOP_K
        )[-TOP_K:]

        selected = set(
            idx.tolist()
        )

        intersection = len(
            selected & set_ref
        )

        union = len(
            selected | set_ref
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
            "configuration": label,
            "radius_arcsec": radius_arcsec,
            "kth_neighbor": kth,
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
            "topk_overlap_n":
                intersection,
            "jaccard_vs_baseline": float(
                intersection / union
            ),
            "spearman_vs_baseline": float(
                rho
            ),
        })

        for feature, weight in zip(
            feature_cols,
            w
        ):
            weight_rows.append({
                "configuration": label,
                "radius_arcsec": radius_arcsec,
                "kth_neighbor": kth,
                "seed": seed,
                "feature": feature,
                "weight": float(weight),
            })


OUT = (
    Path("outputs")
    / "sensitivity_reference"
    / "gaia"
    / "neighbors"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


metrics = pd.DataFrame(rows)

weights = pd.DataFrame(
    weight_rows
)

neighbors = pd.DataFrame(
    neighbor_rows
)


metrics.to_csv(
    OUT / "metrics.csv",
    index=False
)

weights.to_csv(
    OUT / "weights.csv",
    index=False
)

neighbors.to_csv(
    OUT / "neighbor_diagnostics.csv",
    index=False
)


summary = (
    metrics
    .groupby("configuration")
    .agg(
        radius_arcsec=(
            "radius_arcsec",
            "first"
        ),
        kth_neighbor=(
            "kth_neighbor",
            "first"
        ),
        objective_mean=(
            "objective",
            "mean"
        ),
        objective_std=(
            "objective",
            "std"
        ),
        neff_mean=(
            "effective_features",
            "mean"
        ),
        entropy_mean=(
            "entropy",
            "mean"
        ),
        overlap_mean=(
            "topk_overlap_n",
            "mean"
        ),
        jaccard_mean=(
            "jaccard_vs_baseline",
            "mean"
        ),
        jaccard_min=(
            "jaccard_vs_baseline",
            "min"
        ),
        spearman_mean=(
            "spearman_vs_baseline",
            "mean"
        ),
        spearman_min=(
            "spearman_vs_baseline",
            "min"
        ),
    )
)


weight_summary = (
    weights
    .groupby([
        "configuration",
        "feature"
    ])["weight"]
    .agg([
        "mean",
        "std",
        "min",
        "max"
    ])
)


summary.to_csv(
    OUT / "summary.csv"
)

weight_summary.to_csv(
    OUT / "weight_summary.csv"
)


print()
print("="*100)
print("GAIA NEIGHBOR SENSITIVITY SUMMARY")
print("="*100)

print(
    summary.to_string()
)

print()
print("="*100)
print("NEIGHBOR DIAGNOSTICS")
print("="*100)

print(
    neighbors.to_string(
        index=False
    )
)

print()
print("="*100)
print("WEIGHT SUMMARY")
print("="*100)

print(
    weight_summary.to_string()
)

print()
print("Saved to:", OUT)
