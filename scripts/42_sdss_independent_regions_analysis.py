from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

from reference_neighbors import add_reference_neighbor_features


ROOT = Path.home() / "universe_revision" / "revision_v2"

DATA_ROOT = (
    ROOT
    / "data/independent_regions/sdss"
)

OUT = (
    ROOT
    / "outputs/independent_regions/sdss/analysis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


REGIONS = [
    "region_A",
    "region_G",
]


FEATURE_COLS = [
    "f_photometric_quality",
    "f_color_rarity",
    "f_redshift_interest",
    "f_redshift_precision",
    "f_local_density",
    "f_isolation",
]


RADIUS_ARCSEC = 300.0
KTH_NEIGHBOR = 5
REDSHIFT_RADIUS_ARCSEC = 600.0
REDSHIFT_DZ = 0.05

TOP_K = 1000
MIN_EFF = 3.0

SEEDS = [
    0,
    1,
    2,
    42,
]


# ==========================================================
# Helpers
# ==========================================================

def robust_01(s):

    x = pd.to_numeric(
        s,
        errors="coerce"
    ).astype(float)

    med = np.nanmedian(x)

    x = np.where(
        np.isfinite(x),
        x,
        med
    )

    lo = np.nanquantile(
        x,
        0.01
    )

    hi = np.nanquantile(
        x,
        0.99
    )

    x = np.clip(
        x,
        lo,
        hi
    )

    if hi <= lo:

        return np.zeros(
            len(x),
            dtype=float
        )

    return (
        (x - lo)
        /
        (hi - lo)
    )


def inverse_01(s):

    return (
        1.0
        -
        robust_01(s)
    )


def normalize_weights(w):

    w = np.clip(
        np.asarray(w, float),
        1e-12,
        None
    )

    return (
        w
        /
        w.sum()
    )


def weighted_geometric_score(
    X,
    w
):

    w = normalize_weights(
        w
    )

    return np.exp(
        np.sum(
            w[None, :]
            *
            np.log(
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

    w = normalize_weights(
        w
    )

    return (
        -np.sum(
            w
            *
            np.log(
                w + 1e-12
            )
        )
        /
        np.log(
            len(w)
        )
    )


def effective_number_of_features(w):

    w = normalize_weights(
        w
    )

    return (
        1.0
        /
        np.sum(
            w**2
        )
    )


def prepare_static_features(df):

    df = df.copy()


    numeric = [
        "ra",
        "dec",
        "u",
        "g",
        "r",
        "i",
        "z",
        "err_u",
        "err_g",
        "err_r",
        "err_i",
        "err_z",
        "redshift",
        "redshift_error",
    ]


    for c in numeric:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )


    df["u_g"] = (
        df["u"]
        -
        df["g"]
    )

    df["g_r"] = (
        df["g"]
        -
        df["r"]
    )

    df["r_i"] = (
        df["r"]
        -
        df["i"]
    )

    df["i_z"] = (
        df["i"]
        -
        df["z"]
    )


    err_cols = [
        "err_u",
        "err_g",
        "err_r",
        "err_i",
        "err_z",
    ]


    df["mean_phot_error"] = (
        df[
            err_cols
        ]
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .mean(
            axis=1
        )
    )


    color_cols = [
        "u_g",
        "g_r",
        "r_i",
        "i_z",
    ]


    color_distance = np.zeros(
        len(df)
    )


    for c in color_cols:

        color_distance += (
            df[c]
            -
            df[c].median()
        ) ** 2


    df["color_distance"] = (
        np.sqrt(
            color_distance
        )
    )


    df["log_redshift"] = (
        np.log1p(
            df["redshift"]
        )
    )


    return df


def build_features(df):

    df = df.copy()


    df[
        "f_photometric_quality"
    ] = inverse_01(
        df[
            "mean_phot_error"
        ]
    )


    df[
        "f_color_rarity"
    ] = robust_01(
        df[
            "color_distance"
        ]
    )


    df[
        "f_redshift_interest"
    ] = robust_01(
        df[
            "log_redshift"
        ]
    )


    df[
        "f_redshift_precision"
    ] = inverse_01(
        df[
            "redshift_error"
        ]
    )


    df[
        "f_local_density"
    ] = (
        0.50
        *
        robust_01(
            df[
                "local_density_knn"
            ]
        )
        +
        0.25
        *
        robust_01(
            df[
                "neighbor_count_radius"
            ]
        )
        +
        0.25
        *
        robust_01(
            df[
                "redshift_neighbor_count"
            ]
        )
    )


    df[
        "f_isolation"
    ] = robust_01(
        df[
            "nn_sep_arcsec"
        ]
    )


    df = df.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )


    df = (
        df
        .dropna(
            subset=FEATURE_COLS
        )
        .reset_index(drop=True)
    )


    return df


def optimize(
    df,
    seed,
):

    X = (
        df[
            FEATURE_COLS
        ]
        .to_numpy(float)
    )


    def objective(w_raw):

        w = normalize_weights(
            w_raw
        )


        score = weighted_geometric_score(
            X,
            w
        )


        K = min(
            TOP_K,
            len(score)
        )


        idx = np.argpartition(
            score,
            -K
        )[-K:]


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


        entropy = entropy_of_weights(
            w
        )


        neff = effective_number_of_features(
            w
        )


        penalty = (
            max(
                0.0,
                MIN_EFF - neff
            )
            /
            MIN_EFF
        )


        return -(
            0.55 * mean_score
            +
            0.20 * diversity
            +
            0.20 * entropy
            -
            0.25 * penalty
        )


    result = differential_evolution(
        objective,
        bounds=[
            (0.0, 1.0)
        ] * len(FEATURE_COLS),
        seed=seed,
        maxiter=300,
        popsize=30,
        tol=1e-7,
        polish=True,
        workers=1,
    )


    w = normalize_weights(
        result.x
    )


    score = weighted_geometric_score(
        X,
        w
    )


    return {
        "weights":
            w,

        "score":
            score,

        "objective":
            float(
                -result.fun
            ),

        "neff":
            float(
                effective_number_of_features(
                    w
                )
            ),

        "entropy":
            float(
                entropy_of_weights(
                    w
                )
            ),
    }


# ==========================================================
# Baseline original weights
# ==========================================================

baseline_weights = np.array([
    0.334045,
    0.000010,
    0.000003,
    0.332235,
    0.000009,
    0.333691,
], dtype=float)

baseline_weights = normalize_weights(
    baseline_weights
)


# ==========================================================
# Main
# ==========================================================

metric_rows = []
weight_rows = []
stability_rows = []
diagnostic_rows = []
transfer_rows = []


for region in REGIONS:

    print()
    print("=" * 130)
    print("REGION:", region)
    print("=" * 130)


    region_dir = (
        DATA_ROOT
        /
        region
    )


    targets = pd.read_csv(
        region_dir
        /
        "raw_frozen.csv",
        dtype={
            "objid": str,
            "specobjid": str,
            "class": str,
        }
    )


    reference = pd.read_csv(
        region_dir
        /
        "reference_buffered.csv",
        dtype={
            "objid": str,
            "specobjid": str,
            "class": str,
        }
    )


    print(
        "Targets:",
        f"{len(targets):,}"
    )

    print(
        "Reference:",
        f"{len(reference):,}"
    )


    targets = prepare_static_features(
        targets
    )

    reference = prepare_static_features(
        reference
    )


    target_ids = set(
        targets["objid"]
    )

    reference_ids = set(
        reference["objid"]
    )


    missing = (
        target_ids
        -
        reference_ids
    )


    if missing:

        raise RuntimeError(
            f"{region}: "
            f"{len(missing)} target IDs "
            "missing from reference."
        )


    print(
        "Adding reference-based neighborhood features..."
    )


    targets = add_reference_neighbor_features(
        targets=targets,
        reference=reference,
        radius_arcsec=
            RADIUS_ARCSEC,
        kth_neighbor=
            KTH_NEIGHBOR,
        redshift_radius_arcsec=
            REDSHIFT_RADIUS_ARCSEC,
        dz=
            REDSHIFT_DZ,
        batch_size=2000,
    )


    targets = build_features(
        targets
    )


    print(
        "Final feature rows:",
        f"{len(targets):,}"
    )


    class_counts = (
        targets[
            "class"
        ]
        .value_counts()
    )


    diagnostic_rows.append({

        "region":
            region,

        "n_targets":
            len(targets),

        "n_reference":
            len(reference),

        "n_galaxy":
            int(
                class_counts.get(
                    "GALAXY",
                    0
                )
            ),

        "n_qso":
            int(
                class_counts.get(
                    "QSO",
                    0
                )
            ),

        "qso_fraction":
            float(
                np.mean(
                    targets[
                        "class"
                    ]
                    ==
                    "QSO"
                )
            ),

        "percent_with_neighbor_300":
            float(
                100
                *
                np.mean(
                    targets[
                        "neighbor_count_radius"
                    ]
                    >
                    0
                )
            ),

        "percent_with_redshift_neighbor":
            float(
                100
                *
                np.mean(
                    targets[
                        "redshift_neighbor_count"
                    ]
                    >
                    0
                )
            ),

        "median_nn_arcsec":
            float(
                targets[
                    "nn_sep_arcsec"
                ].median()
            ),

        "median_kth_arcsec":
            float(
                targets[
                    "kth_sep_arcsec"
                ].median()
            ),

        "median_neighbor_count":
            float(
                targets[
                    "neighbor_count_radius"
                ].median()
            ),

        "median_redshift_count":
            float(
                targets[
                    "redshift_neighbor_count"
                ].median()
            ),
    })


    # ======================================================
    # Transfer original weights
    # ======================================================

    X = (
        targets[
            FEATURE_COLS
        ]
        .to_numpy(float)
    )


    transfer_score = (
        weighted_geometric_score(
            X,
            baseline_weights
        )
    )


    idx = np.argpartition(
        transfer_score,
        -TOP_K
    )[-TOP_K:]


    transfer_rows.append({

        "region":
            region,

        "n":
            len(targets),

        "mean_score_all":
            float(
                np.mean(
                    transfer_score
                )
            ),

        "mean_score_top1000":
            float(
                np.mean(
                    transfer_score[
                        idx
                    ]
                )
            ),

        "median_score_all":
            float(
                np.median(
                    transfer_score
                )
            ),
    })


    # ======================================================
    # Reoptimization
    # ======================================================

    rankings = {}


    for seed in SEEDS:

        print(
            "Optimizing seed",
            seed
        )


        res = optimize(
            targets,
            seed
        )


        metric_rows.append({

            "region":
                region,

            "seed":
                seed,

            "objective":
                res[
                    "objective"
                ],

            "neff":
                res[
                    "neff"
                ],

            "entropy":
                res[
                    "entropy"
                ],
        })


        for feature, weight in zip(
            FEATURE_COLS,
            res[
                "weights"
            ]
        ):

            weight_rows.append({

                "region":
                    region,

                "seed":
                    seed,

                "feature":
                    feature,

                "weight":
                    float(
                        weight
                    ),
            })


        rank = pd.DataFrame({
            "objid":
                targets[
                    "objid"
                ].astype(str),

            "score":
                res[
                    "score"
                ],
        })


        rank = (
            rank
            .sort_values(
                [
                    "score",
                    "objid",
                ],
                ascending=[
                    False,
                    True,
                ],
                kind="mergesort"
            )
            .reset_index(drop=True)
        )


        rankings[
            seed
        ] = rank


    # ======================================================
    # Seed stability
    # ======================================================

    ref_seed = 42


    ref_score = (
        rankings[
            ref_seed
        ]
        .set_index(
            "objid"
        )[
            "score"
        ]
    )


    ref_top = set(
        rankings[
            ref_seed
        ]
        .head(
            TOP_K
        )[
            "objid"
        ]
    )


    for seed in SEEDS:

        score = (
            rankings[
                seed
            ]
            .set_index(
                "objid"
            )[
                "score"
            ]
        )


        rho = spearmanr(
            ref_score,
            score
        ).statistic


        top = set(
            rankings[
                seed
            ]
            .head(
                TOP_K
            )[
                "objid"
            ]
        )


        overlap = len(
            ref_top
            &
            top
        )


        union = len(
            ref_top
            |
            top
        )


        stability_rows.append({

            "region":
                region,

            "seed":
                seed,

            "spearman_vs_seed42":
                float(
                    rho
                ),

            "top1000_overlap_vs_seed42":
                overlap,

            "top1000_jaccard_vs_seed42":
                float(
                    overlap
                    /
                    union
                ),
        })


# ==========================================================
# Summaries
# ==========================================================

metrics = pd.DataFrame(
    metric_rows
)

weights = pd.DataFrame(
    weight_rows
)

stability = pd.DataFrame(
    stability_rows
)

diagnostics = pd.DataFrame(
    diagnostic_rows
)

transfer = pd.DataFrame(
    transfer_rows
)


metric_summary = (
    metrics
    .groupby(
        "region"
    )
    .agg(
        objective_mean=(
            "objective",
            "mean"
        ),
        objective_std=(
            "objective",
            "std"
        ),
        neff_mean=(
            "neff",
            "mean"
        ),
        neff_std=(
            "neff",
            "std"
        ),
        entropy_mean=(
            "entropy",
            "mean"
        ),
    )
    .reset_index()
)


weight_summary = (
    weights
    .groupby(
        [
            "region",
            "feature",
        ]
    )[
        "weight"
    ]
    .agg(
        [
            "mean",
            "std",
        ]
    )
    .reset_index()
)


stability_summary = (
    stability
    .groupby(
        "region"
    )
    .agg(
        spearman_mean=(
            "spearman_vs_seed42",
            "mean"
        ),
        spearman_min=(
            "spearman_vs_seed42",
            "min"
        ),
        overlap_mean=(
            "top1000_overlap_vs_seed42",
            "mean"
        ),
        overlap_min=(
            "top1000_overlap_vs_seed42",
            "min"
        ),
        jaccard_mean=(
            "top1000_jaccard_vs_seed42",
            "mean"
        ),
        jaccard_min=(
            "top1000_jaccard_vs_seed42",
            "min"
        ),
    )
    .reset_index()
)


pivot = (
    weight_summary
    .pivot(
        index="feature",
        columns="region",
        values="mean"
    )
    .reset_index()
)


pivot[
    "baseline_original"
] = [
    baseline_weights[
        FEATURE_COLS.index(f)
    ]
    for f in pivot[
        "feature"
    ]
]


def cosine(a, b):

    a = np.asarray(
        a,
        float
    )

    b = np.asarray(
        b,
        float
    )

    return float(
        np.dot(
            a,
            b
        )
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )
    )


wA = (
    weight_summary[
        weight_summary[
            "region"
        ]
        ==
        "region_A"
    ]
    .set_index(
        "feature"
    )
    .loc[
        FEATURE_COLS,
        "mean"
    ]
    .to_numpy(float)
)


wG = (
    weight_summary[
        weight_summary[
            "region"
        ]
        ==
        "region_G"
    ]
    .set_index(
        "feature"
    )
    .loc[
        FEATURE_COLS,
        "mean"
    ]
    .to_numpy(float)
)


similarity = pd.DataFrame([
    {
        "comparison":
            "original_vs_region_A",

        "cosine_weight_similarity":
            cosine(
                baseline_weights,
                wA
            ),

        "l1_weight_distance":
            float(
                np.sum(
                    np.abs(
                        baseline_weights
                        -
                        wA
                    )
                )
            ),
    },
    {
        "comparison":
            "original_vs_region_G",

        "cosine_weight_similarity":
            cosine(
                baseline_weights,
                wG
            ),

        "l1_weight_distance":
            float(
                np.sum(
                    np.abs(
                        baseline_weights
                        -
                        wG
                    )
                )
            ),
    },
    {
        "comparison":
            "region_A_vs_region_G",

        "cosine_weight_similarity":
            cosine(
                wA,
                wG
            ),

        "l1_weight_distance":
            float(
                np.sum(
                    np.abs(
                        wA
                        -
                        wG
                    )
                )
            ),
    },
])


# ==========================================================
# Save
# ==========================================================

metrics.to_csv(
    OUT / "metrics.csv",
    index=False
)

weights.to_csv(
    OUT / "weights.csv",
    index=False
)

stability.to_csv(
    OUT / "seed_stability.csv",
    index=False
)

diagnostics.to_csv(
    OUT / "diagnostics.csv",
    index=False
)

transfer.to_csv(
    OUT / "baseline_weight_transfer.csv",
    index=False
)

metric_summary.to_csv(
    OUT / "metric_summary.csv",
    index=False
)

weight_summary.to_csv(
    OUT / "weight_summary.csv",
    index=False
)

stability_summary.to_csv(
    OUT / "stability_summary.csv",
    index=False
)

pivot.to_csv(
    OUT / "cross_region_weights.csv",
    index=False
)

similarity.to_csv(
    OUT / "weight_similarity.csv",
    index=False
)


# ==========================================================
# Print
# ==========================================================

print()
print("=" * 160)
print(
    "SDSS INDEPENDENT-REGION METRIC SUMMARY"
)
print("=" * 160)

print(
    metric_summary.to_string(
        index=False
    )
)


print()
print("=" * 170)
print(
    "SDSS INDEPENDENT-REGION WEIGHTS"
)
print("=" * 170)

print(
    pivot.to_string(
        index=False
    )
)


print()
print("=" * 130)
print(
    "SDSS INDEPENDENT-REGION WEIGHT SIMILARITY"
)
print("=" * 130)

print(
    similarity.to_string(
        index=False
    )
)


print()
print("=" * 150)
print(
    "SDSS INDEPENDENT-REGION SEED STABILITY"
)
print("=" * 150)

print(
    stability_summary.to_string(
        index=False
    )
)


print()
print("=" * 170)
print(
    "SDSS INDEPENDENT-REGION DIAGNOSTICS"
)
print("=" * 170)

print(
    diagnostics.to_string(
        index=False
    )
)


print()
print("=" * 130)
print(
    "ORIGINAL-WEIGHT TRANSFER"
)
print("=" * 130)

print(
    transfer.to_string(
        index=False
    )
)


print()
print(
    "Saved:",
    OUT
)

print("DONE.")
