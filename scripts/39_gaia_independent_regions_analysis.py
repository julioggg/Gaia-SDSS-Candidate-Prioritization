from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.neighbors import BallTree
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr


ROOT = Path.home() / "universe_revision" / "revision_v2"

CONFIG = (
    ROOT
    / "configs/gaia_config_reference.yaml"
)

with open(
    CONFIG,
    "r",
    encoding="utf-8"
) as f:
    cfg = yaml.safe_load(f)


DATA_ROOT = (
    ROOT
    / "data/independent_regions/gaia"
)

OUT = (
    ROOT
    / "outputs/independent_regions/gaia"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


FIELDS = [
    "field_C",
    "field_D",
]


FEATURE_COLS = [
    "f_astrometric_quality",
    "f_photometric_color_rarity",
    "f_absolute_magnitude_rarity",
    "f_kinematic_interest",
    "f_local_density",
    "f_isolation",
]


ARCSEC_PER_RAD = (
    180.0 / np.pi
) * 3600.0


RADIUS_ARCSEC = float(
    cfg["match_internal"]["radius_arcsec"]
)

KTH_NEIGHBOR = int(
    cfg["match_internal"]["kth_neighbor"]
)

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


def coords_rad(df):

    return np.column_stack([
        np.deg2rad(
            df["dec"].to_numpy(float)
        ),
        np.deg2rad(
            df["ra"].to_numpy(float)
        ),
    ])


def prepare_gaia(df):

    df = df.copy()

    numeric = [
        "ra",
        "dec",
        "parallax",
        "parallax_error",
        "pmra",
        "pmdec",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "bp_rp",
        "ruwe",
        "random_index",
    ]


    for c in numeric:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )


    df = df.dropna(
        subset=[
            "ra",
            "dec",
            "parallax",
            "parallax_error",
            "phot_g_mean_mag",
            "bp_rp",
            "ruwe",
            "pmra",
            "pmdec",
        ]
    )


    df = df[
        (df["parallax"] > 0)
        &
        (df["parallax_error"] > 0)
    ].copy()


    df["abs_g_mag"] = (
        df["phot_g_mean_mag"]
        +
        5.0
        *
        np.log10(
            df["parallax"]
        )
        -
        10.0
    )


    df["pm_total"] = np.sqrt(
        df["pmra"]**2
        +
        df["pmdec"]**2
    )


    df["vtan_kms"] = (
        4.74047
        *
        df["pm_total"]
        /
        df["parallax"]
    )


    df["parallax_snr"] = (
        df["parallax"]
        /
        df["parallax_error"]
    )


    df["log_vtan"] = np.log1p(
        df["vtan_kms"]
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
            subset=["source_id"],
            keep="first"
        )
        .reset_index(drop=True)
    )


def add_neighbor_features(
    targets,
    reference,
):

    targets = targets.copy()


    target_coord = coords_rad(
        targets
    )

    ref_coord = coords_rad(
        reference
    )


    target_ids = (
        targets["source_id"]
        .astype(str)
        .to_numpy()
    )

    ref_ids = (
        reference["source_id"]
        .astype(str)
        .to_numpy()
    )


    tree = BallTree(
        ref_coord,
        metric="haversine"
    )


    query_k = max(
        KTH_NEIGHBOR + 3,
        13
    )


    dist, ind = tree.query(
        target_coord,
        k=query_k
    )


    nn_sep = np.empty(
        len(targets)
    )

    kth_sep = np.empty(
        len(targets)
    )


    for i in range(
        len(targets)
    ):

        idx = ind[i]
        dd = dist[i]

        keep = (
            ref_ids[idx]
            !=
            target_ids[i]
        )

        dd = dd[keep]


        if len(dd) < KTH_NEIGHBOR:

            raise RuntimeError(
                f"Not enough neighbors "
                f"for row {i}"
            )


        nn_sep[i] = (
            dd[0]
            *
            ARCSEC_PER_RAD
        )

        kth_sep[i] = (
            dd[
                KTH_NEIGHBOR - 1
            ]
            *
            ARCSEC_PER_RAD
        )


    targets[
        "nn_sep_arcsec"
    ] = nn_sep

    targets[
        "kth_sep_arcsec"
    ] = kth_sep


    kth_arcmin = (
        targets[
            "kth_sep_arcsec"
        ]
        /
        60.0
    )


    targets[
        "local_density_knn"
    ] = (
        KTH_NEIGHBOR
        /
        (
            np.pi
            *
            np.maximum(
                kth_arcmin,
                1e-12
            )**2
        )
    )


    radius_rad = (
        RADIUS_ARCSEC
        /
        ARCSEC_PER_RAD
    )


    neighbor_lists = (
        tree.query_radius(
            target_coord,
            r=radius_rad
        )
    )


    counts = np.empty(
        len(targets),
        dtype=int
    )


    for i, idx in enumerate(
        neighbor_lists
    ):

        counts[i] = np.sum(
            ref_ids[idx]
            !=
            target_ids[i]
        )


    targets[
        "neighbor_count_radius"
    ] = counts


    return targets


def build_features(df):

    df = df.copy()


    df[
        "f_astrometric_quality"
    ] = (
        0.50
        *
        inverse_01(
            df["ruwe"]
        )
        +
        0.50
        *
        robust_01(
            df["parallax_snr"]
        )
    )


    color_med = (
        df["bp_rp"]
        .median()
    )

    df[
        "f_photometric_color_rarity"
    ] = robust_01(
        np.abs(
            df["bp_rp"]
            -
            color_med
        )
    )


    mag_med = (
        df["abs_g_mag"]
        .median()
    )

    df[
        "f_absolute_magnitude_rarity"
    ] = robust_01(
        np.abs(
            df["abs_g_mag"]
            -
            mag_med
        )
    )


    df[
        "f_kinematic_interest"
    ] = robust_01(
        df["log_vtan"]
    )


    df[
        "f_local_density"
    ] = (
        0.70
        *
        robust_01(
            df[
                "local_density_knn"
            ]
        )
        +
        0.30
        *
        robust_01(
            df[
                "neighbor_count_radius"
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

        mean_score = np.mean(
            score[idx]
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

        neff = (
            effective_number_of_features(
                w
            )
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
        "objective":
            float(
                -result.fun
            ),

        "weights":
            w,

        "score":
            score,

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
# Baseline weights from original field
# ==========================================================

baseline_weights_df = pd.read_csv(
    ROOT
    / "outputs/baseline_reference/gaia/metrics/weights_gaia.csv"
)


baseline_weight_map = dict(
    zip(
        baseline_weights_df[
            "feature"
        ],
        baseline_weights_df[
            "weight"
        ]
    )
)


baseline_weights = np.array([
    baseline_weight_map[
        f
    ]
    for f in FEATURE_COLS
])


baseline_weights = normalize_weights(
    baseline_weights
)


# ==========================================================
# Main
# ==========================================================

metrics_rows = []
weight_rows = []
diagnostic_rows = []
stability_rows = []
baseline_transfer_rows = []


for field in FIELDS:

    print()
    print("=" * 120)
    print("FIELD:", field)
    print("=" * 120)


    field_dir = (
        DATA_ROOT
        /
        field
    )


    targets = pd.read_csv(
        field_dir
        /
        "raw_frozen.csv",
        dtype={
            "source_id": str
        }
    )


    reference = pd.read_csv(
        field_dir
        /
        "reference_buffered.csv",
        dtype={
            "source_id": str
        }
    )


    targets = prepare_gaia(
        targets
    )

    reference = prepare_gaia(
        reference
    )


    print(
        "Targets after scientific filters:",
        f"{len(targets):,}"
    )

    print(
        "Reference after scientific filters:",
        f"{len(reference):,}"
    )


    target_ids = set(
        targets["source_id"]
    )

    reference_ids = set(
        reference["source_id"]
    )

    missing = (
        target_ids
        -
        reference_ids
    )


    if missing:

        raise RuntimeError(
            f"{field}: "
            f"{len(missing)} filtered targets "
            "missing from filtered reference."
        )


    print(
        "Adding neighborhood features..."
    )


    targets = add_neighbor_features(
        targets,
        reference
    )


    targets = build_features(
        targets
    )


    print(
        "Final feature rows:",
        f"{len(targets):,}"
    )


    diagnostic_rows.append({

        "field":
            field,

        "n_targets":
            len(targets),

        "n_reference":
            len(reference),

        "median_parallax_snr":
            float(
                targets[
                    "parallax_snr"
                ].median()
            ),

        "median_ruwe":
            float(
                targets[
                    "ruwe"
                ].median()
            ),

        "median_vtan_kms":
            float(
                targets[
                    "vtan_kms"
                ].median()
            ),

        "percent_with_neighbor_120":
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

        "median_density":
            float(
                targets[
                    "local_density_knn"
                ].median()
            ),
    })


    # ======================================================
    # Transfer original-field weights without reoptimization
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


    K = min(
        TOP_K,
        len(targets)
    )


    transfer_idx = np.argpartition(
        transfer_score,
        -K
    )[-K:]


    baseline_transfer_rows.append({

        "field":
            field,

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
                        transfer_idx
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


        metrics_rows.append({

            "field":
                field,

            "seed":
                seed,

            "n":
                len(targets),

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


        for f, w in zip(
            FEATURE_COLS,
            res["weights"]
        ):

            weight_rows.append({

                "field":
                    field,

                "seed":
                    seed,

                "feature":
                    f,

                "weight":
                    float(w),
            })


        ranking = pd.DataFrame({
            "source_id":
                targets[
                    "source_id"
                ].astype(str),

            "score":
                res[
                    "score"
                ],
        })


        ranking = (
            ranking
            .sort_values(
                [
                    "score",
                    "source_id",
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
        ] = ranking


    # ======================================================
    # Within-field seed stability
    # ======================================================

    ref_seed = 42

    ref_rank = (
        rankings[
            ref_seed
        ]
        .set_index(
            "source_id"
        )[
            "score"
        ]
    )


    ref_top = set(
        rankings[
            ref_seed
        ]
        .head(
            min(
                TOP_K,
                len(
                    rankings[
                        ref_seed
                    ]
                )
            )
        )[
            "source_id"
        ]
    )


    for seed in SEEDS:

        rank = (
            rankings[
                seed
            ]
            .set_index(
                "source_id"
            )[
                "score"
            ]
        )


        rho = spearmanr(
            ref_rank,
            rank
        ).statistic


        top = set(
            rankings[
                seed
            ]
            .head(
                min(
                    TOP_K,
                    len(
                        rankings[
                            seed
                        ]
                    )
                )
            )[
                "source_id"
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

            "field":
                field,

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
# Tables
# ==========================================================

metrics = pd.DataFrame(
    metrics_rows
)

weights = pd.DataFrame(
    weight_rows
)

diagnostics = pd.DataFrame(
    diagnostic_rows
)

stability = pd.DataFrame(
    stability_rows
)

transfer = pd.DataFrame(
    baseline_transfer_rows
)


metric_summary = (
    metrics
    .groupby(
        "field"
    )
    .agg(
        n=(
            "n",
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
            "field",
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
        "field"
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


# ==========================================================
# Cross-region weight comparison
# ==========================================================

pivot_weights = (
    weight_summary
    .pivot(
        index="feature",
        columns="field",
        values="mean"
    )
    .reset_index()
)


pivot_weights[
    "baseline_original"
] = [
    baseline_weight_map[
        f
    ]
    for f in pivot_weights[
        "feature"
    ]
]


# Cosine similarity of mean weight vectors.
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


w_original = np.array([
    baseline_weight_map[f]
    for f in FEATURE_COLS
])


similarity_rows = []


for field in FIELDS:

    wf = (
        weight_summary[
            weight_summary[
                "field"
            ]
            ==
            field
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


    similarity_rows.append({

        "comparison":
            f"original_vs_{field}",

        "cosine_weight_similarity":
            cosine(
                w_original,
                wf
            ),

        "l1_weight_distance":
            float(
                np.sum(
                    np.abs(
                        w_original
                        -
                        wf
                    )
                )
            ),
    })


wC = (
    weight_summary[
        weight_summary["field"]
        ==
        "field_C"
    ]
    .set_index("feature")
    .loc[
        FEATURE_COLS,
        "mean"
    ]
    .to_numpy(float)
)


wD = (
    weight_summary[
        weight_summary["field"]
        ==
        "field_D"
    ]
    .set_index("feature")
    .loc[
        FEATURE_COLS,
        "mean"
    ]
    .to_numpy(float)
)


similarity_rows.append({

    "comparison":
        "field_C_vs_field_D",

    "cosine_weight_similarity":
        cosine(
            wC,
            wD
        ),

    "l1_weight_distance":
        float(
            np.sum(
                np.abs(
                    wC
                    -
                    wD
                )
            )
        ),
})


weight_similarity = pd.DataFrame(
    similarity_rows
)


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

diagnostics.to_csv(
    OUT / "diagnostics.csv",
    index=False
)

stability.to_csv(
    OUT / "seed_stability.csv",
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

pivot_weights.to_csv(
    OUT / "cross_region_weights.csv",
    index=False
)

weight_similarity.to_csv(
    OUT / "weight_similarity.csv",
    index=False
)


# ==========================================================
# Print
# ==========================================================

print()
print("=" * 150)
print(
    "GAIA INDEPENDENT-REGION METRIC SUMMARY"
)
print("=" * 150)

print(
    metric_summary.to_string(
        index=False
    )
)


print()
print("=" * 170)
print(
    "GAIA INDEPENDENT-REGION WEIGHTS"
)
print("=" * 170)

print(
    pivot_weights.to_string(
        index=False
    )
)


print()
print("=" * 120)
print(
    "GAIA INDEPENDENT-REGION WEIGHT SIMILARITY"
)
print("=" * 120)

print(
    weight_similarity.to_string(
        index=False
    )
)


print()
print("=" * 150)
print(
    "GAIA INDEPENDENT-REGION SEED STABILITY"
)
print("=" * 150)

print(
    stability_summary.to_string(
        index=False
    )
)


print()
print("=" * 150)
print(
    "GAIA INDEPENDENT-REGION DIAGNOSTICS"
)
print("=" * 150)

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
