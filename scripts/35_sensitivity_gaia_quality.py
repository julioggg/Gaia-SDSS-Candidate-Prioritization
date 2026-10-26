from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from sklearn.neighbors import BallTree
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr


ROOT = Path.home() / "universe_revision" / "revision_v2"
CONFIG = ROOT / "configs/gaia_config_reference.yaml"

with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


TARGET_FILE = Path(cfg["outputs"]["raw_csv"])
REFERENCE_FILE = Path(cfg["outputs"]["reference_csv"])

with open(
    ROOT / "data/processed/gaia_feature_columns_reference.json",
    "r",
    encoding="utf-8",
) as f:
    FEATURE_COLS = json.load(f)


OUT = ROOT / "outputs/sensitivity_reference/gaia/quality"
OUT.mkdir(parents=True, exist_ok=True)


ARCSEC_PER_RAD = (180.0 / np.pi) * 3600.0

RADIUS_ARCSEC = float(
    cfg["match_internal"]["radius_arcsec"]
)

KTH_NEIGHBOR = int(
    cfg["match_internal"]["kth_neighbor"]
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


# ==========================================================
# Quality-cut configurations
# ==========================================================

CONFIGS = [
    {
        "label": "baseline_snr5_ruwe14",
        "min_snr": 5.0,
        "max_ruwe": 1.4,
    },
    {
        "label": "snr3_ruwe14",
        "min_snr": 3.0,
        "max_ruwe": 1.4,
    },
    {
        "label": "snr10_ruwe14",
        "min_snr": 10.0,
        "max_ruwe": 1.4,
    },
    {
        "label": "snr5_ruwe12",
        "min_snr": 5.0,
        "max_ruwe": 1.2,
    },
]


# ==========================================================
# Helpers
# ==========================================================

def robust_01(s):

    x = pd.to_numeric(
        s,
        errors="coerce"
    )

    x = x.fillna(
        x.median()
    )

    lo = x.quantile(0.01)
    hi = x.quantile(0.99)

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

    w = normalize_weights(w)

    return np.exp(
        np.sum(
            w
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

    w = normalize_weights(w)

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

    w = normalize_weights(w)

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


def prepare_base(df):

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
    ]

    for c in numeric:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )


    df = df.dropna(
        subset=numeric
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


    return (
        df
        .drop_duplicates(
            subset=["source_id"]
        )
        .reset_index(drop=True)
    )


def apply_quality(
    df,
    min_snr,
    max_ruwe,
):

    return (
        df[
            (df["parallax_snr"] >= min_snr)
            &
            (df["ruwe"] < max_ruwe)
            &
            (df["vtan_kms"] >= 0)
            &
            (df["vtan_kms"] <= MAX_VTAN)
        ]
        .copy()
        .reset_index(drop=True)
    )


def build_neighbor_features(
    targets,
    reference,
):

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
                f"for target row {i}"
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


def build_features(
    targets
):

    targets = targets.copy()


    targets[
        "f_astrometric_quality"
    ] = (
        0.50
        *
        inverse_01(
            targets["ruwe"]
        )
        +
        0.50
        *
        robust_01(
            targets[
                "parallax_snr"
            ]
        )
    )


    color_median = (
        targets[
            "bp_rp"
        ]
        .median()
    )

    targets[
        "f_photometric_color_rarity"
    ] = robust_01(
        np.abs(
            targets["bp_rp"]
            -
            color_median
        )
    )


    abs_mag_median = (
        targets[
            "abs_g_mag"
        ]
        .median()
    )

    targets[
        "f_absolute_magnitude_rarity"
    ] = robust_01(
        np.abs(
            targets[
                "abs_g_mag"
            ]
            -
            abs_mag_median
        )
    )


    targets[
        "f_kinematic_interest"
    ] = robust_01(
        targets[
            "log_vtan"
        ]
    )


    targets[
        "f_local_density"
    ] = (
        0.70
        *
        robust_01(
            targets[
                "local_density_knn"
            ]
        )
        +
        0.30
        *
        robust_01(
            targets[
                "neighbor_count_radius"
            ]
        )
    )


    targets[
        "f_isolation"
    ] = robust_01(
        targets[
            "nn_sep_arcsec"
        ]
    )


    targets = targets.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )


    targets = (
        targets
        .dropna(
            subset=FEATURE_COLS
        )
        .reset_index(drop=True)
    )


    return targets


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

        score = (
            weighted_geometric_score(
                X,
                w
            )
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


        entropy = (
            entropy_of_weights(
                w
            )
        )


        neff = (
            effective_number_of_features(
                w
            )
        )


        penalty = max(
            0.0,
            MIN_EFF - neff
        ) / MIN_EFF


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
        maxiter=180,
        popsize=20,
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
        "weights": w,
        "score": score,
        "objective": float(
            -result.fun
        ),
        "neff": float(
            effective_number_of_features(
                w
            )
        ),
        "entropy": float(
            entropy_of_weights(
                w
            )
        ),
    }


# ==========================================================
# Load RAW frozen target + buffered reference
# ==========================================================

targets_raw = pd.read_csv(
    TARGET_FILE,
    dtype={
        "source_id": str
    }
)

reference_raw = pd.read_csv(
    REFERENCE_FILE,
    dtype={
        "source_id": str
    }
)


targets_raw = prepare_base(
    targets_raw
)

reference_raw = prepare_base(
    reference_raw
)


print("=" * 110)
print("GAIA QUALITY-CUT SENSITIVITY")
print("=" * 110)

print(
    "Raw frozen targets:",
    f"{len(targets_raw):,}"
)

print(
    "Available buffered reference:",
    f"{len(reference_raw):,}"
)


# ==========================================================
# Build every quality configuration
# ==========================================================

datasets = {}

diagnostic_rows = []


for conf in CONFIGS:

    label = conf["label"]

    print()
    print("=" * 110)
    print(label)
    print("=" * 110)

    targets = apply_quality(
        targets_raw,
        conf["min_snr"],
        conf["max_ruwe"],
    )

    reference = apply_quality(
        reference_raw,
        conf["min_snr"],
        conf["max_ruwe"],
    )


    target_ids = set(
        targets[
            "source_id"
        ]
    )

    ref_ids = set(
        reference[
            "source_id"
        ]
    )


    missing = (
        target_ids
        -
        ref_ids
    )


    if missing:

        raise RuntimeError(
            f"{label}: "
            f"{len(missing)} targets "
            f"missing from reference."
        )


    print(
        "Targets retained:",
        f"{len(targets):,}"
    )

    print(
        "Reference retained:",
        f"{len(reference):,}"
    )


    targets = build_neighbor_features(
        targets,
        reference,
    )


    targets = build_features(
        targets
    )


    print(
        "Final feature rows:",
        f"{len(targets):,}"
    )


    diagnostic_rows.append({

        "configuration":
            label,

        "min_parallax_snr":
            conf["min_snr"],

        "max_ruwe":
            conf["max_ruwe"],

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
    })


    datasets[label] = targets


# ==========================================================
# Optimize all configurations
# ==========================================================

metric_rows = []
weight_rows = []
ranking_by_config_seed = {}


for conf in CONFIGS:

    label = conf["label"]

    df = datasets[
        label
    ]

    print()
    print("=" * 110)
    print("OPTIMIZATION:", label)
    print("=" * 110)


    for seed in SEEDS:

        print(
            f"seed={seed}"
        )

        res = optimize(
            df,
            seed
        )


        ranked = (
            df[
                [
                    "source_id"
                ]
            ]
            .copy()
        )

        ranked[
            "score"
        ] = res[
            "score"
        ]


        ranked = (
            ranked
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


        ranking_by_config_seed[
            (
                label,
                seed
            )
        ] = ranked


        metric_rows.append({

            "configuration":
                label,

            "seed":
                seed,

            "n":
                len(df),

            "objective":
                res["objective"],

            "neff":
                res["neff"],

            "entropy":
                res["entropy"],
        })


        for feature, w in zip(
            FEATURE_COLS,
            res["weights"]
        ):

            weight_rows.append({

                "configuration":
                    label,

                "seed":
                    seed,

                "feature":
                    feature,

                "weight":
                    float(w),
            })


# ==========================================================
# Comparison versus baseline on COMMON OBJECT UNIVERSE
# ==========================================================

BASE_LABEL = (
    "baseline_snr5_ruwe14"
)

comparison_rows = []


for conf in CONFIGS:

    label = conf["label"]


    for seed in SEEDS:

        base_rank = (
            ranking_by_config_seed[
                (
                    BASE_LABEL,
                    seed
                )
            ]
        )

        alt_rank = (
            ranking_by_config_seed[
                (
                    label,
                    seed
                )
            ]
        )


        base_score = (
            base_rank
            .set_index(
                "source_id"
            )["score"]
        )

        alt_score = (
            alt_rank
            .set_index(
                "source_id"
            )["score"]
        )


        common = (
            base_score.index
            .intersection(
                alt_score.index
            )
        )


        base_common = (
            base_score.loc[
                common
            ]
        )

        alt_common = (
            alt_score.loc[
                common
            ]
        )


        rho = spearmanr(
            base_common,
            alt_common
        ).statistic


        # Rank both methods only within the common universe.
        base_common_ranked = (
            base_common
            .sort_values(
                ascending=False
            )
        )

        alt_common_ranked = (
            alt_common
            .sort_values(
                ascending=False
            )
        )


        K_common = min(
            TOP_K,
            len(common)
        )


        base_top_common = set(
            base_common_ranked
            .head(K_common)
            .index
        )

        alt_top_common = set(
            alt_common_ranked
            .head(K_common)
            .index
        )


        overlap = len(
            base_top_common
            &
            alt_top_common
        )


        union = len(
            base_top_common
            |
            alt_top_common
        )


        comparison_rows.append({

            "configuration":
                label,

            "seed":
                seed,

            "n_common":
                len(common),

            "baseline_fraction_retained":
                len(common)
                /
                len(base_rank),

            "alt_fraction_in_common":
                len(common)
                /
                len(alt_rank),

            "spearman_common":
                float(rho),

            "topK_common":
                K_common,

            "topK_overlap_common":
                overlap,

            "topK_jaccard_common":
                float(
                    overlap
                    /
                    union
                ),
        })


# ==========================================================
# Save
# ==========================================================

diagnostics = pd.DataFrame(
    diagnostic_rows
)

metrics = pd.DataFrame(
    metric_rows
)

weights = pd.DataFrame(
    weight_rows
)

comparison = pd.DataFrame(
    comparison_rows
)


diagnostics.to_csv(
    OUT / "quality_diagnostics.csv",
    index=False
)

metrics.to_csv(
    OUT / "metrics.csv",
    index=False
)

weights.to_csv(
    OUT / "weights.csv",
    index=False
)

comparison.to_csv(
    OUT / "comparison_common_universe.csv",
    index=False
)


# ==========================================================
# Summary
# ==========================================================

metric_summary = (
    metrics
    .groupby(
        "configuration"
    )
    .agg(
        n=("n", "first"),
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


comparison_summary = (
    comparison
    .groupby(
        "configuration"
    )
    .agg(
        n_common=(
            "n_common",
            "first"
        ),
        baseline_fraction_retained=(
            "baseline_fraction_retained",
            "first"
        ),
        spearman_common_mean=(
            "spearman_common",
            "mean"
        ),
        spearman_common_min=(
            "spearman_common",
            "min"
        ),
        topK_overlap_common_mean=(
            "topK_overlap_common",
            "mean"
        ),
        topK_overlap_common_min=(
            "topK_overlap_common",
            "min"
        ),
        topK_jaccard_common_mean=(
            "topK_jaccard_common",
            "mean"
        ),
        topK_jaccard_common_min=(
            "topK_jaccard_common",
            "min"
        ),
    )
    .reset_index()
)


summary = (
    metric_summary
    .merge(
        comparison_summary,
        on="configuration",
        how="left"
    )
    .merge(
        diagnostics[
            [
                "configuration",
                "min_parallax_snr",
                "max_ruwe",
                "n_reference",
                "median_parallax_snr",
                "median_ruwe",
                "median_nn_arcsec",
                "median_neighbor_count",
            ]
        ],
        on="configuration",
        how="left"
    )
)


summary.to_csv(
    OUT / "summary.csv",
    index=False
)


weight_summary = (
    weights
    .groupby(
        [
            "configuration",
            "feature",
        ]
    )["weight"]
    .agg(
        [
            "mean",
            "std",
        ]
    )
    .reset_index()
)


weight_summary.to_csv(
    OUT / "weight_summary.csv",
    index=False
)


print()
print("=" * 170)
print("GAIA QUALITY-CUT SENSITIVITY SUMMARY")
print("=" * 170)

show_cols = [
    "configuration",
    "n",
    "n_reference",
    "objective_mean",
    "neff_mean",
    "n_common",
    "baseline_fraction_retained",
    "spearman_common_mean",
    "spearman_common_min",
    "topK_overlap_common_mean",
    "topK_overlap_common_min",
    "topK_jaccard_common_mean",
]

print(
    summary[
        show_cols
    ].to_string(
        index=False
    )
)


print()
print("=" * 150)
print("GAIA QUALITY-CUT WEIGHT SUMMARY")
print("=" * 150)

weight_pivot = (
    weight_summary
    .pivot(
        index="configuration",
        columns="feature",
        values="mean"
    )
    .reset_index()
)

print(
    weight_pivot.to_string(
        index=False
    )
)


print()
print("=" * 120)
print("QUALITY DIAGNOSTICS")
print("=" * 120)

print(
    diagnostics.to_string(
        index=False
    )
)


print()
print(
    "Saved:",
    OUT
)

print("DONE.")
