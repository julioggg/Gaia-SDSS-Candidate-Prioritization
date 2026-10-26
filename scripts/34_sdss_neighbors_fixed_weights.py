from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from reference_neighbors import add_reference_neighbor_features


ROOT = Path.home() / "universe_revision" / "revision_v2"

TARGET_FILE = (
    ROOT / "data/raw/frozen/sdss_raw_frozen.csv"
)

REFERENCE_FILE = (
    ROOT / "data/reference/sdss/sdss_reference_buffered.csv"
)

BASELINE_FILE = (
    ROOT / "data/processed/sdss_features_reference.csv"
)

OUT = (
    ROOT / "outputs/sensitivity/sdss_neighbors_fixed_weights"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


FEATURE_COLS = [
    "f_photometric_quality",
    "f_color_rarity",
    "f_redshift_interest",
    "f_redshift_precision",
    "f_local_density",
    "f_isolation",
]


# Frozen baseline weights obtained from the
# baseline neighborhood configuration.
BASELINE_WEIGHTS = np.array([
    0.397183,
    0.005514,
    0.331924,
    0.252021,
    0.00001861529,
    0.013340,
], dtype=float)

BASELINE_WEIGHTS /= (
    BASELINE_WEIGHTS.sum()
)


CONFIGS = [

    {
        "name": "r600_k5_rz600_dz005",
        "radius": 600.0,
        "k": 5,
        "redshift_radius": 600.0,
        "dz": 0.05,
    },

    {
        "name": "r300_k3_rz600_dz005",
        "radius": 300.0,
        "k": 3,
        "redshift_radius": 600.0,
        "dz": 0.05,
    },

    {
        "name": "r300_k5_rz600_dz010",
        "radius": 300.0,
        "k": 5,
        "redshift_radius": 600.0,
        "dz": 0.10,
    },
]


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


def weighted_geometric_score(
    X,
    w
):

    eps = 1e-9

    return np.exp(
        np.sum(
            w[None, :]
            *
            np.log(
                np.clip(
                    X,
                    eps,
                    1.0
                )
            ),
            axis=1
        )
    )


# ==========================================================
# Load definitive data
# ==========================================================

targets = pd.read_csv(
    TARGET_FILE,
    dtype={
        "objid": str,
        "specobjid": str,
        "class": str,
    }
)

reference = pd.read_csv(
    REFERENCE_FILE,
    dtype={
        "objid": str,
        "specobjid": str,
        "class": str,
    }
)

baseline = pd.read_csv(
    BASELINE_FILE,
    dtype={
        "objid": str,
        "specobjid": str,
        "class": str,
    }
)


# ==========================================================
# Static features
# ==========================================================

targets["u_g"] = (
    targets["u"]
    -
    targets["g"]
)

targets["g_r"] = (
    targets["g"]
    -
    targets["r"]
)

targets["r_i"] = (
    targets["r"]
    -
    targets["i"]
)

targets["i_z"] = (
    targets["i"]
    -
    targets["z"]
)


err_cols = [
    "err_u",
    "err_g",
    "err_r",
    "err_i",
    "err_z",
]

targets["mean_phot_error"] = (
    targets[
        err_cols
    ]
    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
    .mean(axis=1)
)


color_cols = [
    "u_g",
    "g_r",
    "r_i",
    "i_z",
]

color_distance = np.zeros(
    len(targets)
)

for c in color_cols:

    color_distance += (
        targets[c]
        -
        targets[c].median()
    ) ** 2

targets["color_distance"] = (
    np.sqrt(
        color_distance
    )
)

targets["log_redshift"] = (
    np.log1p(
        targets["redshift"]
    )
)


# ==========================================================
# Baseline ranking using SAME frozen weights
# ==========================================================

baseline_X = (
    baseline[
        FEATURE_COLS
    ]
    .to_numpy(float)
)

baseline[
    "fixed_weight_score"
] = weighted_geometric_score(
    baseline_X,
    BASELINE_WEIGHTS
)

baseline_rank = (
    baseline[
        [
            "objid",
            "fixed_weight_score",
        ]
    ]
    .sort_values(
        [
            "fixed_weight_score",
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

baseline_score_map = (
    baseline_rank
    .set_index("objid")[
        "fixed_weight_score"
    ]
)

baseline_top = set(
    baseline_rank
    .head(1000)[
        "objid"
    ]
)


print("=" * 120)
print("SDSS FIXED-WEIGHT NEIGHBOR SENSITIVITY")
print("=" * 120)

print()
print("Frozen baseline weights:")

for name, w in zip(
    FEATURE_COLS,
    BASELINE_WEIGHTS
):

    print(
        f"{name:30s} {w:.9f}"
    )


rows = []


# ==========================================================
# Recalculate ONLY pathological configurations
# ==========================================================

for cfg in CONFIGS:

    print()
    print("=" * 120)
    print(cfg["name"])
    print("=" * 120)

    df = targets.copy()

    df = add_reference_neighbor_features(
        targets=df,
        reference=reference,
        radius_arcsec=cfg["radius"],
        kth_neighbor=cfg["k"],
        redshift_radius_arcsec=(
            cfg["redshift_radius"]
        ),
        dz=cfg["dz"],
        batch_size=2000,
    )


    # ------------------------------------------------------
    # Exact same feature definitions
    # ------------------------------------------------------

    df[
        "f_photometric_quality"
    ] = inverse_01(
        df["mean_phot_error"]
    )

    df[
        "f_color_rarity"
    ] = robust_01(
        df["color_distance"]
    )

    df[
        "f_redshift_interest"
    ] = robust_01(
        df["log_redshift"]
    )

    df[
        "f_redshift_precision"
    ] = inverse_01(
        df["redshift_error"]
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
        df["nn_sep_arcsec"]
    )


    df = (
        df
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
        .dropna(
            subset=FEATURE_COLS
        )
        .reset_index(drop=True)
    )


    if len(df) != 50000:
        raise RuntimeError(
            f"{cfg['name']}: "
            f"expected 50000 rows, "
            f"got {len(df)}"
        )


    # ------------------------------------------------------
    # FIXED baseline weights
    # ------------------------------------------------------

    X = (
        df[
            FEATURE_COLS
        ]
        .to_numpy(float)
    )

    df[
        "fixed_weight_score"
    ] = weighted_geometric_score(
        X,
        BASELINE_WEIGHTS
    )


    rank = (
        df[
            [
                "objid",
                "specobjid",
                "fixed_weight_score",
            ]
        ]
        .sort_values(
            [
                "fixed_weight_score",
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


    score_map = (
        rank
        .set_index(
            "objid"
        )[
            "fixed_weight_score"
        ]
    )


    common = (
        baseline_score_map.index
        .intersection(
            score_map.index
        )
    )


    rho = spearmanr(
        baseline_score_map.loc[
            common
        ],
        score_map.loc[
            common
        ]
    ).statistic


    top = set(
        rank
        .head(1000)[
            "objid"
        ]
    )

    overlap = len(
        baseline_top
        &
        top
    )

    jaccard = (
        overlap
        /
        len(
            baseline_top
            |
            top
        )
    )


    # ------------------------------------------------------
    # Direct feature differences
    # ------------------------------------------------------

    merged = (
        baseline[
            [
                "objid"
            ]
            +
            FEATURE_COLS
        ]
        .merge(
            df[
                [
                    "objid"
                ]
                +
                FEATURE_COLS
            ],
            on="objid",
            suffixes=(
                "_baseline",
                "_alt"
            ),
            validate="one_to_one"
        )
    )


    feature_diff = {}

    for f in FEATURE_COLS:

        x = merged[
            f"{f}_baseline"
        ].to_numpy(float)

        y = merged[
            f"{f}_alt"
        ].to_numpy(float)

        feature_diff[
            f"mean_abs_diff_{f}"
        ] = float(
            np.mean(
                np.abs(
                    x-y
                )
            )
        )


    row = {

        "config":
            cfg["name"],

        "spearman_fixed_weights":
            float(rho),

        "top1000_overlap_fixed_weights":
            overlap,

        "top1000_jaccard_fixed_weights":
            float(jaccard),

        **feature_diff,
    }

    rows.append(row)


    print(
        "Spearman vs baseline:",
        f"{rho:.9f}"
    )

    print(
        "Top-1000 overlap:",
        overlap
    )

    print(
        "Top-1000 Jaccard:",
        f"{jaccard:.9f}"
    )

    print(
        "Mean absolute feature changes:"
    )

    for f in FEATURE_COLS:

        print(
            f"  {f:30s}",
            f"{feature_diff['mean_abs_diff_'+f]:.9f}"
        )


    rank.to_csv(
        OUT
        / f"{cfg['name']}_fixed_weight_ranking.csv",
        index=False
    )


# ==========================================================
# Summary
# ==========================================================

summary = pd.DataFrame(
    rows
)

summary.to_csv(
    OUT
    / "fixed_weight_summary.csv",
    index=False
)


print()
print("=" * 160)
print("FIXED-WEIGHT SUMMARY")
print("=" * 160)

print(
    summary[
        [
            "config",
            "spearman_fixed_weights",
            "top1000_overlap_fixed_weights",
            "top1000_jaccard_fixed_weights",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 160)
print("FEATURE CHANGE SUMMARY")
print("=" * 160)

print(
    summary.to_string(
        index=False
    )
)


print()
print(
    "Saved:",
    OUT
)

print("DONE.")
