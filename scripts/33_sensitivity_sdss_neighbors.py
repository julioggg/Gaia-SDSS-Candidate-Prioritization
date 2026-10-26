from pathlib import Path
import json
import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

from reference_neighbors import add_reference_neighbor_features


ROOT = Path.home() / "universe_revision" / "revision_v2"

TARGET_FILE = (
    ROOT / "data/raw/frozen/sdss_raw_frozen.csv"
)

REFERENCE_FILE = (
    ROOT / "data/reference/sdss/sdss_reference_buffered.csv"
)

BASELINE_FEATURE_FILE = (
    ROOT / "data/processed/sdss_features_reference.csv"
)

OUT = (
    ROOT / "outputs/sensitivity/sdss_neighbors"
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


CONFIGS = [

    {
        "name": "baseline_r300_k5_rz600_dz005",
        "radius": 300.0,
        "k": 5,
        "redshift_radius": 600.0,
        "dz": 0.05,
    },

    {
        "name": "r150_k5_rz600_dz005",
        "radius": 150.0,
        "k": 5,
        "redshift_radius": 600.0,
        "dz": 0.05,
    },

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
        "name": "r300_k10_rz600_dz005",
        "radius": 300.0,
        "k": 10,
        "redshift_radius": 600.0,
        "dz": 0.05,
    },

    {
        "name": "r300_k5_rz300_dz005",
        "radius": 300.0,
        "k": 5,
        "redshift_radius": 300.0,
        "dz": 0.05,
    },

    {
        "name": "r300_k5_rz1200_dz005",
        "radius": 300.0,
        "k": 5,
        "redshift_radius": 1200.0,
        "dz": 0.05,
    },

    {
        "name": "r300_k5_rz600_dz0025",
        "radius": 300.0,
        "k": 5,
        "redshift_radius": 600.0,
        "dz": 0.025,
    },

    {
        "name": "r300_k5_rz600_dz010",
        "radius": 300.0,
        "k": 5,
        "redshift_radius": 600.0,
        "dz": 0.10,
    },
]


# ==========================================================
# Utilities
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


def weighted_geometric_score(
    X,
    w
):

    eps = 1e-9

    w = np.asarray(
        w,
        dtype=float
    )

    w = (
        w
        /
        w.sum()
    )

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


def entropy_of_weights(w):

    w = np.asarray(
        w,
        dtype=float
    )

    w = (
        w
        /
        w.sum()
    )

    eps = 1e-12

    H = -np.sum(
        w
        *
        np.log(
            w + eps
        )
    )

    return (
        H
        /
        np.log(
            len(w)
        )
    )


def effective_number(w):

    w = np.asarray(
        w,
        dtype=float
    )

    w = (
        w
        /
        w.sum()
    )

    return (
        1.0
        /
        np.sum(
            w**2
        )
    )


def optimize(
    df,
    seed=42,
    top_k=1000,
    min_eff=3.0,
):

    X = (
        df[
            FEATURE_COLS
        ]
        .to_numpy(float)
    )

    def objective(
        w_raw
    ):

        w_raw = np.asarray(
            w_raw,
            dtype=float
        )

        if (
            not np.isfinite(
                w_raw
            ).all()
            or
            w_raw.sum() <= 0
        ):
            return 1e6

        w = (
            w_raw
            /
            w_raw.sum()
        )

        score = (
            weighted_geometric_score(
                X,
                w
            )
        )

        K = min(
            top_k,
            len(score)
        )

        idx = np.argpartition(
            score,
            -K
        )[-K:]

        mean_top = float(
            np.mean(
                score[idx]
            )
        )

        feature_top_means = (
            X[idx]
            .mean(axis=0)
        )

        diversity_top = float(
            np.mean(
                feature_top_means
            )
        )

        entropy = float(
            entropy_of_weights(
                w
            )
        )

        neff = float(
            effective_number(
                w
            )
        )

        concentration_penalty = max(
            0.0,
            (
                min_eff
                -
                neff
            )
            /
            min_eff
        )

        value = (
            0.55 * mean_top
            +
            0.20 * diversity_top
            +
            0.20 * entropy
            -
            0.25 * concentration_penalty
        )

        return -value


    result = differential_evolution(
        objective,
        bounds=[
            (0.0, 1.0)
            for _ in FEATURE_COLS
        ],
        seed=seed,
        maxiter=300,
        popsize=30,
        tol=1e-6,
        polish=True,
        workers=1,
        updating="immediate",
    )


    w = np.asarray(
        result.x,
        dtype=float
    )

    w = (
        w
        /
        w.sum()
    )

    score = (
        weighted_geometric_score(
            X,
            w
        )
    )

    return {
        "weights": w,
        "score": score,
        "objective": float(
            -result.fun
        ),
        "neff": float(
            effective_number(
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
# Load data
# ==========================================================

targets = pd.read_csv(
    TARGET_FILE,
    dtype={
        "objid": str,
        "specobjid": str,
        "class": str,
    }
)

ref = pd.read_csv(
    REFERENCE_FILE,
    dtype={
        "objid": str,
        "specobjid": str,
        "class": str,
    }
)

baseline_df = pd.read_csv(
    BASELINE_FEATURE_FILE,
    dtype={
        "objid": str,
        "specobjid": str,
        "class": str,
    }
)


# ==========================================================
# Basic photometric / redshift quantities
# ==========================================================

# Resolve column aliases used in frozen source table.

if "redshift" not in targets.columns:
    if "z_spec" in targets.columns:
        targets["redshift"] = targets["z_spec"]
    elif "z" in targets.columns:
        targets["redshift"] = targets["z"]

if "redshift_error" not in targets.columns:
    if "zerr" in targets.columns:
        targets["redshift_error"] = targets["zerr"]
    elif "zErr" in targets.columns:
        targets["redshift_error"] = targets["zErr"]


# SDSS color indices

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


# Photometric error aliases.

err_aliases = {
    "err_u": [
        "err_u",
        "uErr",
    ],
    "err_g": [
        "err_g",
        "gErr",
    ],
    "err_r": [
        "err_r",
        "rErr",
    ],
    "err_i": [
        "err_i",
        "iErr",
    ],
    "err_z": [
        "err_z",
        "zErrPhot",
        "zErr_photo",
    ],
}

for outcol, candidates in err_aliases.items():

    if outcol in targets.columns:
        continue

    found = None

    for c in candidates:

        if c in targets.columns:
            found = c
            break

    if found is None:
        raise RuntimeError(
            f"Cannot find photometric "
            f"error column for {outcol}. "
            f"Columns include: "
            f"{targets.columns.tolist()}"
        )

    targets[outcol] = (
        targets[found]
    )


err_cols = [
    "err_u",
    "err_g",
    "err_r",
    "err_i",
    "err_z",
]

targets[
    "mean_phot_error"
] = (
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
    .mean(
        axis=1
    )
)


# ==========================================================
# Run configurations
# ==========================================================

summary_rows = []
ranking_tables = {}


for cfg in CONFIGS:

    print()
    print("=" * 110)
    print(cfg["name"])
    print("=" * 110)

    df = targets.copy()


    df = add_reference_neighbor_features(

        targets=df,

        reference=ref,

        radius_arcsec=(
            cfg["radius"]
        ),

        kth_neighbor=(
            cfg["k"]
        ),

        redshift_radius_arcsec=(
            cfg[
                "redshift_radius"
            ]
        ),

        dz=(
            cfg["dz"]
        ),

        batch_size=2000,
    )


    # ======================================================
    # Same feature construction as definitive pipeline
    # ======================================================

    color_cols = [
        "u_g",
        "g_r",
        "r_i",
        "i_z",
    ]

    color_distance = (
        np.zeros(
            len(df)
        )
    )

    for c in color_cols:

        color_distance += (
            df[c]
            -
            df[c].median()
        ) ** 2

    df[
        "color_distance"
    ] = np.sqrt(
        color_distance
    )

    df[
        "log_redshift"
    ] = np.log1p(
        df[
            "redshift"
        ]
    )


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


    if len(df) != 50000:

        print(
            "WARNING rows after features:",
            len(df)
        )


    # ======================================================
    # Optimize
    # ======================================================

    res = optimize(
        df,
        seed=42,
        top_k=1000,
        min_eff=3.0,
    )

    df[
        "sensitivity_score"
    ] = res[
        "score"
    ]


    ranked = (
        df[
            [
                "objid",
                "specobjid",
                "class",
                "sensitivity_score",
            ]
        ]
        .sort_values(
            [
                "sensitivity_score",
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

    ranking_tables[
        cfg["name"]
    ] = ranked


    # ======================================================
    # Neighborhood diagnostics
    # ======================================================

    nn = pd.to_numeric(
        df[
            "nn_sep_arcsec"
        ],
        errors="coerce"
    )

    kth = pd.to_numeric(
        df[
            "kth_sep_arcsec"
        ],
        errors="coerce"
    )

    cnt = pd.to_numeric(
        df[
            "neighbor_count_radius"
        ],
        errors="coerce"
    )

    zcnt = pd.to_numeric(
        df[
            "redshift_neighbor_count"
        ],
        errors="coerce"
    )


    row = {

        "config":
            cfg["name"],

        "radius_arcsec":
            cfg["radius"],

        "k":
            cfg["k"],

        "redshift_radius_arcsec":
            cfg[
                "redshift_radius"
            ],

        "dz":
            cfg["dz"],

        "n":
            len(df),

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

        "median_nn_arcsec":
            float(
                np.nanmedian(
                    nn
                )
            ),

        "median_kth_arcsec":
            float(
                np.nanmedian(
                    kth
                )
            ),

        "median_count":
            float(
                np.nanmedian(
                    cnt
                )
            ),

        "median_redshift_count":
            float(
                np.nanmedian(
                    zcnt
                )
            ),
    }


    for feature, w in zip(
        FEATURE_COLS,
        res["weights"]
    ):

        row[
            "weight_"
            + feature
        ] = float(w)


    summary_rows.append(
        row
    )

    print(
        "objective:",
        row["objective"]
    )

    print(
        "Neff:",
        row["neff"]
    )

    print(
        "weights:"
    )

    for feature, w in zip(
        FEATURE_COLS,
        res["weights"]
    ):

        print(
            f"  {feature:30s} "
            f"{w:.6f}"
        )


# ==========================================================
# Compare against sensitivity baseline
# ==========================================================

base_name = (
    "baseline_r300_k5_rz600_dz005"
)

base = (
    ranking_tables[
        base_name
    ]
    .copy()
)

base_score = (
    base
    .set_index(
        "objid"
    )[
        "sensitivity_score"
    ]
)

base_top = set(
    base
    .head(1000)[
        "objid"
    ]
)


comparison_rows = []


for name, rank in (
    ranking_tables.items()
):

    s = (
        rank
        .set_index(
            "objid"
        )[
            "sensitivity_score"
        ]
    )

    common = (
        base_score.index
        .intersection(
            s.index
        )
    )

    rho = spearmanr(
        base_score.loc[
            common
        ],
        s.loc[
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
        base_top
        &
        top
    )

    union = len(
        base_top
        |
        top
    )

    jaccard = (
        overlap
        /
        union
    )


    comparison_rows.append({

        "config":
            name,

        "spearman_vs_baseline":
            float(rho),

        "top1000_overlap":
            overlap,

        "top1000_jaccard":
            jaccard,
    })


summary = pd.DataFrame(
    summary_rows
)

comparison = pd.DataFrame(
    comparison_rows
)

final = (
    summary
    .merge(
        comparison,
        on="config",
        how="left"
    )
)


# ==========================================================
# Save
# ==========================================================

final.to_csv(
    OUT
    / "sdss_neighbor_sensitivity_summary.csv",
    index=False
)


for name, rank in (
    ranking_tables.items()
):

    rank.to_csv(
        OUT
        / f"{name}_ranking.csv",
        index=False
    )


print()
print("=" * 150)
print("SDSS NEIGHBOR SENSITIVITY SUMMARY")
print("=" * 150)

cols = [
    "config",
    "objective",
    "neff",
    "median_nn_arcsec",
    "median_kth_arcsec",
    "median_count",
    "median_redshift_count",
    "spearman_vs_baseline",
    "top1000_overlap",
    "top1000_jaccard",
]

print(
    final[
        cols
    ].to_string(
        index=False
    )
)


print()
print("=" * 150)
print("WEIGHT SENSITIVITY")
print("=" * 150)

weight_cols = [
    c
    for c
    in final.columns
    if c.startswith(
        "weight_"
    )
]

print(
    final[
        [
            "config"
        ]
        +
        weight_cols
    ]
    .to_string(
        index=False
    )
)


print()
print(
    "Saved:",
    OUT
)

print("DONE.")
