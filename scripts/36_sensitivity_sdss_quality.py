from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution
from scipy.stats import spearmanr


ROOT = Path.home() / "universe_revision" / "revision_v2"

INPUT = (
    ROOT
    / "data/processed/sdss_features_reference.csv"
)

OUT = (
    ROOT
    / "outputs/sensitivity/sdss_quality"
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

TOP_K = 1000
MIN_EFF = 3.0

SEEDS = [
    0,
    1,
    2,
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


def rebuild_features(df):

    out = df.copy()


    out[
        "f_photometric_quality"
    ] = inverse_01(
        out[
            "mean_phot_error"
        ]
    )


    out[
        "f_color_rarity"
    ] = robust_01(
        out[
            "color_distance"
        ]
    )


    out[
        "f_redshift_interest"
    ] = robust_01(
        out[
            "log_redshift"
        ]
    )


    out[
        "f_redshift_precision"
    ] = inverse_01(
        out[
            "redshift_error"
        ]
    )


    out[
        "f_local_density"
    ] = (
        0.50
        *
        robust_01(
            out[
                "local_density_knn"
            ]
        )
        +
        0.25
        *
        robust_01(
            out[
                "neighbor_count_radius"
            ]
        )
        +
        0.25
        *
        robust_01(
            out[
                "redshift_neighbor_count"
            ]
        )
    )


    out[
        "f_isolation"
    ] = robust_01(
        out[
            "nn_sep_arcsec"
        ]
    )


    out = out.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )


    out = (
        out
        .dropna(
            subset=FEATURE_COLS
        )
        .reset_index(drop=True)
    )


    return out


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

        top_scores = score[idx]


        mean_score = float(
            np.mean(
                top_scores
            )
        )


        diversity = float(
            np.mean(
                np.std(
                    top_X,
                    axis=0
                )
            )
        )


        entropy = float(
            entropy_of_weights(
                w
            )
        )


        neff = float(
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
# Load definitive SDSS feature table
# ==========================================================

raw = pd.read_csv(
    INPUT,
    dtype={
        "objid": str,
        "specobjid": str,
        "class": str,
    }
)


print("=" * 120)
print("SDSS QUALITY-CUT SENSITIVITY")
print("=" * 120)

print(
    "Rows:",
    f"{len(raw):,}"
)

print()
print("Class composition:")
print(
    raw[
        "class"
    ].value_counts()
)


# ==========================================================
# Rebuild baseline and verify feature implementation
# ==========================================================

baseline_rebuilt = (
    rebuild_features(
        raw
    )
)


print()
print("=" * 120)
print("BASELINE FEATURE RECONSTRUCTION AUDIT")
print("=" * 120)


audit_rows = []


for f in FEATURE_COLS:

    original = (
        pd.to_numeric(
            raw[f],
            errors="coerce"
        )
        .to_numpy(float)
    )

    rebuilt = (
        baseline_rebuilt[f]
        .to_numpy(float)
    )


    diff = np.abs(
        original
        -
        rebuilt
    )


    row = {
        "feature": f,
        "mean_abs_diff":
            float(
                np.nanmean(
                    diff
                )
            ),
        "max_abs_diff":
            float(
                np.nanmax(
                    diff
                )
            ),
    }

    audit_rows.append(
        row
    )


    print(
        f"{f:30s}",
        "mean_abs_diff=",
        f"{row['mean_abs_diff']:.12g}",
        "max_abs_diff=",
        f"{row['max_abs_diff']:.12g}",
    )


audit = pd.DataFrame(
    audit_rows
)

audit.to_csv(
    OUT
    / "baseline_feature_reconstruction_audit.csv",
    index=False
)


# Require essentially exact reproduction.
if (
    audit[
        "max_abs_diff"
    ].max()
    >
    1e-10
):

    raise RuntimeError(
        "Baseline feature reconstruction "
        "does not reproduce definitive features."
    )


# ==========================================================
# Freeze quality thresholds BEFORE ranking evaluation
# ==========================================================

threshold_rows = []


for cls in [
    "GALAXY",
    "QSO",
]:

    d = raw[
        raw["class"] == cls
    ]


    for variable in [
        "mean_phot_error",
        "redshift_error",
    ]:

        for q in [
            0.90,
            0.95,
        ]:

            threshold_rows.append({
                "class": cls,
                "variable": variable,
                "quantile": q,
                "threshold": float(
                    d[
                        variable
                    ].quantile(
                        q
                    )
                ),
            })


thresholds = pd.DataFrame(
    threshold_rows
)


thresholds.to_csv(
    OUT
    / "quality_thresholds_frozen.csv",
    index=False
)


print()
print("=" * 120)
print("FROZEN CLASS-CONDITIONAL QUALITY THRESHOLDS")
print("=" * 120)

print(
    thresholds.to_string(
        index=False
    )
)


def threshold_for(
    cls,
    variable,
    quantile,
):

    row = thresholds[
        (thresholds["class"] == cls)
        &
        (thresholds["variable"] == variable)
        &
        (
            np.isclose(
                thresholds["quantile"],
                quantile
            )
        )
    ]

    if len(row) != 1:

        raise RuntimeError(
            "Threshold lookup failed."
        )

    return float(
        row.iloc[0][
            "threshold"
        ]
    )


# ==========================================================
# Define fixed configurations
# ==========================================================

CONFIGS = [
    {
        "label": "baseline",
        "variable": None,
        "quantile": None,
    },
    {
        "label": "phot_q95",
        "variable":
            "mean_phot_error",
        "quantile": 0.95,
    },
    {
        "label": "phot_q90",
        "variable":
            "mean_phot_error",
        "quantile": 0.90,
    },
    {
        "label": "zerr_q95",
        "variable":
            "redshift_error",
        "quantile": 0.95,
    },
    {
        "label": "zerr_q90",
        "variable":
            "redshift_error",
        "quantile": 0.90,
    },
]


datasets = {}

diagnostic_rows = []


for cfg in CONFIGS:

    label = cfg[
        "label"
    ]


    if cfg[
        "variable"
    ] is None:

        d = raw.copy()

    else:

        keep = np.zeros(
            len(raw),
            dtype=bool
        )


        for cls in [
            "GALAXY",
            "QSO",
        ]:

            t = threshold_for(
                cls,
                cfg[
                    "variable"
                ],
                cfg[
                    "quantile"
                ],
            )


            cls_mask = (
                raw[
                    "class"
                ]
                ==
                cls
            )


            quality_mask = (
                pd.to_numeric(
                    raw[
                        cfg[
                            "variable"
                        ]
                    ],
                    errors="coerce"
                )
                <=
                t
            )


            keep |= (
                cls_mask
                &
                quality_mask
            )


        d = (
            raw[
                keep
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


    d = rebuild_features(
        d
    )


    datasets[
        label
    ] = d


    counts = (
        d[
            "class"
        ]
        .value_counts()
    )


    diagnostic_rows.append({

        "configuration":
            label,

        "quality_variable":
            cfg[
                "variable"
            ],

        "quantile":
            cfg[
                "quantile"
            ],

        "n_total":
            len(d),

        "n_galaxy":
            int(
                counts.get(
                    "GALAXY",
                    0
                )
            ),

        "n_qso":
            int(
                counts.get(
                    "QSO",
                    0
                )
            ),

        "qso_fraction":
            float(
                np.mean(
                    d[
                        "class"
                    ]
                    ==
                    "QSO"
                )
            ),

        "median_phot_error":
            float(
                d[
                    "mean_phot_error"
                ].median()
            ),

        "median_redshift_error":
            float(
                d[
                    "redshift_error"
                ].median()
            ),
    })


diagnostics = pd.DataFrame(
    diagnostic_rows
)


print()
print("=" * 130)
print("QUALITY-CUT SAMPLE DIAGNOSTICS")
print("=" * 130)

print(
    diagnostics.to_string(
        index=False
    )
)


# ==========================================================
# Optimization
# ==========================================================

metric_rows = []
weight_rows = []

rankings = {}


for cfg in CONFIGS:

    label = cfg[
        "label"
    ]

    d = datasets[
        label
    ]


    print()
    print("=" * 120)
    print(
        "OPTIMIZATION:",
        label,
        "| N =",
        f"{len(d):,}"
    )
    print("=" * 120)


    for seed in SEEDS:

        print(
            "seed=",
            seed
        )


        res = optimize(
            d,
            seed
        )


        ranked = (
            d[
                [
                    "objid",
                    "specobjid",
                    "class",
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
                    "objid",
                ],
                ascending=[
                    False,
                    True,
                ],
                kind="mergesort"
            )
            .reset_index(
                drop=True
            )
        )


        rankings[
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
                len(d),

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

                "configuration":
                    label,

                "seed":
                    seed,

                "feature":
                    feature,

                "weight":
                    float(
                        weight
                    ),
            })


metrics = pd.DataFrame(
    metric_rows
)

weights = pd.DataFrame(
    weight_rows
)


# ==========================================================
# Compare on common object universe
# ==========================================================

comparison_rows = []


for cfg in CONFIGS:

    label = cfg[
        "label"
    ]


    for seed in SEEDS:

        base_rank = rankings[
            (
                "baseline",
                seed
            )
        ]

        alt_rank = rankings[
            (
                label,
                seed
            )
        ]


        base_score = (
            base_rank
            .set_index(
                "objid"
            )[
                "score"
            ]
        )

        alt_score = (
            alt_rank
            .set_index(
                "objid"
            )[
                "score"
            ]
        )


        common = (
            base_score.index
            .intersection(
                alt_score.index
            )
        )


        b = (
            base_score.loc[
                common
            ]
        )

        a = (
            alt_score.loc[
                common
            ]
        )


        rho = spearmanr(
            b,
            a
        ).statistic


        K = min(
            TOP_K,
            len(common)
        )


        b_top = set(
            b
            .sort_values(
                ascending=False
            )
            .head(K)
            .index
        )


        a_top = set(
            a
            .sort_values(
                ascending=False
            )
            .head(K)
            .index
        )


        overlap = len(
            b_top
            &
            a_top
        )


        union = len(
            b_top
            |
            a_top
        )


        comparison_rows.append({

            "configuration":
                label,

            "seed":
                seed,

            "n_common":
                len(common),

            "fraction_baseline_retained":
                len(common)
                /
                len(base_rank),

            "spearman_common":
                float(
                    rho
                ),

            "top1000_overlap_common":
                overlap,

            "top1000_jaccard_common":
                float(
                    overlap
                    /
                    union
                ),
        })


comparison = pd.DataFrame(
    comparison_rows
)


# ==========================================================
# Summaries
# ==========================================================

metric_summary = (
    metrics
    .groupby(
        "configuration"
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
        fraction_baseline_retained=(
            "fraction_baseline_retained",
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
        top1000_overlap_common_mean=(
            "top1000_overlap_common",
            "mean"
        ),
        top1000_overlap_common_min=(
            "top1000_overlap_common",
            "min"
        ),
        top1000_jaccard_common_mean=(
            "top1000_jaccard_common",
            "mean"
        ),
        top1000_jaccard_common_min=(
            "top1000_jaccard_common",
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
        diagnostics,
        on="configuration",
        how="left"
    )
)


weight_summary = (
    weights
    .groupby(
        [
            "configuration",
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


# ==========================================================
# Save
# ==========================================================

diagnostics.to_csv(
    OUT
    / "sample_diagnostics.csv",
    index=False
)

metrics.to_csv(
    OUT
    / "metrics.csv",
    index=False
)

weights.to_csv(
    OUT
    / "weights.csv",
    index=False
)

comparison.to_csv(
    OUT
    / "comparison_common_universe.csv",
    index=False
)

summary.to_csv(
    OUT
    / "summary.csv",
    index=False
)

weight_summary.to_csv(
    OUT
    / "weight_summary.csv",
    index=False
)


# ==========================================================
# Print
# ==========================================================

print()
print("=" * 180)
print("SDSS QUALITY-CUT SENSITIVITY SUMMARY")
print("=" * 180)

cols = [
    "configuration",
    "n",
    "n_galaxy",
    "n_qso",
    "qso_fraction",
    "objective_mean",
    "neff_mean",
    "n_common",
    "fraction_baseline_retained",
    "spearman_common_mean",
    "spearman_common_min",
    "top1000_overlap_common_mean",
    "top1000_overlap_common_min",
    "top1000_jaccard_common_mean",
]

print(
    summary[
        cols
    ].to_string(
        index=False
    )
)


print()
print("=" * 165)
print("SDSS QUALITY-CUT WEIGHT SUMMARY")
print("=" * 165)

pivot = (
    weight_summary
    .pivot(
        index="configuration",
        columns="feature",
        values="mean"
    )
    .reset_index()
)

print(
    pivot.to_string(
        index=False
    )
)


print()
print("=" * 130)
print("SDSS QUALITY-CUT SAMPLE DIAGNOSTICS")
print("=" * 130)

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
