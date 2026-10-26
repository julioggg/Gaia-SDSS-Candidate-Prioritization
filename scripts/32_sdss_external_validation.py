from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import fisher_exact
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


ROOT = Path.home() / "universe_revision" / "revision_v2"

LABEL_FILE = (
    ROOT
    / "data/external_validation/sdss/"
      "sdss_external_labels_exact.csv"
)

OUT = (
    ROOT
    / "outputs/external_validation/sdss"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# Ranking files
# ==========================================================

METHODS = {

    "GALAXY": {

        "Proposed":
            ROOT
            / "outputs/class_specific/sdss/GALAXY/"
              "ranked_galaxy.csv",

        "IsolationForest":
            ROOT
            / "outputs/anomaly_baselines/sdss_galaxy/"
              "isolation_forest_ranked.csv",

        "LOF":
            ROOT
            / "outputs/anomaly_baselines/sdss_galaxy/"
              "lof_ranked.csv",

        "PCA_Mahalanobis":
            ROOT
            / "outputs/anomaly_baselines/sdss_galaxy/"
              "pca_mahalanobis_ranked.csv",
    },

    "QSO": {

        "Proposed":
            ROOT
            / "outputs/class_specific/sdss/QSO/"
              "ranked_qso.csv",

        "IsolationForest":
            ROOT
            / "outputs/anomaly_baselines/sdss_qso/"
              "isolation_forest_ranked.csv",

        "LOF":
            ROOT
            / "outputs/anomaly_baselines/sdss_qso/"
              "lof_ranked.csv",

        "PCA_Mahalanobis":
            ROOT
            / "outputs/anomaly_baselines/sdss_qso/"
              "pca_mahalanobis_ranked.csv",
    },
}


FRACTIONS = [
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
]


# ==========================================================
# Helpers
# ==========================================================

def odds_ratio_ci(a, b, c, d):

    cells = np.array(
        [a, b, c, d],
        dtype=float
    )

    if np.any(cells == 0):
        cells += 0.5

    aa, bb, cc, dd = cells

    OR = aa * dd / (bb * cc)

    se = np.sqrt(
        1/aa
        + 1/bb
        + 1/cc
        + 1/dd
    )

    lo = np.exp(
        np.log(OR) - 1.96*se
    )

    hi = np.exp(
        np.log(OR) + 1.96*se
    )

    return OR, lo, hi


# ==========================================================
# Load labels
# ==========================================================

labels = pd.read_csv(
    LABEL_FILE,
    dtype=str
)

# Normalize the definitive target identifier after the
# previous merge used suffixes (_target, _spec).
if "objid_target" in labels.columns:
    labels["objid"] = labels["objid_target"].astype(str)
elif "objid" in labels.columns:
    labels["objid"] = labels["objid"].astype(str)
else:
    raise RuntimeError(
        "No target objid column found in SDSS external-label file."
    )

labels["specobjid"] = labels["specobjid"].astype(str)


for c in [
    "class_target",
    "subClass",
]:

    labels[c] = (
        labels[c]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# ==========================================================
# Define benchmarks BEFORE ranking evaluation
# ==========================================================

# ----------------------------------------------------------
# GALAXY:
# only objects with an available subclass.
# Positive if subclass contains AGN or BROADLINE.
# ----------------------------------------------------------

galaxy = labels[
    labels["class_target"] == "GALAXY"
].copy()

galaxy = galaxy[
    galaxy["subClass"] != ""
].copy()

galaxy["label_positive"] = (
    galaxy["subClass"]
    .str.upper()
    .str.contains(
        "AGN|BROADLINE",
        regex=True
    )
    .astype(int)
)

galaxy["benchmark"] = (
    "GALAXY_AGN_OR_BROADLINE"
)


# ----------------------------------------------------------
# QSO:
# only objects with an available subclass.
# BROADLINE is the canonical dominant subclass.
# Positive = any other non-empty subclass.
# ----------------------------------------------------------

qso = labels[
    labels["class_target"] == "QSO"
].copy()

qso = qso[
    qso["subClass"] != ""
].copy()

qso["label_positive"] = (
    qso["subClass"]
    .str.upper()
    .ne("BROADLINE")
    .astype(int)
)

qso["benchmark"] = (
    "QSO_NONCANONICAL_SUBCLASS"
)


BENCHMARKS = {
    "GALAXY": galaxy,
    "QSO": qso,
}


print("="*110)
print("SDSS EXTERNAL SPECTROSCOPIC VALIDATION")
print("="*110)

for cls, d in BENCHMARKS.items():

    P = int(
        d["label_positive"].sum()
    )

    N = len(d)

    print()
    print(
        cls,
        "evaluable:",
        f"{N:,}"
    )

    print(
        cls,
        "positives:",
        f"{P:,}",
        f"({100*P/N:.3f}%)"
    )

    print(
        "Subclass positives:"
    )

    print(
        d.loc[
            d["label_positive"] == 1,
            "subClass"
        ]
        .value_counts()
        .to_string()
    )


# ==========================================================
# Evaluation
# ==========================================================

global_rows = []
top_rows = []


for cls, label_df in BENCHMARKS.items():

    N = len(label_df)

    P = int(
        label_df[
            "label_positive"
        ].sum()
    )

    prevalence = P / N

    label_map = (
        label_df[
            [
                "objid",
                "label_positive",
            ]
        ]
        .drop_duplicates("objid")
        .set_index("objid")
    )


    for method, path in METHODS[cls].items():

        ranked = pd.read_csv(
            path,
            dtype={"objid": str}
        )

        score_col = (
            "score"
            if method == "Proposed"
            else "anomaly_score"
        )

        if score_col not in ranked.columns:
            raise RuntimeError(
                f"{cls}/{method}: "
                f"{score_col} missing."
            )

        ranked = (
            ranked
            .drop_duplicates("objid")
            .sort_values(
                score_col,
                ascending=False
            )
            .reset_index(drop=True)
        )

        # Join labels while preserving ranking order.
        ranked["label_positive"] = (
            ranked["objid"]
            .map(
                label_map[
                    "label_positive"
                ]
            )
        )

        eval_df = (
            ranked[
                ranked[
                    "label_positive"
                ].notna()
            ]
            .copy()
            .reset_index(drop=True)
        )

        eval_df[
            "label_positive"
        ] = (
            eval_df[
                "label_positive"
            ]
            .astype(int)
        )

        if len(eval_df) != N:

            raise RuntimeError(
                f"{cls}/{method}: "
                f"coverage "
                f"{len(eval_df)}/{N}"
            )


        y = (
            eval_df[
                "label_positive"
            ]
            .to_numpy(int)
        )

        s = (
            eval_df[
                score_col
            ]
            .to_numpy(float)
        )


        # --------------------------------------------------
        # Global metrics
        # --------------------------------------------------

        auc = roc_auc_score(
            y,
            s
        )

        ap = average_precision_score(
            y,
            s
        )

        global_rows.append({

            "class":
                cls,

            "benchmark":
                label_df[
                    "benchmark"
                ].iloc[0],

            "method":
                method,

            "n_evaluable":
                N,

            "positives":
                P,

            "prevalence":
                prevalence,

            "roc_auc":
                auc,

            "average_precision":
                ap,

            "ap_over_prevalence":
                ap / prevalence,
        })


        # --------------------------------------------------
        # Top-fraction metrics
        # --------------------------------------------------

        for fraction in FRACTIONS:

            K = max(
                1,
                int(
                    round(
                        fraction * N
                    )
                )
            )

            top = (
                eval_df
                .head(K)
            )

            obs = int(
                top[
                    "label_positive"
                ].sum()
            )

            expected = (
                K * P / N
            )

            precision = (
                obs / K
            )

            recall = (
                obs / P
            )

            enrichment = (
                precision
                /
                prevalence
            )

            a = obs
            b = K - obs
            c = P - obs
            d = (
                (N-K) - c
            )

            fisher_or, pval = (
                fisher_exact(
                    [
                        [a, b],
                        [c, d],
                    ],
                    alternative="greater"
                )
            )

            _, lo, hi = (
                odds_ratio_ci(
                    a, b, c, d
                )
            )

            top_rows.append({

                "class":
                    cls,

                "benchmark":
                    label_df[
                        "benchmark"
                    ].iloc[0],

                "method":
                    method,

                "fraction":
                    fraction,

                "K":
                    K,

                "observed_positive":
                    obs,

                "expected_random":
                    expected,

                "precision_at_k":
                    precision,

                "recall_at_k":
                    recall,

                "enrichment":
                    enrichment,

                "fisher_odds_ratio":
                    fisher_or,

                "or_ci95_low":
                    lo,

                "or_ci95_high":
                    hi,

                "fisher_p_one_sided":
                    pval,
            })


# ==========================================================
# Save
# ==========================================================

global_df = pd.DataFrame(
    global_rows
)

top_df = pd.DataFrame(
    top_rows
)


global_df.to_csv(
    OUT
    / "global_spectroscopic_validation.csv",
    index=False
)

top_df.to_csv(
    OUT
    / "top_fraction_spectroscopic_validation.csv",
    index=False
)


# ==========================================================
# Print results
# ==========================================================

print()
print("="*120)
print("GLOBAL SPECTROSCOPIC VALIDATION")
print("="*120)

print(
    global_df
    .sort_values(
        [
            "class",
            "average_precision",
        ],
        ascending=[
            True,
            False,
        ]
    )
    .to_string(
        index=False
    )
)


for cls in [
    "GALAXY",
    "QSO",
]:

    print()
    print("="*120)
    print(
        f"{cls} TOP-FRACTION VALIDATION"
    )
    print("="*120)

    tab = (
        top_df[
            top_df[
                "class"
            ] == cls
        ]
        .sort_values(
            [
                "fraction",
                "enrichment",
            ],
            ascending=[
                True,
                False,
            ]
        )
    )

    print(
        tab.to_string(
            index=False
        )
    )


print()
print("="*120)
print("BEST METHOD BY CLASS / FRACTION")
print("="*120)

best = (
    top_df
    .sort_values(
        "enrichment",
        ascending=False
    )
    .groupby(
        [
            "class",
            "fraction",
        ],
        as_index=False
    )
    .first()
)

print(
    best[
        [
            "class",
            "fraction",
            "K",
            "method",
            "observed_positive",
            "expected_random",
            "enrichment",
            "fisher_p_one_sided",
        ]
    ]
    .sort_values(
        [
            "class",
            "fraction",
        ]
    )
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
