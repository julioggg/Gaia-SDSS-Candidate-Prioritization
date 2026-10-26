from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import fisher_exact, spearmanr
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


ROOT = Path.home() / "universe_revision" / "revision_v2"

LABEL_FILE = (
    ROOT
    / "data"
    / "external_validation"
    / "gaia"
    / "gaia_extended_external_labels_exact.csv"
)

OUT = (
    ROOT
    / "outputs"
    / "external_validation"
    / "gaia"
    / "dsc"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


METHODS = {
    "Proposed": (
        ROOT
        / "outputs"
        / "baseline_reference"
        / "gaia"
        / "metrics"
        / "ranked_gaia.csv"
    ),

    "IsolationForest": (
        ROOT
        / "outputs"
        / "anomaly_baselines"
        / "gaia"
        / "isolation_forest_ranked.csv"
    ),

    "LOF": (
        ROOT
        / "outputs"
        / "anomaly_baselines"
        / "gaia"
        / "lof_ranked.csv"
    ),

    "PCA_Mahalanobis": (
        ROOT
        / "outputs"
        / "anomaly_baselines"
        / "gaia"
        / "pca_mahalanobis_ranked.csv"
    ),
}


K_VALUES = [
    50,
    100,
    250,
    500,
    1000,
]


labels = pd.read_csv(
    LABEL_FILE,
    dtype={"source_id": str}
)

labels = labels.drop_duplicates(
    subset=["source_id"]
)


# ==========================================================
# Keep rows with DSC information
# ==========================================================

labels["classprob_dsc_combmod_star"] = (
    pd.to_numeric(
        labels[
            "classprob_dsc_combmod_star"
        ],
        errors="coerce"
    )
)

valid = labels[
    labels[
        "classprob_dsc_combmod_star"
    ].notna()
].copy()

valid["label_nonstar"] = (
    valid[
        "classprob_dsc_combmod_star"
    ] < 0.5
).astype(int)

valid["dsc_nonstar_probability"] = (
    1.0
    -
    valid[
        "classprob_dsc_combmod_star"
    ]
)

N = len(valid)
P = int(
    valid["label_nonstar"].sum()
)

prevalence = P / N


print("="*110)
print("GAIA DSC SECONDARY VALIDATION")
print("="*110)

print(
    "DSC-valid objects:",
    N
)

print(
    "DSC non-star positives:",
    P,
    f"({100*prevalence:.3f}%)"
)


def odds_ratio_ci(a, b, c, d):

    cells = np.array(
        [a, b, c, d],
        dtype=float
    )

    if np.any(cells == 0):
        cells += 0.5

    aa, bb, cc, dd = cells

    OR = aa*dd/(bb*cc)

    se = np.sqrt(
        1/aa
        + 1/bb
        + 1/cc
        + 1/dd
    )

    lo = np.exp(
        np.log(OR)
        - 1.96*se
    )

    hi = np.exp(
        np.log(OR)
        + 1.96*se
    )

    return OR, lo, hi


global_rows = []
topk_rows = []


for method, path in METHODS.items():

    ranked = pd.read_csv(
        path,
        dtype={"source_id": str}
    )

    score_col = (
        "score"
        if method == "Proposed"
        else "anomaly_score"
    )

    ranked = (
        ranked
        .drop_duplicates(
            subset=["source_id"]
        )
        .sort_values(
            score_col,
            ascending=False
        )
        .reset_index(drop=True)
    )

    merged = ranked[
        [
            "source_id",
            score_col
        ]
    ].merge(
        valid[
            [
                "source_id",
                "label_nonstar",
                "dsc_nonstar_probability",
            ]
        ],
        on="source_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != N:
        raise RuntimeError(
            f"{method}: coverage "
            f"{len(merged)}/{N}"
        )

    y = merged[
        "label_nonstar"
    ].to_numpy(int)

    s = merged[
        score_col
    ].to_numpy(float)

    p_nonstar = merged[
        "dsc_nonstar_probability"
    ].to_numpy(float)

    auc = roc_auc_score(
        y,
        s
    )

    ap = average_precision_score(
        y,
        s
    )

    rho = spearmanr(
        s,
        p_nonstar
    ).statistic

    global_rows.append({
        "method": method,
        "n": N,
        "positives": P,
        "prevalence": prevalence,
        "roc_auc": auc,
        "average_precision": ap,
        "ap_over_prevalence":
            ap/prevalence,
        "spearman_score_vs_dsc_nonstar_probability":
            rho,
    })


    for K in K_VALUES:

        top = merged.head(K)

        obs = int(
            top[
                "label_nonstar"
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

        orci, lo, hi = (
            odds_ratio_ci(
                a, b, c, d
            )
        )

        mean_prob_top = float(
            top[
                "dsc_nonstar_probability"
            ].mean()
        )

        topk_rows.append({
            "method": method,
            "K": K,
            "observed_nonstar":
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
            "mean_dsc_nonstar_probability_topk":
                mean_prob_top,
        })


global_df = pd.DataFrame(
    global_rows
)

topk_df = pd.DataFrame(
    topk_rows
)


global_df.to_csv(
    OUT
    / "global_dsc_validation.csv",
    index=False
)

topk_df.to_csv(
    OUT
    / "topk_dsc_validation.csv",
    index=False
)


print()
print("="*120)
print("GLOBAL DSC VALIDATION")
print("="*120)

print(
    global_df
    .sort_values(
        "average_precision",
        ascending=False
    )
    .to_string(
        index=False
    )
)


print()
print("="*120)
print("TOP-K DSC NON-STAR VALIDATION")
print("="*120)

print(
    topk_df
    .sort_values(
        [
            "K",
            "enrichment"
        ],
        ascending=[
            True,
            False
        ]
    )
    .to_string(
        index=False
    )
)


print()
print("="*120)
print("BEST METHOD BY K")
print("="*120)

best = (
    topk_df
    .sort_values(
        "enrichment",
        ascending=False
    )
    .groupby(
        "K",
        as_index=False
    )
    .first()
)

print(
    best[
        [
            "K",
            "method",
            "observed_nonstar",
            "expected_random",
            "enrichment",
            "fisher_p_one_sided",
        ]
    ]
    .sort_values("K")
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
