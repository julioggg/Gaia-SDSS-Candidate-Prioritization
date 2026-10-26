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
    / "data"
    / "external_validation"
    / "gaia"
    / "gaia_external_labels_exact.csv"
)

OUT = (
    ROOT
    / "outputs"
    / "external_validation"
    / "gaia"
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


LABELS = {
    "VARIABLE": "label_variable",
    "VARIABLE_OR_NSS":
        "label_variable_or_nss",
}


K_VALUES = [
    50,
    100,
    250,
    500,
    1000,
]


# ==========================================================
# Labels
# ==========================================================

labels = pd.read_csv(
    LABEL_FILE,
    dtype={"source_id": str}
)

labels = (
    labels
    .drop_duplicates(
        subset=["source_id"]
    )
    .copy()
)

N = len(labels)

print("=" * 100)
print("GAIA INDEPENDENT EXTERNAL VALIDATION")
print("=" * 100)

print("N:", N)

for label_name, label_col in LABELS.items():

    positives = int(
        labels[label_col].sum()
    )

    prevalence = (
        positives / N
    )

    print(
        f"{label_name}: "
        f"{positives}/{N} "
        f"({100*prevalence:.3f}%)"
    )


# ==========================================================
# Helpers
# ==========================================================

def odds_ratio_ci(a, b, c, d):
    """
    Approximate Wald CI on log(OR).
    Haldane-Anscombe correction is applied if any
    contingency-table cell is zero.
    """

    cells = np.array(
        [a, b, c, d],
        dtype=float
    )

    if np.any(cells == 0):
        cells += 0.5

    aa, bb, cc, dd = cells

    OR = (
        aa * dd
        /
        (bb * cc)
    )

    se = np.sqrt(
        1/aa
        + 1/bb
        + 1/cc
        + 1/dd
    )

    log_or = np.log(OR)

    lower = np.exp(
        log_or - 1.96*se
    )

    upper = np.exp(
        log_or + 1.96*se
    )

    return OR, lower, upper


all_rows = []
global_rows = []


# ==========================================================
# Evaluate all methods
# ==========================================================

for method, path in METHODS.items():

    print()
    print("=" * 100)
    print("METHOD:", method)
    print("=" * 100)

    ranked = pd.read_csv(
        path,
        dtype={"source_id": str}
    )

    if "source_id" not in ranked.columns:
        raise RuntimeError(
            f"{method}: source_id missing"
        )

    ranked = (
        ranked
        .drop_duplicates(
            subset=["source_id"]
        )
        .copy()
    )

    # ------------------------------------------------------
    # Determine score column
    # ------------------------------------------------------

    if method == "Proposed":

        score_col = "score"

    else:

        score_col = "anomaly_score"

    if score_col not in ranked.columns:
        raise RuntimeError(
            f"{method}: {score_col} missing"
        )

    # Explicitly enforce descending ranking.
    ranked = (
        ranked
        .sort_values(
            score_col,
            ascending=False
        )
        .reset_index(drop=True)
    )

    ranked["rank_eval"] = (
        np.arange(
            1,
            len(ranked)+1
        )
    )

    merged = ranked[
        [
            "source_id",
            score_col,
            "rank_eval",
        ]
    ].merge(
        labels[
            [
                "source_id",
                "label_variable",
                "label_nss",
                "label_variable_or_nss",
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

    # ------------------------------------------------------
    # Global ranking metrics
    # ------------------------------------------------------

    for label_name, label_col in LABELS.items():

        y = merged[
            label_col
        ].to_numpy(int)

        scores = merged[
            score_col
        ].to_numpy(float)

        auc = roc_auc_score(
            y,
            scores
        )

        ap = average_precision_score(
            y,
            scores
        )

        prevalence = (
            y.mean()
        )

        ap_enrichment = (
            ap / prevalence
            if prevalence > 0
            else np.nan
        )

        global_rows.append({
            "method": method,
            "label": label_name,
            "n": len(y),
            "positives": int(
                y.sum()
            ),
            "prevalence":
                prevalence,
            "roc_auc":
                float(auc),
            "average_precision":
                float(ap),
            "ap_over_prevalence":
                float(ap_enrichment),
        })

        # --------------------------------------------------
        # Top-K metrics
        # --------------------------------------------------

        total_positive = int(
            y.sum()
        )

        total_negative = (
            N - total_positive
        )

        for K in K_VALUES:

            top = (
                merged
                .head(K)
            )

            observed = int(
                top[label_col].sum()
            )

            expected = (
                K
                * total_positive
                / N
            )

            precision = (
                observed / K
            )

            recall = (
                observed / total_positive
                if total_positive > 0
                else np.nan
            )

            enrichment = (
                precision
                /
                (
                    total_positive / N
                )
                if total_positive > 0
                else np.nan
            )

            # Contingency table:
            #
            #             positive negative
            # top K          a        b
            # outside        c        d

            a = observed
            b = K - observed
            c = (
                total_positive
                - observed
            )
            d = (
                (N - K)
                - c
            )

            fisher_or, pvalue = (
                fisher_exact(
                    [
                        [a, b],
                        [c, d],
                    ],
                    alternative="greater"
                )
            )

            orci, ci_low, ci_high = (
                odds_ratio_ci(
                    a, b, c, d
                )
            )

            all_rows.append({
                "method":
                    method,

                "label":
                    label_name,

                "K":
                    K,

                "observed_positive":
                    observed,

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

                "odds_ratio_ci_estimate":
                    orci,

                "or_ci95_low":
                    ci_low,

                "or_ci95_high":
                    ci_high,

                "fisher_p_one_sided":
                    pvalue,
            })


# ==========================================================
# Save
# ==========================================================

results = pd.DataFrame(
    all_rows
)

global_metrics = pd.DataFrame(
    global_rows
)


results.to_csv(
    OUT
    / "topk_external_validation.csv",
    index=False
)

global_metrics.to_csv(
    OUT
    / "global_external_validation.csv",
    index=False
)


# ==========================================================
# Friendly tables
# ==========================================================

print()
print("=" * 115)
print("GLOBAL RANKING METRICS")
print("=" * 115)

print(
    global_metrics[
        [
            "method",
            "label",
            "positives",
            "prevalence",
            "roc_auc",
            "average_precision",
            "ap_over_prevalence",
        ]
    ]
    .sort_values(
        [
            "label",
            "average_precision"
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


for label_name in LABELS:

    print()
    print("=" * 115)
    print(
        f"TOP-K EXTERNAL VALIDATION: "
        f"{label_name}"
    )
    print("=" * 115)

    tab = (
        results[
            results["label"]
            == label_name
        ]
        [
            [
                "method",
                "K",
                "observed_positive",
                "expected_random",
                "precision_at_k",
                "recall_at_k",
                "enrichment",
                "fisher_odds_ratio",
                "or_ci95_low",
                "or_ci95_high",
                "fisher_p_one_sided",
            ]
        ]
        .sort_values(
            [
                "K",
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


# ==========================================================
# Best method per K
# ==========================================================

print()
print("=" * 115)
print("BEST ENRICHMENT BY LABEL / K")
print("=" * 115)

best = (
    results
    .sort_values(
        "enrichment",
        ascending=False
    )
    .groupby(
        [
            "label",
            "K"
        ],
        as_index=False
    )
    .first()
)

print(
    best[
        [
            "label",
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
            "label",
            "K"
        ]
    )
    .to_string(
        index=False
    )
)


print()
print(
    "Saved to:",
    OUT
)

print("DONE.")
