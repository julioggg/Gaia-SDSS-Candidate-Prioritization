from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import yaml

from scipy.optimize import differential_evolution
from scipy.stats import spearmanr


ROOT = Path.home() / "universe_revision" / "revision_v2"

sys.path.insert(
    0,
    str(ROOT / "scripts_revision")
)

sys.path.insert(
    0,
    str(ROOT / "scripts")
)

from reference_neighbors import (
    add_reference_neighbor_features
)

from utils_features import (
    robust_01,
    inverse_01
)


CONFIG = "configs/sdss_config_reference.yaml"

with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


TARGET_FILE = Path(
    cfg["outputs"]["processed_csv"]
)

REFERENCE_FILE = Path(
    cfg["outputs"]["reference_csv"]
)

OUT_ROOT = Path(
    "outputs/class_specific/sdss"
)

DATA_OUT = Path(
    "data/processed/class_specific"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

DATA_OUT.mkdir(
    parents=True,
    exist_ok=True
)


ANGULAR_RADIUS = float(
    cfg["match_internal"][
        "angular_radius_arcsec"
    ]
)

KTH = int(
    cfg["match_internal"][
        "kth_neighbor"
    ]
)

REDSHIFT_RADIUS = float(
    cfg["match_internal"][
        "redshift_radius_arcsec"
    ]
)

DZ = float(
    cfg["match_internal"]["dz"]
)

TOP_K = int(
    cfg["optimization"]["top_k"]
)

MIN_EFF = float(
    cfg["optimization"][
        "min_effective_features"
    ]
)

SEEDS = [0, 1, 2, 42]

CLASSES = [
    "GALAXY",
    "QSO"
]


FEATURE_COLS = [
    "f_photometric_quality",
    "f_color_rarity",
    "f_redshift_interest",
    "f_redshift_precision",
    "f_local_density",
    "f_isolation",
]


# ==========================================================
# Helpers
# ==========================================================

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
            w * np.log(
                w + 1e-12
            )
        )
        /
        np.log(len(w))
    )


def effective_number(w):
    w = normalize_weights(w)

    return (
        1.0
        /
        np.sum(w**2)
    )


def rebuild_class_features(df):
    df = df.copy()

    # -----------------------------------------
    # Colors already available in definitive
    # processed target table.
    # -----------------------------------------

    color_cols = [
        "u_g",
        "g_r",
        "r_i",
        "i_z",
    ]

    med = (
        df[color_cols]
        .median()
    )

    diff = (
        df[color_cols]
        - med
    )

    df["color_distance"] = np.sqrt(
        np.sum(
            diff.to_numpy(float)**2,
            axis=1
        )
    )

    df["log_redshift"] = np.log1p(
        df["redshift"]
    )

    # -----------------------------------------
    # Same original feature definitions,
    # but normalization is class-specific.
    # -----------------------------------------

    df["f_photometric_quality"] = (
        inverse_01(
            df["mean_phot_error"]
        )
    )

    df["f_color_rarity"] = (
        robust_01(
            df["color_distance"]
        )
    )

    df["f_redshift_interest"] = (
        robust_01(
            df["log_redshift"]
        )
    )

    df["f_redshift_precision"] = (
        inverse_01(
            df["redshift_error"]
        )
    )

    df["f_local_density"] = (
        0.50
        * robust_01(
            df["local_density_knn"]
        )
        +
        0.25
        * robust_01(
            df["neighbor_count_radius"]
        )
        +
        0.25
        * robust_01(
            df["redshift_neighbor_count"]
        )
    )

    df["f_isolation"] = (
        robust_01(
            df["nn_sep_arcsec"]
        )
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=FEATURE_COLS
    )

    return df


# ==========================================================
# Load
# ==========================================================

targets_all = pd.read_csv(
    TARGET_FILE,
    dtype={
        "objid": str,
        "specobjid": str,
        "class": str,
    }
)

reference_all = pd.read_csv(
    REFERENCE_FILE,
    dtype={
        "objid": str,
        "specobjid": str,
        "class": str,
    }
)


for col in [
    "ra",
    "dec",
    "redshift",
]:
    reference_all[col] = pd.to_numeric(
        reference_all[col],
        errors="coerce"
    )


reference_all = (
    reference_all
    .dropna(
        subset=[
            "ra",
            "dec",
            "redshift",
            "class",
        ]
    )
    .drop_duplicates(
        subset=["objid"],
        keep="first"
    )
)


print("="*88)
print("SDSS CLASS-SPECIFIC ANALYSIS")
print("="*88)

print(
    "Targets total:",
    f"{len(targets_all):,}"
)

print(
    "Reference total:",
    f"{len(reference_all):,}"
)


summary_rows = []
all_weight_rows = []
class_results = {}


for cls in CLASSES:

    print()
    print("="*88)
    print(f"CLASS: {cls}")
    print("="*88)

    targets = (
        targets_all[
            targets_all["class"] == cls
        ]
        .copy()
        .reset_index(drop=True)
    )

    reference = (
        reference_all[
            reference_all["class"] == cls
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        "Targets:",
        f"{len(targets):,}"
    )

    print(
        "Reference:",
        f"{len(reference):,}"
    )

    # -----------------------------------------
    # Recompute neighbors using only same class.
    # -----------------------------------------

    targets = add_reference_neighbor_features(
        targets,
        reference,
        radius_arcsec=ANGULAR_RADIUS,
        kth_neighbor=KTH,
        redshift_radius_arcsec=REDSHIFT_RADIUS,
        dz=DZ,
    )

    targets = rebuild_class_features(
        targets
    )

    feature_file = (
        DATA_OUT
        / f"sdss_features_{cls.lower()}_reference.csv"
    )

    targets.to_csv(
        feature_file,
        index=False
    )

    print()
    print(
        "% angular neighbor <=300\":",
        f"{100*(targets['neighbor_count_radius'] > 0).mean():.2f}%"
    )

    print(
        "% similar-z neighbor <=600\":",
        f"{100*(targets['redshift_neighbor_count'] > 0).mean():.2f}%"
    )

    print(
        "Median NN:",
        f"{targets['nn_sep_arcsec'].median():.3f}\""
    )

    print(
        "Median kth:",
        f"{targets['kth_sep_arcsec'].median():.3f}\""
    )

    print(
        "Median N<=300\":",
        f"{targets['neighbor_count_radius'].median():.1f}"
    )

    print(
        "Median similar-z N<=600\":",
        f"{targets['redshift_neighbor_count'].median():.1f}"
    )


    X = targets[
        FEATURE_COLS
    ].to_numpy(float)

    n_features = len(
        FEATURE_COLS
    )

    def objective(w_raw):

        w = normalize_weights(
            w_raw
        )

        score = weighted_geometric_score(
            X,
            w
        )

        idx = np.argpartition(
            score,
            -TOP_K
        )[-TOP_K:]

        top_X = X[idx]

        mean_top = np.mean(
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

        neff = effective_number(
            w
        )

        penalty = max(
            0.0,
            MIN_EFF - neff
        ) / MIN_EFF

        value = (
            0.55 * mean_top
            +
            0.20 * diversity
            +
            0.20 * entropy
            -
            0.25 * penalty
        )

        return -value


    class_out = (
        OUT_ROOT / cls
    )

    class_out.mkdir(
        parents=True,
        exist_ok=True
    )

    run_rows = []
    weight_rows = []

    scores_by_seed = {}

    for seed in SEEDS:

        print()
        print(f"Optimizing seed={seed}")

        result = differential_evolution(
            objective,
            bounds=[
                (0, 1)
            ] * n_features,
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

        scores_by_seed[seed] = score

        idx = np.argpartition(
            score,
            -TOP_K
        )[-TOP_K:]

        neff = effective_number(
            w
        )

        entropy = entropy_of_weights(
            w
        )

        run_rows.append({
            "class": cls,
            "seed": seed,
            "objective": float(
                -result.fun
            ),
            "mean_top_score": float(
                score[idx].mean()
            ),
            "effective_features": float(
                neff
            ),
            "entropy": float(
                entropy
            ),
            "nfev": int(
                result.nfev
            ),
        })

        for feature, weight in zip(
            FEATURE_COLS,
            w
        ):
            weight_rows.append({
                "class": cls,
                "seed": seed,
                "feature": feature,
                "weight": float(
                    weight
                ),
            })

        print(
            "objective:",
            f"{-result.fun:.6f}"
        )

        print(
            "N_eff:",
            f"{neff:.4f}"
        )

        print(
            "weights:"
        )

        print(
            pd.DataFrame({
                "feature": FEATURE_COLS,
                "weight": w
            })
            .sort_values(
                "weight",
                ascending=False
            )
            .to_string(
                index=False
            )
        )


    runs = pd.DataFrame(
        run_rows
    )

    weights = pd.DataFrame(
        weight_rows
    )

    runs.to_csv(
        class_out / "runs.csv",
        index=False
    )

    weights.to_csv(
        class_out / "weights_all_seeds.csv",
        index=False
    )


    # -----------------------------------------
    # Seed 42 = canonical class-specific result
    # -----------------------------------------

    seed42_weights = (
        weights[
            weights["seed"] == 42
        ]
        .set_index("feature")[
            "weight"
        ]
    )

    w42 = np.array([
        seed42_weights[f]
        for f in FEATURE_COLS
    ])

    score42 = scores_by_seed[42]

    ranked = targets.copy()

    ranked["score"] = (
        score42
    )

    ranked = (
        ranked
        .sort_values(
            "score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    ranked[
        "rank"
    ] = np.arange(
        1,
        len(ranked) + 1
    )

    top = (
        ranked
        .head(TOP_K)
        .copy()
    )

    ranked.to_csv(
        class_out
        / f"ranked_{cls.lower()}.csv",
        index=False
    )

    top.to_csv(
        class_out
        / f"top_{cls.lower()}.csv",
        index=False
    )

    pd.DataFrame({
        "feature": FEATURE_COLS,
        "weight": w42
    }).to_csv(
        class_out
        / f"weights_{cls.lower()}_seed42.csv",
        index=False
    )


    # -----------------------------------------
    # Seed stability relative to 42
    # -----------------------------------------

    baseline_rank = pd.Series(
        score42
    ).rank(
        method="average"
    )

    baseline_top = set(
        np.argpartition(
            score42,
            -TOP_K
        )[-TOP_K:]
        .tolist()
    )

    stability = []

    for seed in SEEDS:

        s = scores_by_seed[seed]

        current_rank = pd.Series(
            s
        ).rank(
            method="average"
        )

        rho = spearmanr(
            baseline_rank,
            current_rank
        ).statistic

        current_top = set(
            np.argpartition(
                s,
                -TOP_K
            )[-TOP_K:]
            .tolist()
        )

        inter = len(
            baseline_top
            &
            current_top
        )

        union = len(
            baseline_top
            |
            current_top
        )

        stability.append({
            "seed": seed,
            "spearman_vs_seed42":
                float(rho),
            "top_overlap":
                inter,
            "jaccard_vs_seed42":
                float(inter / union),
        })

    stability = pd.DataFrame(
        stability
    )

    stability.to_csv(
        class_out
        / "seed_stability.csv",
        index=False
    )


    summary_rows.append({
        "class": cls,
        "n_targets": int(
            len(targets)
        ),
        "n_reference": int(
            len(reference)
        ),
        "pct_angular_neighbor":
            float(
                100
                *
                (
                    targets[
                        "neighbor_count_radius"
                    ] > 0
                ).mean()
            ),
        "pct_redshift_neighbor":
            float(
                100
                *
                (
                    targets[
                        "redshift_neighbor_count"
                    ] > 0
                ).mean()
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
        "median_redshift_neighbor_count":
            float(
                targets[
                    "redshift_neighbor_count"
                ].median()
            ),
        "objective_seed42":
            float(
                runs.loc[
                    runs["seed"] == 42,
                    "objective"
                ].iloc[0]
            ),
        "neff_seed42":
            float(
                runs.loc[
                    runs["seed"] == 42,
                    "effective_features"
                ].iloc[0]
            ),
        "mean_seed_jaccard":
            float(
                stability[
                    "jaccard_vs_seed42"
                ].mean()
            ),
        "min_seed_jaccard":
            float(
                stability[
                    "jaccard_vs_seed42"
                ].min()
            ),
        "mean_seed_spearman":
            float(
                stability[
                    "spearman_vs_seed42"
                ].mean()
            ),
        "min_seed_spearman":
            float(
                stability[
                    "spearman_vs_seed42"
                ].min()
            ),
    })

    all_weight_rows.extend(
        weight_rows
    )

    class_results[cls] = {
        "targets": targets,
        "ranked": ranked,
        "top": top,
        "w42": w42,
    }


# ==========================================================
# Cross-class summaries
# ==========================================================

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT_ROOT
    / "class_summary.csv",
    index=False
)


all_weights = pd.DataFrame(
    all_weight_rows
)

weight_summary = (
    all_weights
    .groupby([
        "class",
        "feature"
    ])["weight"]
    .agg([
        "mean",
        "std",
        "min",
        "max"
    ])
)

weight_summary.to_csv(
    OUT_ROOT
    / "class_weight_summary.csv"
)


# Canonical seed42 side-by-side
seed42_compare = (
    all_weights[
        all_weights["seed"] == 42
    ][
        [
            "class",
            "feature",
            "weight"
        ]
    ]
    .pivot(
        index="feature",
        columns="class",
        values="weight"
    )
)

if (
    "GALAXY" in seed42_compare.columns
    and
    "QSO" in seed42_compare.columns
):
    seed42_compare[
        "QSO_minus_GALAXY"
    ] = (
        seed42_compare["QSO"]
        -
        seed42_compare["GALAXY"]
    )

seed42_compare.to_csv(
    OUT_ROOT
    / "seed42_weight_comparison.csv"
)


print()
print("="*100)
print("CLASS-SPECIFIC SUMMARY")
print("="*100)

print(
    summary.to_string(
        index=False
    )
)

print()
print("="*100)
print("SEED-42 WEIGHT COMPARISON")
print("="*100)

print(
    seed42_compare.to_string()
)

print()
print("="*100)
print("ALL-SEED WEIGHT SUMMARY")
print("="*100)

print(
    weight_summary.to_string()
)

print()
print(
    "Saved to:",
    OUT_ROOT
)

print("DONE.")
