from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


OUT_ROOT = Path(
    "outputs/anomaly_baselines"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


DATASETS = {
    "gaia": {
        "file":
            "data/processed/gaia_features_reference.csv",
        "id": "source_id",
        "features": [
            "f_astrometric_quality",
            "f_photometric_color_rarity",
            "f_absolute_magnitude_rarity",
            "f_kinematic_interest",
            "f_local_density",
            "f_isolation",
        ],
    },

    "sdss_mixed": {
        "file":
            "data/processed/sdss_features_reference.csv",
        "id": "objid",
        "features": [
            "f_photometric_quality",
            "f_color_rarity",
            "f_redshift_interest",
            "f_redshift_precision",
            "f_local_density",
            "f_isolation",
        ],
    },

    "sdss_galaxy": {
        "file":
            "data/processed/class_specific/"
            "sdss_features_galaxy_reference.csv",
        "id": "objid",
        "features": [
            "f_photometric_quality",
            "f_color_rarity",
            "f_redshift_interest",
            "f_redshift_precision",
            "f_local_density",
            "f_isolation",
        ],
    },

    "sdss_qso": {
        "file":
            "data/processed/class_specific/"
            "sdss_features_qso_reference.csv",
        "id": "objid",
        "features": [
            "f_photometric_quality",
            "f_color_rarity",
            "f_redshift_interest",
            "f_redshift_precision",
            "f_local_density",
            "f_isolation",
        ],
    },
}


TOP_K = 1000


def rank_from_score(df, score, id_col):
    out = df.copy()

    out["anomaly_score"] = score

    out = (
        out
        .sort_values(
            "anomaly_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    out["rank"] = (
        np.arange(
            1,
            len(out) + 1
        )
    )

    return out


summary_rows = []


for name, info in DATASETS.items():

    print()
    print("=" * 90)
    print("DATASET:", name.upper())
    print("=" * 90)

    df = pd.read_csv(
        info["file"],
        dtype={
            info["id"]: str
        }
    )

    features = info["features"]

    X = (
        df[features]
        .to_numpy(float)
    )

    if not np.isfinite(X).all():
        raise RuntimeError(
            f"Non-finite values in {name}"
        )

    out_dir = (
        OUT_ROOT / name
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "N:",
        f"{len(df):,}"
    )

    print(
        "Features:",
        features
    )

    # ======================================================
    # 1. Isolation Forest
    # ======================================================

    print()
    print("Isolation Forest...")

    iso_scores = []

    for seed in [
        0, 1, 2, 42
    ]:

        model = IsolationForest(
            n_estimators=500,
            contamination="auto",
            random_state=seed,
            n_jobs=1,
            max_samples="auto",
        )

        model.fit(X)

        # sklearn decision_function:
        # larger = more normal.
        # We invert so larger = more anomalous.
        score = (
            -model.decision_function(X)
        )

        iso_scores.append(
            score
        )

    iso_matrix = np.vstack(
        iso_scores
    )

    iso_mean = (
        iso_matrix.mean(axis=0)
    )

    iso_ranked = rank_from_score(
        df,
        iso_mean,
        info["id"]
    )

    iso_ranked.to_csv(
        out_dir
        / "isolation_forest_ranked.csv",
        index=False
    )

    iso_ranked.head(
        min(TOP_K, len(iso_ranked))
    ).to_csv(
        out_dir
        / "isolation_forest_top1000.csv",
        index=False
    )

    # Stability across IF seeds
    rank_matrix = []

    for score in iso_scores:

        r = (
            pd.Series(score)
            .rank(
                ascending=False,
                method="average"
            )
            .to_numpy()
        )

        rank_matrix.append(r)

    rank_matrix = np.vstack(
        rank_matrix
    )

    correlations = []

    ref_rank = rank_matrix[-1]

    for i in range(
        len(rank_matrix) - 1
    ):
        correlations.append(
            np.corrcoef(
                ref_rank,
                rank_matrix[i]
            )[0, 1]
        )

    # ======================================================
    # 2. Local Outlier Factor
    # ======================================================

    print("Local Outlier Factor...")

    lof = LocalOutlierFactor(
        n_neighbors=35,
        contamination="auto",
        metric="euclidean",
        n_jobs=1,
    )

    lof.fit_predict(X)

    # More negative = more anomalous,
    # invert sign.
    lof_score = (
        -lof.negative_outlier_factor_
    )

    lof_ranked = rank_from_score(
        df,
        lof_score,
        info["id"]
    )

    lof_ranked.to_csv(
        out_dir
        / "lof_ranked.csv",
        index=False
    )

    lof_ranked.head(
        min(TOP_K, len(lof_ranked))
    ).to_csv(
        out_dir
        / "lof_top1000.csv",
        index=False
    )

    # ======================================================
    # 3. PCA + Mahalanobis distance
    # ======================================================

    print("PCA + Mahalanobis...")

    scaler = StandardScaler()

    Xs = scaler.fit_transform(
        X
    )

    pca = PCA(
        n_components=None,
        svd_solver="full"
    )

    Z = pca.fit_transform(
        Xs
    )

    var = (
        pca.explained_variance_
    )

    # Mahalanobis distance in decorrelated
    # PCA space.
    md2 = np.sum(
        (Z**2)
        /
        np.maximum(
            var,
            1e-12
        ),
        axis=1
    )

    mahal_score = np.sqrt(
        md2
    )

    mahal_ranked = rank_from_score(
        df,
        mahal_score,
        info["id"]
    )

    mahal_ranked.to_csv(
        out_dir
        / "pca_mahalanobis_ranked.csv",
        index=False
    )

    mahal_ranked.head(
        min(TOP_K, len(mahal_ranked))
    ).to_csv(
        out_dir
        / "pca_mahalanobis_top1000.csv",
        index=False
    )

    # ======================================================
    # Pairwise top-K overlaps
    # ======================================================

    K = min(
        TOP_K,
        len(df)
    )

    sets = {
        "IsolationForest":
            set(
                iso_ranked
                .head(K)[info["id"]]
            ),

        "LOF":
            set(
                lof_ranked
                .head(K)[info["id"]]
            ),

        "PCA_Mahalanobis":
            set(
                mahal_ranked
                .head(K)[info["id"]]
            ),
    }

    methods = list(
        sets.keys()
    )

    overlap_rows = []

    for i in range(
        len(methods)
    ):
        for j in range(
            i + 1,
            len(methods)
        ):

            a = methods[i]
            b = methods[j]

            inter = len(
                sets[a]
                &
                sets[b]
            )

            union = len(
                sets[a]
                |
                sets[b]
            )

            overlap_rows.append({
                "method_a": a,
                "method_b": b,
                "intersection_topk":
                    inter,
                "jaccard_topk":
                    inter / union,
            })

    overlaps = pd.DataFrame(
        overlap_rows
    )

    overlaps.to_csv(
        out_dir
        / "baseline_topk_overlap.csv",
        index=False
    )

    summary_rows.append({
        "dataset": name,
        "n_objects": len(df),
        "n_features": len(features),
        "if_seed_rank_correlation_mean":
            float(
                np.mean(
                    correlations
                )
            ),
        "if_seed_rank_correlation_min":
            float(
                np.min(
                    correlations
                )
            ),
        "pca_variance_pc1":
            float(
                pca.explained_variance_ratio_[0]
            ),
        "pca_variance_first3":
            float(
                pca
                .explained_variance_ratio_[
                    :3
                ]
                .sum()
            ),
    })

    print()
    print("Baseline top-K overlaps:")
    print(
        overlaps.to_string(
            index=False
        )
    )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT_ROOT
    / "summary.csv",
    index=False
)


print()
print("=" * 100)
print("ANOMALY BASELINE SUMMARY")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)

print()
print(
    "Saved to:",
    OUT_ROOT
)

print("DONE.")
