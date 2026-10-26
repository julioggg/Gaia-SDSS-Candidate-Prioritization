from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

from sklearn.neighbors import BallTree


ROOT = Path.home() / "universe_revision" / "revision_v2"

sys.path.insert(
    0,
    str(ROOT / "scripts")
)

from utils_features import (
    robust_01,
    inverse_01,
    save_feature_columns,
)


ARCSEC_PER_RAD = (
    180.0 / np.pi
) * 3600.0


parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    default="configs/gaia_config_reference.yaml"
)

args = parser.parse_args()


with open(
    args.config,
    "r",
    encoding="utf-8"
) as f:
    cfg = yaml.safe_load(f)


TARGET_FILE = Path(
    cfg["outputs"]["raw_csv"]
)

REFERENCE_FILE = Path(
    cfg["outputs"]["reference_csv"]
)

OUT_FILE = Path(
    cfg["outputs"]["processed_csv"]
)

META_FILE = Path(
    "data/processed/"
    "gaia_feature_columns_reference.json"
)

DIAG_FILE = Path(
    "outputs/baseline/"
    "gaia_reference_neighbor_diagnostics.json"
)


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


# ==========================================================
# Helpers
# ==========================================================

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
        if c in df.columns:
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
        + 5 * np.log10(
            df["parallax"]
        )
        - 10
    )

    df["pm_total"] = np.sqrt(
        df["pmra"]**2
        +
        df["pmdec"]**2
    )

    df["vtan_kms"] = (
        4.74047
        * df["pm_total"]
        / df["parallax"]
    )

    df["parallax_snr"] = (
        df["parallax"]
        / df["parallax_error"]
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

    return df


def coords_rad(df):
    return np.column_stack([
        np.deg2rad(
            df["dec"].to_numpy(float)
        ),
        np.deg2rad(
            df["ra"].to_numpy(float)
        ),
    ])


# ==========================================================
# Load
# ==========================================================

targets = pd.read_csv(
    TARGET_FILE,
    dtype={
        "source_id": str
    }
)

reference = pd.read_csv(
    REFERENCE_FILE,
    dtype={
        "source_id": str
    }
)


print("=" * 78)
print(
    "GAIA FEATURES WITH BUFFERED REFERENCE POPULATION"
)
print("=" * 78)

print(
    "Target file:",
    TARGET_FILE
)

print(
    "Reference file:",
    REFERENCE_FILE
)


targets = prepare_gaia(
    targets
)

reference = prepare_gaia(
    reference
)


# Deduplicate reference on Gaia identifier.
reference = (
    reference
    .drop_duplicates(
        subset=["source_id"],
        keep="first"
    )
    .reset_index(drop=True)
)

targets = (
    targets
    .drop_duplicates(
        subset=["source_id"],
        keep="first"
    )
    .reset_index(drop=True)
)


print()
print(
    "Targets after scientific filters:",
    len(targets)
)

print(
    "Reference after scientific filters:",
    len(reference)
)


# ==========================================================
# Audit
# ==========================================================

target_ids = set(
    targets["source_id"]
)

reference_ids = set(
    reference["source_id"]
)

present = (
    target_ids
    &
    reference_ids
)

missing = (
    target_ids
    -
    reference_ids
)

print(
    "Filtered targets present in reference:",
    f"{len(present)} / {len(targets)}"
)

if missing:
    raise RuntimeError(
        f"{len(missing)} filtered target IDs "
        "missing from filtered reference pool."
    )


# ==========================================================
# BallTree
# ==========================================================

target_coord = coords_rad(
    targets
)

reference_coord = coords_rad(
    reference
)

print()
print("Building Gaia BallTree...")

tree = BallTree(
    reference_coord,
    metric="haversine"
)


# ----------------------------------------------------------
# nearest + kth neighbor
# ----------------------------------------------------------

# Query extra objects because the target itself
# exists in the reference catalog.
QUERY_K = max(
    KTH_NEIGHBOR + 3,
    8
)

print(
    "Computing nearest and kth neighbors..."
)

distances, indices = tree.query(
    target_coord,
    k=QUERY_K
)

nn_sep = np.empty(
    len(targets),
    dtype=float
)

kth_sep = np.empty(
    len(targets),
    dtype=float
)


ref_ids_array = (
    reference["source_id"]
    .to_numpy()
)

target_ids_array = (
    targets["source_id"]
    .to_numpy()
)


for i in range(
    len(targets)
):
    candidate_idx = (
        indices[i]
    )

    candidate_dist = (
        distances[i]
    )

    keep = (
        ref_ids_array[
            candidate_idx
        ]
        != target_ids_array[i]
    )

    d = candidate_dist[
        keep
    ]

    if len(d) < KTH_NEIGHBOR:
        raise RuntimeError(
            "Not enough non-self neighbors "
            f"for target row {i}"
        )

    nn_sep[i] = (
        d[0]
        * ARCSEC_PER_RAD
    )

    kth_sep[i] = (
        d[KTH_NEIGHBOR - 1]
        * ARCSEC_PER_RAD
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
    / 60.0
)

targets[
    "local_density_knn"
] = (
    KTH_NEIGHBOR
    /
    (
        np.pi
        * np.maximum(
            kth_arcmin,
            1e-12
        ) ** 2
    )
)


# ----------------------------------------------------------
# fixed-radius counts
# ----------------------------------------------------------

print(
    "Computing fixed-radius neighbor counts..."
)

radius_rad = (
    RADIUS_ARCSEC
    /
    ARCSEC_PER_RAD
)

neighbor_lists = tree.query_radius(
    target_coord,
    r=radius_rad,
    return_distance=False
)

counts = np.empty(
    len(targets),
    dtype=int
)


for i, idx in enumerate(
    neighbor_lists
):
    ids = ref_ids_array[
        idx
    ]

    counts[i] = int(
        np.sum(
            ids
            != target_ids_array[i]
        )
    )


targets[
    "neighbor_count_radius"
] = counts


# ==========================================================
# Feature construction
# Same definitions as original manuscript pipeline
# ==========================================================

targets[
    "log_vtan"
] = np.log1p(
    targets["vtan_kms"]
)

targets[
    "f_astrometric_quality"
] = (
    0.50
    * inverse_01(
        targets["ruwe"]
    )
    +
    0.50
    * robust_01(
        targets["parallax_snr"]
    )
)

targets[
    "f_photometric_color_rarity"
] = robust_01(
    abs(
        targets["bp_rp"]
        -
        targets["bp_rp"].median()
    )
)

targets[
    "f_absolute_magnitude_rarity"
] = robust_01(
    abs(
        targets["abs_g_mag"]
        -
        targets[
            "abs_g_mag"
        ].median()
    )
)

targets[
    "f_kinematic_interest"
] = robust_01(
    targets["log_vtan"]
)

targets[
    "f_local_density"
] = (
    0.70
    * robust_01(
        targets[
            "local_density_knn"
        ]
    )
    +
    0.30
    * robust_01(
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


feature_cols = [
    "f_astrometric_quality",
    "f_photometric_color_rarity",
    "f_absolute_magnitude_rarity",
    "f_kinematic_interest",
    "f_local_density",
    "f_isolation",
]


targets = targets.replace(
    [np.inf, -np.inf],
    np.nan
)

targets = targets.dropna(
    subset=feature_cols
)


# ==========================================================
# Save
# ==========================================================

OUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

DIAG_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

targets.to_csv(
    OUT_FILE,
    index=False
)

save_feature_columns(
    META_FILE,
    feature_cols
)


diagnostics = {
    "targets": int(
        len(targets)
    ),
    "reference_after_filters": int(
        len(reference)
    ),
    "neighbor_radius_arcsec":
        RADIUS_ARCSEC,
    "kth_neighbor":
        KTH_NEIGHBOR,
    "percent_with_neighbor":
        float(
            100
            * np.mean(
                targets[
                    "neighbor_count_radius"
                ] > 0
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
}


DIAG_FILE.write_text(
    json.dumps(
        diagnostics,
        indent=2
    ),
    encoding="utf-8"
)


print()
print("=" * 78)
print(
    "REFERENCE-BASED GAIA FEATURES COMPLETE"
)
print("=" * 78)

print(
    "Features Gaia:",
    len(targets)
)

print(
    "% with neighbor <=120\":",
    f"{diagnostics['percent_with_neighbor']:.2f}%"
)

print(
    "Median nearest-neighbor separation:",
    f"{diagnostics['median_nn_arcsec']:.3f}\""
)

print(
    "Median kth-neighbor separation:",
    f"{diagnostics['median_kth_arcsec']:.3f}\""
)

print(
    "Median neighbors <=120\":",
    f"{diagnostics['median_neighbor_count']:.1f}"
)

print()
print(
    "Output:",
    OUT_FILE
)

print(
    "Diagnostics:",
    DIAG_FILE
)

print("DONE.")
