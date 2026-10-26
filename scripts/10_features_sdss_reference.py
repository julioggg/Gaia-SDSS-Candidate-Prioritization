from pathlib import Path
import argparse
import json
import yaml
import numpy as np
import pandas as pd

import sys

ROOT = Path.home() / "universe_revision" / "revision_v2"

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts_revision"))

from utils_features import (
    clean_numeric,
    robust_01,
    inverse_01,
    save_feature_columns,
)

from reference_neighbors import add_reference_neighbor_features


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default="configs/sdss_config_reference.yaml"
)
args = parser.parse_args()


with open(args.config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


IN_FILE = Path(cfg["outputs"]["raw_csv"])
OUT_FILE = Path(cfg["outputs"]["processed_csv"])

REFERENCE_FILE = Path(
    cfg["outputs"]["reference_csv"]
)

META_FILE = Path(
    "data/processed/sdss_feature_columns_reference.json"
)

DIAG_FILE = Path(
    "outputs/baseline/sdss_reference_neighbor_diagnostics.json"
)


NEIGHBOR_RADIUS_ARCSEC = float(
    cfg["match_internal"]["angular_radius_arcsec"]
)

KTH_NEIGHBOR = int(
    cfg["match_internal"]["kth_neighbor"]
)

REDSHIFT_RADIUS_ARCSEC = float(
    cfg["match_internal"]["redshift_radius_arcsec"]
)

REDSHIFT_DZ = float(
    cfg["match_internal"]["dz"]
)

MIN_Z = float(
    cfg["scientific_filter"]["min_redshift"]
)

MAX_Z = float(
    cfg["scientific_filter"]["max_redshift"]
)


print("=" * 72)
print("SDSS FEATURES WITH BUFFERED REFERENCE POPULATION")
print("=" * 72)

print("Target file:")
print(" ", IN_FILE)

print("Reference file:")
print(" ", REFERENCE_FILE)


# ==========================================================
# TARGETS
# ==========================================================

df = pd.read_csv(
    IN_FILE,
    dtype={
        "objid": str,
        "specobjid": str,
        "htmid": str,
    }
)

num_cols = [
    "ra",
    "dec",
    "u",
    "g",
    "r",
    "i",
    "z",
    "err_u",
    "err_g",
    "err_r",
    "err_i",
    "err_z",
    "redshift",
    "redshift_error",
    "zWarning",
]

df = clean_numeric(df, num_cols)

df = df.dropna(
    subset=[
        "ra",
        "dec",
        "u",
        "g",
        "r",
        "i",
        "z",
        "redshift",
    ]
)

if "class" in df.columns:
    df["class"] = (
        df["class"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df = df[
        df["class"].isin(["GALAXY", "QSO"])
    ].copy()


df = df[
    (df["u"] > 0)
    & (df["g"] > 0)
    & (df["r"] > 0)
    & (df["i"] > 0)
    & (df["z"] > 0)
    & (df["redshift"] > MIN_Z)
    & (df["redshift"] < MAX_Z)
].copy()


print()
print("Targets after scientific filters:", len(df))


# ==========================================================
# REFERENCE POPULATION
# ==========================================================

ref = pd.read_csv(
    REFERENCE_FILE,
    dtype={
        "objid": str,
        "specobjid": str,
        "class": str,
    }
)

ref = clean_numeric(
    ref,
    ["ra", "dec", "redshift"]
)

ref = ref.dropna(
    subset=["ra", "dec"]
).copy()

# One physical photometric object = one spatial reference position.
ref = ref.drop_duplicates(
    subset="objid",
    keep="first"
).copy()


print("Reference objects:", len(ref))

target_in_reference = df["objid"].isin(
    set(ref["objid"])
)

print(
    "Targets present in reference:",
    int(target_in_reference.sum()),
    "/",
    len(df)
)

if not target_in_reference.all():
    missing = df.loc[
        ~target_in_reference,
        "objid"
    ].tolist()[:10]

    raise RuntimeError(
        "Some targets are absent from reference pool. "
        f"Examples: {missing}"
    )


# ==========================================================
# PHOTOMETRIC / SCIENTIFIC FEATURES
# ==========================================================

df["u_g"] = df["u"] - df["g"]
df["g_r"] = df["g"] - df["r"]
df["r_i"] = df["r"] - df["i"]
df["i_z"] = df["i"] - df["z"]


err_cols = [
    "err_u",
    "err_g",
    "err_r",
    "err_i",
    "err_z",
]

df["mean_phot_error"] = (
    df[err_cols]
    .replace([np.inf, -np.inf], np.nan)
    .mean(axis=1)
)


# ==========================================================
# REFERENCE-BASED NEIGHBOR FEATURES
# ==========================================================

df = add_reference_neighbor_features(
    targets=df,
    reference=ref,
    radius_arcsec=NEIGHBOR_RADIUS_ARCSEC,
    kth_neighbor=KTH_NEIGHBOR,
    redshift_radius_arcsec=REDSHIFT_RADIUS_ARCSEC,
    dz=REDSHIFT_DZ,
    batch_size=2000,
)


# ==========================================================
# REMAINING ORIGINAL FEATURE DEFINITIONS
# ==========================================================

color_cols = [
    "u_g",
    "g_r",
    "r_i",
    "i_z",
]

color_distance = np.zeros(len(df))

for c in color_cols:
    color_distance += (
        df[c] - df[c].median()
    ) ** 2

df["color_distance"] = np.sqrt(color_distance)

df["log_redshift"] = np.log1p(
    df["redshift"]
)


df["f_photometric_quality"] = inverse_01(
    df["mean_phot_error"]
)

df["f_color_rarity"] = robust_01(
    df["color_distance"]
)

df["f_redshift_interest"] = robust_01(
    df["log_redshift"]
)

df["f_redshift_precision"] = inverse_01(
    df["redshift_error"]
)

df["f_local_density"] = (
    0.50 * robust_01(df["local_density_knn"])
    + 0.25 * robust_01(df["neighbor_count_radius"])
    + 0.25 * robust_01(df["redshift_neighbor_count"])
)

df["f_isolation"] = robust_01(
    df["nn_sep_arcsec"]
)


feature_cols = [
    "f_photometric_quality",
    "f_color_rarity",
    "f_redshift_interest",
    "f_redshift_precision",
    "f_local_density",
    "f_isolation",
]


df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=feature_cols
)


# ==========================================================
# SAVE
# ==========================================================

OUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

DIAG_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUT_FILE,
    index=False
)

save_feature_columns(
    META_FILE,
    feature_cols
)


diagnostics = {
    "targets_n": int(len(df)),
    "reference_n": int(len(ref)),
    "angular_radius_arcsec": NEIGHBOR_RADIUS_ARCSEC,
    "redshift_radius_arcsec": REDSHIFT_RADIUS_ARCSEC,
    "dz": REDSHIFT_DZ,
    "kth_neighbor": KTH_NEIGHBOR,
    "pct_with_angular_neighbor": float(
        (df["neighbor_count_radius"] > 0).mean() * 100
    ),
    "pct_with_redshift_neighbor": float(
        (df["redshift_neighbor_count"] > 0).mean() * 100
    ),
    "median_nn_arcsec": float(
        df["nn_sep_arcsec"].median()
    ),
    "median_kth_arcsec": float(
        df["kth_sep_arcsec"].median()
    ),
    "median_neighbor_count_300": float(
        df["neighbor_count_radius"].median()
    ),
    "median_redshift_neighbor_count_600": float(
        df["redshift_neighbor_count"].median()
    ),
}


with open(
    DIAG_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        diagnostics,
        f,
        indent=2
    )


print()
print("=" * 72)
print("REFERENCE-BASED SDSS FEATURES COMPLETE")
print("=" * 72)

print("Features SDSS:", len(df))

print()
print("Class composition:")
print(
    df["class"]
    .value_counts(dropna=False)
    .to_string()
)

print()
print(
    "% with neighbor <= "
    f"{NEIGHBOR_RADIUS_ARCSEC:.0f}\" : "
    f"{diagnostics['pct_with_angular_neighbor']:.2f}%"
)

print(
    "% with similar-z neighbor <= "
    f"{REDSHIFT_RADIUS_ARCSEC:.0f}\" : "
    f"{diagnostics['pct_with_redshift_neighbor']:.2f}%"
)

print(
    "Median nearest-neighbor separation:",
    f"{diagnostics['median_nn_arcsec']:.3f} arcsec"
)

print(
    "Median kth-neighbor separation:",
    f"{diagnostics['median_kth_arcsec']:.3f} arcsec"
)

print(
    "Median neighbors <=300\":",
    diagnostics["median_neighbor_count_300"]
)

print(
    "Median similar-z neighbors <=600\":",
    diagnostics["median_redshift_neighbor_count_600"]
)

print()
print("Output:", OUT_FILE)
print("Diagnostics:", DIAG_FILE)
print("DONE.")
