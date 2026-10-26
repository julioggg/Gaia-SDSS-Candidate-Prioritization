from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from astroquery.gaia import Gaia


ROOT = Path.home() / "universe_revision" / "revision_v2"

OUT_DIR = ROOT / "data" / "reference" / "gaia"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "gaia_reference_buffered.csv"
QUERY_FILE = OUT_DIR / "gaia_reference_buffered.adql"
META_FILE = OUT_DIR / "gaia_reference_buffered_metadata.json"

# ----------------------------------------------------------
# Geometry
# ----------------------------------------------------------

CENTER_RA = 180.0
CENTER_DEC = 30.0

CORE_RADIUS_DEG = 3.0

# 120 arcsec = 0.033333 deg.
# Use a conservative 0.10 deg buffer.
BUFFER_DEG = 0.10

REFERENCE_RADIUS_DEG = CORE_RADIUS_DEG + BUFFER_DEG


QUERY = f"""
SELECT
    source_id,
    ra,
    dec,
    parallax,
    parallax_error,
    pmra,
    pmdec,
    phot_g_mean_mag,
    phot_bp_mean_mag,
    phot_rp_mean_mag,
    bp_rp,
    ruwe,
    random_index
FROM gaiadr3.gaia_source
WHERE
    1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE(
            'ICRS',
            {CENTER_RA},
            {CENTER_DEC},
            {REFERENCE_RADIUS_DEG}
        )
    )
    AND ra IS NOT NULL
    AND dec IS NOT NULL
    AND phot_g_mean_mag IS NOT NULL
    AND phot_bp_mean_mag IS NOT NULL
    AND phot_rp_mean_mag IS NOT NULL
    AND bp_rp IS NOT NULL
    AND parallax IS NOT NULL
    AND parallax_error IS NOT NULL
    AND pmra IS NOT NULL
    AND pmdec IS NOT NULL
    AND ruwe IS NOT NULL
    AND parallax > 0
    AND parallax_error > 0
    AND ruwe < 1.4
    AND phot_g_mean_mag BETWEEN 8 AND 20
"""


QUERY_FILE.write_text(
    QUERY.strip() + "\n",
    encoding="utf-8"
)


print("=" * 78)
print("GAIA DR3 BUFFERED REFERENCE DOWNLOAD")
print("=" * 78)

print("Core radius:", CORE_RADIUS_DEG, "deg")
print("Buffer:", BUFFER_DEG, "deg")
print("Reference radius:", REFERENCE_RADIUS_DEG, "deg")

print()
print("Submitting asynchronous Gaia query...")


job = Gaia.launch_job_async(
    QUERY,
    dump_to_file=False
)

table = job.get_results()

print("Rows returned:", f"{len(table):,}")


df = table.to_pandas()

# ----------------------------------------------------------
# Preserve source identifiers safely
# ----------------------------------------------------------

df["source_id"] = (
    df["source_id"]
    .astype(str)
)

# Astroquery may return masked/object types.
numeric_cols = [
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

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ----------------------------------------------------------
# Audit duplicates
# ----------------------------------------------------------

n_before = len(df)

duplicate_source_ids = (
    df["source_id"]
    .duplicated(keep=False)
)

n_duplicate_rows = int(
    duplicate_source_ids.sum()
)

n_duplicate_ids = int(
    df.loc[
        duplicate_source_ids,
        "source_id"
    ].nunique()
)

if n_duplicate_rows > 0:

    audit = (
        df.loc[duplicate_source_ids]
        .sort_values("source_id")
    )

    audit.to_csv(
        OUT_DIR / "duplicate_source_id_audit.csv",
        index=False
    )

# Gaia source_id should be unique.
df = (
    df
    .drop_duplicates(
        subset=["source_id"],
        keep="first"
    )
    .reset_index(drop=True)
)

n_after = len(df)


# ----------------------------------------------------------
# Deterministic ordering
# ----------------------------------------------------------

df = df.sort_values(
    ["source_id"],
    kind="mergesort"
).reset_index(drop=True)


df.to_csv(
    OUT_CSV,
    index=False
)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


metadata = {
    "catalog": "Gaia DR3",
    "center_ra_deg": CENTER_RA,
    "center_dec_deg": CENTER_DEC,
    "core_radius_deg": CORE_RADIUS_DEG,
    "buffer_deg": BUFFER_DEG,
    "reference_radius_deg": REFERENCE_RADIUS_DEG,
    "neighbor_radius_arcsec": 120.0,
    "rows_downloaded": int(n_before),
    "rows_after_source_id_deduplication": int(n_after),
    "duplicate_rows": n_duplicate_rows,
    "duplicate_source_ids": n_duplicate_ids,
    "sha256_csv": sha256(OUT_CSV),
    "sha256_query": sha256(QUERY_FILE),
}

META_FILE.write_text(
    json.dumps(
        metadata,
        indent=2
    ),
    encoding="utf-8"
)


print()
print("=" * 78)
print("GAIA REFERENCE POOL COMPLETE")
print("=" * 78)

print("Rows downloaded:", f"{n_before:,}")
print(
    "Unique source_id:",
    f"{df['source_id'].nunique():,}"
)

print(
    "Duplicate source_id rows:",
    n_duplicate_rows
)

print()
print(
    "RA range:",
    df["ra"].min(),
    "to",
    df["ra"].max()
)

print(
    "Dec range:",
    df["dec"].min(),
    "to",
    df["dec"].max()
)

print()
print("Output:")
print(" ", OUT_CSV)

print()
print("SHA256:")
print(" ", metadata["sha256_csv"])

print()
print("Query:")
print(" ", QUERY_FILE)

print("DONE.")
