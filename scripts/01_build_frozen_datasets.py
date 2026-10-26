from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys

import pandas as pd
import numpy as np
from astroquery.gaia import Gaia
from astroquery.sdss import SDSS
import astroquery
import astropy


ROOT = Path.home() / "universe_revision" / "revision_v2"
OUT = ROOT / "data" / "raw" / "frozen"
OUT.mkdir(parents=True, exist_ok=True)

GAIA_QUERY_FILE = OUT / "gaia_query.adql"
SDSS_QUERY_FILE = OUT / "sdss_query.sql"


# ==========================================================
# GAIA
# ==========================================================

GAIA_QUERY = """
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
        CIRCLE('ICRS', 180.0, 30.0, 3.0)
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

GAIA_QUERY_FILE.write_text(GAIA_QUERY.strip() + "\n", encoding="utf-8")

print("=" * 70)
print("GAIA DR3")
print("=" * 70)
print("Downloading complete eligible Gaia population...")

job = Gaia.launch_job_async(
    GAIA_QUERY,
    dump_to_file=False
)

gaia_all = job.get_results().to_pandas()

print("Eligible downloaded:", len(gaia_all))

# Sanity checks
assert len(gaia_all) >= 10000, "Fewer than 10,000 eligible Gaia sources."
assert gaia_all["source_id"].nunique() == len(gaia_all), \
    "Duplicate Gaia source_id detected."
assert gaia_all["random_index"].notna().all(), \
    "Missing Gaia random_index."

# Deterministic local selection.
# random_index is supplied by Gaia and avoids relying on server row order.
gaia_all = (
    gaia_all
    .sort_values(["random_index", "source_id"], kind="mergesort")
    .reset_index(drop=True)
)

gaia = gaia_all.iloc[:10000].copy()

print("Frozen Gaia sample:", len(gaia))
print("Unique source_id:", gaia["source_id"].nunique())
print(
    "random_index range:",
    int(gaia["random_index"].min()),
    int(gaia["random_index"].max())
)

gaia_all_path = OUT / "gaia_eligible_population.csv"
gaia_path = OUT / "gaia_raw_frozen.csv"

gaia_all.to_csv(gaia_all_path, index=False)
gaia.to_csv(gaia_path, index=False)

gaia["source_id"].astype(str).to_csv(
    OUT / "gaia_source_ids.txt",
    index=False,
    header=False
)


# ==========================================================
# SDSS
# ==========================================================

SDSS_QUERY = """
SELECT TOP 50000
    p.objid,
    p.ra,
    p.dec,
    p.u,
    p.g,
    p.r,
    p.i,
    p.z,
    p.err_u,
    p.err_g,
    p.err_r,
    p.err_i,
    p.err_z,
    p.clean,
    p.mode,
    s.specobjid,
    s.class,
    s.z AS redshift,
    s.zErr AS redshift_error,
    s.zWarning
FROM PhotoObj AS p
JOIN SpecObj AS s
    ON p.objid = s.bestobjid
WHERE
    p.ra BETWEEN 120 AND 240
    AND p.dec BETWEEN 0 AND 60
    AND p.ra IS NOT NULL
    AND p.dec IS NOT NULL
    AND p.u BETWEEN 10 AND 30
    AND p.g BETWEEN 10 AND 30
    AND p.r BETWEEN 10 AND 30
    AND p.i BETWEEN 10 AND 30
    AND p.z BETWEEN 10 AND 30
    AND p.clean = 1
    AND p.mode = 1
    AND s.z IS NOT NULL
    AND s.zWarning = 0
    AND s.z > 0.01
    AND s.z < 5.0
    AND s.class IN ('GALAXY', 'QSO')
ORDER BY p.objid
"""

SDSS_QUERY_FILE.write_text(SDSS_QUERY.strip() + "\n", encoding="utf-8")

print()
print("=" * 70)
print("SDSS DR17")
print("=" * 70)
print("Downloading deterministic SDSS sample...")

table = SDSS.query_sql(
    SDSS_QUERY,
    data_release=17,
    timeout=600
)

if table is None:
    raise RuntimeError("SDSS returned no table.")

sdss = table.to_pandas()

print("Frozen SDSS sample:", len(sdss))
print("Unique objid:", sdss["objid"].nunique())
print("Unique specobjid:", sdss["specobjid"].nunique())

print("\nClass composition:")
print(sdss["class"].value_counts(dropna=False))

# Sanity checks
assert len(sdss) == 50000, \
    f"Expected 50,000 SDSS rows, obtained {len(sdss)}."
assert sdss["objid"].nunique() == len(sdss), \
    "Duplicate SDSS objid detected."
assert sdss["specobjid"].nunique() == len(sdss), \
    "Duplicate SDSS specobjid detected."
assert (sdss["clean"] == 1).all(), \
    "SDSS contains clean != 1."
assert (sdss["mode"] == 1).all(), \
    "SDSS contains mode != 1."
assert (sdss["zWarning"] == 0).all(), \
    "SDSS contains zWarning != 0."

sdss_path = OUT / "sdss_raw_frozen.csv"
sdss.to_csv(sdss_path, index=False)

sdss[["objid", "specobjid"]].to_csv(
    OUT / "sdss_object_ids.csv",
    index=False
)


# ==========================================================
# HASHES
# ==========================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


files_to_hash = [
    gaia_all_path,
    gaia_path,
    sdss_path,
    GAIA_QUERY_FILE,
    SDSS_QUERY_FILE,
    OUT / "gaia_source_ids.txt",
    OUT / "sdss_object_ids.csv",
]

hashes = {
    p.name: sha256_file(p)
    for p in files_to_hash
}

with open(OUT / "SHA256SUMS.txt", "w", encoding="utf-8") as f:
    for name, value in hashes.items():
        f.write(f"{value}  {name}\n")


# ==========================================================
# MANIFEST
# ==========================================================

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Frozen deterministic datasets for peer-review revision",
    "software": {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "astroquery": astroquery.__version__,
        "astropy": astropy.__version__,
    },
    "gaia": {
        "release": "Gaia DR3",
        "region": {
            "ra_deg": 180.0,
            "dec_deg": 30.0,
            "radius_deg": 3.0,
        },
        "eligible_population_n": int(len(gaia_all)),
        "frozen_sample_n": int(len(gaia)),
        "selection": (
            "Complete eligible field downloaded; "
            "locally sorted by random_index then source_id; "
            "first 10,000 frozen."
        ),
        "unique_source_id": int(gaia["source_id"].nunique()),
        "random_index_min": int(gaia["random_index"].min()),
        "random_index_max": int(gaia["random_index"].max()),
    },
    "sdss": {
        "release": "SDSS DR17",
        "frozen_sample_n": int(len(sdss)),
        "selection": (
            "PhotoObj joined to SpecObj using objid=bestobjid; "
            "clean=1; mode=1; zWarning=0; GALAXY/QSO; "
            "deterministic ORDER BY objid."
        ),
        "unique_objid": int(sdss["objid"].nunique()),
        "unique_specobjid": int(sdss["specobjid"].nunique()),
        "class_counts": {
            str(k): int(v)
            for k, v in sdss["class"].value_counts().items()
        },
    },
    "sha256": hashes,
}

with open(OUT / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print()
print("=" * 70)
print("FROZEN DATASETS COMPLETE")
print("=" * 70)
print("Output:", OUT)
print()
print("Files:")
for p in sorted(OUT.iterdir()):
    print(f"{p.name:32s} {p.stat().st_size / 1024 / 1024:10.2f} MB")

print()
print("SHA256:")
for name, value in hashes.items():
    print(name, value)

print()
print("DONE.")
