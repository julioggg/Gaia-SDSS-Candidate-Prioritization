from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd
from astroquery.sdss import SDSS

ROOT = Path.home() / "universe_revision" / "revision_v2"
OUT = ROOT / "data" / "raw" / "frozen"
OUT.mkdir(parents=True, exist_ok=True)

QUERY = """
SELECT
    p.objid,
    p.htmid,
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
    s.specobjid,
    s.class,
    s.z AS redshift,
    s.zErr AS redshift_error,
    s.zWarning
FROM PhotoPrimary AS p
JOIN SpecObj AS s
    ON p.objid = s.bestobjid
WHERE
    p.ra BETWEEN 120 AND 240
    AND p.dec BETWEEN 0 AND 60
    AND p.u BETWEEN 10 AND 30
    AND p.g BETWEEN 10 AND 30
    AND p.r BETWEEN 10 AND 30
    AND p.i BETWEEN 10 AND 30
    AND p.z BETWEEN 10 AND 30
    AND p.clean = 1
    AND s.zWarning = 0
    AND s.z > 0.01
    AND s.z < 5.0
    AND s.class IN ('GALAXY', 'QSO')
    AND (p.htmid*37 & 0x000000000000FFFF) < 6500
"""

query_path = OUT / "sdss_query_htm.sql"
query_path.write_text(QUERY.strip() + "\n", encoding="utf-8")

print("=" * 72)
print("SDSS DR17 HTM PSEUDO-RANDOM FROZEN SAMPLE")
print("=" * 72)
print("Downloading HTM candidate population...")

table = SDSS.query_sql(
    QUERY,
    data_release=17,
    timeout=1200
)

if table is None:
    raise RuntimeError("SDSS returned no table.")

df = table.to_pandas()

print("Downloaded:", len(df))

# Preserve identifiers exactly as identifiers
df["objid"] = df["objid"].astype(str)
df["specobjid"] = df["specobjid"].astype(str)
df["htmid"] = df["htmid"].astype(str)

print("Unique objid:", df["objid"].nunique())
print("Unique specobjid:", df["specobjid"].nunique())

if df["objid"].duplicated().any():
    print("Removing duplicated objid...")
    df = df.drop_duplicates(subset="objid", keep="first")

if df["specobjid"].duplicated().any():
    print("Removing duplicated specobjid...")
    df = df.drop_duplicates(subset="specobjid", keep="first")

# Compute deterministic pseudo-random key locally using arbitrary-size ints
df["sample_key"] = df["htmid"].map(
    lambda x: (int(x) * 37) & 0xFFFF
)

# Deterministic ordering
df = (
    df.sort_values(
        ["sample_key", "objid"],
        kind="mergesort"
    )
    .reset_index(drop=True)
)

if len(df) < 50000:
    raise RuntimeError(
        f"Only {len(df)} usable rows after duplicate removal."
    )

candidate_path = OUT / "sdss_htm_candidate_population.csv"
frozen_path = OUT / "sdss_raw_frozen.csv"

df.to_csv(candidate_path, index=False)

frozen = df.iloc[:50000].copy()
frozen.to_csv(frozen_path, index=False)

frozen[["objid", "specobjid", "htmid", "sample_key"]].to_csv(
    OUT / "sdss_object_ids.csv",
    index=False
)

print()
print("Frozen rows:", len(frozen))
print("Unique objid:", frozen["objid"].nunique())
print("Unique specobjid:", frozen["specobjid"].nunique())

print()
print("Class composition:")
print(frozen["class"].value_counts())

print()
print("RA range:", frozen["ra"].min(), frozen["ra"].max())
print("DEC range:", frozen["dec"].min(), frozen["dec"].max())

print()
print("sample_key range:",
      frozen["sample_key"].min(),
      frozen["sample_key"].max())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


files = [
    candidate_path,
    frozen_path,
    query_path,
    OUT / "sdss_object_ids.csv",
]

hashes = {p.name: sha256(p) for p in files}

with open(OUT / "sdss_htm_SHA256SUMS.txt", "w") as f:
    for name, digest in hashes.items():
        f.write(f"{digest}  {name}\n")

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "release": "SDSS DR17",
    "source_view": "PhotoPrimary joined to SpecObj",
    "sampling": {
        "method": "deterministic HTM pseudo-random subsampling",
        "expression": "(htmid * 37) & 0xFFFF",
        "server_threshold": 6500,
        "candidate_population_n": int(len(df)),
        "frozen_sample_n": int(len(frozen)),
        "local_order": ["sample_key", "objid"],
    },
    "quality_filters": {
        "clean": 1,
        "zWarning": 0,
        "redshift_min": 0.01,
        "redshift_max": 5.0,
        "classes": ["GALAXY", "QSO"],
        "all_magnitudes_range": [10, 30],
    },
    "class_counts": {
        str(k): int(v)
        for k, v in frozen["class"].value_counts().items()
    },
    "spatial_extent": {
        "ra_min": float(frozen["ra"].min()),
        "ra_max": float(frozen["ra"].max()),
        "dec_min": float(frozen["dec"].min()),
        "dec_max": float(frozen["dec"].max()),
    },
    "sha256": hashes,
}

with open(OUT / "sdss_htm_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print()
print("SHA256:")
for name, digest in hashes.items():
    print(name, digest)

print()
print("DONE.")
