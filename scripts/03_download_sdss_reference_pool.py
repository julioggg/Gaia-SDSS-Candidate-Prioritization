from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time

import pandas as pd
from astroquery.sdss import SDSS

ROOT = Path.home() / "universe_revision" / "revision_v2"
OUT = ROOT / "data" / "reference" / "sdss"
CHUNKS = OUT / "chunks"

OUT.mkdir(parents=True, exist_ok=True)
CHUNKS.mkdir(parents=True, exist_ok=True)

RA_MIN = 119.5
RA_MAX = 240.5
STEP = 5.0

EXPECTED_N = 1732111


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_query(lo, hi, last=False):
    upper_op = "<=" if last else "<"

    return f"""
SELECT
    p.objid,
    p.ra,
    p.dec,
    s.specobjid,
    s.class,
    s.z AS redshift
FROM PhotoPrimary AS p
JOIN SpecObj AS s
    ON p.objid = s.bestobjid
WHERE
    p.ra >= {lo:.6f}
    AND p.ra {upper_op} {hi:.6f}
    AND p.dec BETWEEN -0.25 AND 60.25
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
"""


edges = []
x = RA_MIN
while x < RA_MAX:
    hi = min(x + STEP, RA_MAX)
    edges.append((x, hi))
    x = hi

print("=" * 72)
print("SDSS DR17 BUFFERED REFERENCE POOL")
print("=" * 72)
print("RA chunks:", len(edges))
print("Expected global count:", EXPECTED_N)
print()

chunk_files = []
chunk_meta = []

for i, (lo, hi) in enumerate(edges, start=1):

    last = i == len(edges)

    chunk_path = CHUNKS / f"sdss_ref_{i:03d}_{lo:.1f}_{hi:.1f}.csv"
    query_path = CHUNKS / f"sdss_ref_{i:03d}_{lo:.1f}_{hi:.1f}.sql"

    query = build_query(lo, hi, last=last)
    query_path.write_text(query.strip() + "\n", encoding="utf-8")

    # Resume support
    if chunk_path.exists():
        try:
            old = pd.read_csv(
                chunk_path,
                usecols=["objid", "ra", "dec", "specobjid", "class", "redshift"]
            )
            print(
                f"[{i:02d}/{len(edges)}] "
                f"EXISTS RA {lo:.1f}-{hi:.1f}: {len(old):,}"
            )

            chunk_files.append(chunk_path)

            chunk_meta.append({
                "chunk": i,
                "ra_min": lo,
                "ra_max": hi,
                "rows": int(len(old)),
                "file": chunk_path.name
            })

            continue

        except Exception:
            print("Existing chunk invalid; downloading again:", chunk_path.name)

    print(
        f"[{i:02d}/{len(edges)}] "
        f"Downloading RA {lo:.1f}-{hi:.1f}..."
    )

    table = None
    last_error = None

    for attempt in range(1, 4):
        try:
            table = SDSS.query_sql(
                query,
                data_release=17,
                timeout=900
            )

            if table is None:
                raise RuntimeError("SDSS returned None.")

            break

        except Exception as e:
            last_error = e
            print(f"  attempt {attempt}/3 failed: {e}")

            if attempt < 3:
                time.sleep(10 * attempt)

    if table is None:
        raise RuntimeError(
            f"Chunk {i} failed permanently: {last_error}"
        )

    df = table.to_pandas()

    # Preserve large identifiers safely
    df["objid"] = df["objid"].astype(str)
    df["specobjid"] = df["specobjid"].astype(str)

    df.to_csv(chunk_path, index=False)

    print("  rows:", f"{len(df):,}")

    chunk_files.append(chunk_path)

    chunk_meta.append({
        "chunk": i,
        "ra_min": lo,
        "ra_max": hi,
        "rows": int(len(df)),
        "file": chunk_path.name
    })


print()
print("Combining chunks...")

dfs = [
    pd.read_csv(
        p,
        dtype={
            "objid": str,
            "specobjid": str,
            "class": str
        }
    )
    for p in chunk_files
]

ref = pd.concat(dfs, ignore_index=True)

print("Rows before duplicate check:", f"{len(ref):,}")
print("Unique objid:", f"{ref['objid'].nunique():,}")
print("Unique specobjid:", f"{ref['specobjid'].nunique():,}")

dup_obj = int(ref["objid"].duplicated().sum())
dup_spec = int(ref["specobjid"].duplicated().sum())

print("Duplicated objid:", dup_obj)
print("Duplicated specobjid:", dup_spec)

# Only remove exact duplicate identifiers if they occur
if dup_obj:
    ref = ref.drop_duplicates("objid", keep="first")

if ref["specobjid"].duplicated().any():
    ref = ref.drop_duplicates("specobjid", keep="first")

ref = ref.sort_values(
    ["ra", "dec", "objid"],
    kind="mergesort"
).reset_index(drop=True)

final_path = OUT / "sdss_reference_buffered.csv"
ref.to_csv(final_path, index=False)

print()
print("FINAL REFERENCE POOL")
print("Rows:", f"{len(ref):,}")
print("GALAXY:", f"{(ref['class'] == 'GALAXY').sum():,}")
print("QSO:", f"{(ref['class'] == 'QSO').sum():,}")
print("RA:", ref["ra"].min(), ref["ra"].max())
print("Dec:", ref["dec"].min(), ref["dec"].max())

digest = sha256(final_path)

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "catalog": "SDSS DR17",
    "purpose": (
        "Buffered reference population used exclusively "
        "for local-neighborhood calculations."
    ),
    "target_region": {
        "ra_min": 120.0,
        "ra_max": 240.0,
        "dec_min": 0.0,
        "dec_max": 60.0
    },
    "reference_region": {
        "ra_min": 119.5,
        "ra_max": 240.5,
        "dec_min": -0.25,
        "dec_max": 60.25
    },
    "maximum_neighborhood_radius_arcsec": 600.0,
    "expected_count_from_precheck": EXPECTED_N,
    "downloaded_unique_count": int(len(ref)),
    "duplicate_objid_detected": dup_obj,
    "duplicate_specobjid_detected": dup_spec,
    "class_counts": {
        str(k): int(v)
        for k, v in ref["class"].value_counts().items()
    },
    "chunks": chunk_meta,
    "sha256": digest
}

with open(OUT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

with open(OUT / "SHA256SUMS.txt", "w") as f:
    f.write(f"{digest}  {final_path.name}\n")

print()
print("File:", final_path)
print("SHA256:", digest)

if len(ref) != EXPECTED_N:
    print()
    print("WARNING:")
    print(
        f"Pre-count was {EXPECTED_N:,}, "
        f"but final unique pool has {len(ref):,} rows."
    )
    print(
        "This is not automatically an error; "
        "inspect duplicates and query timing."
    )
else:
    print()
    print("COUNT MATCHES PRECHECK EXACTLY.")

print()
print("DONE.")
