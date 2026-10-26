from pathlib import Path
import hashlib
import time

import pandas as pd
from astroquery.gaia import Gaia


ROOT = Path.home() / "universe_revision" / "revision_v2"

TARGET_FILE = (
    ROOT
    / "data"
    / "processed"
    / "gaia_features_reference.csv"
)

OUT = (
    ROOT
    / "data"
    / "external_validation"
    / "gaia"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

LABEL_FILE = (
    OUT
    / "gaia_external_labels_exact.csv"
)

QUERY_LOG = (
    OUT
    / "gaia_external_labels_queries.txt"
)


# ==========================================================
# Load exact definitive target population
# ==========================================================

targets = pd.read_csv(
    TARGET_FILE,
    dtype={"source_id": str}
)

target_ids = (
    targets["source_id"]
    .drop_duplicates()
    .astype(str)
    .tolist()
)

print("=" * 84)
print("EXACT GAIA EXTERNAL LABEL RETRIEVAL")
print("=" * 84)

print(
    "Target source_ids:",
    f"{len(target_ids):,}"
)


# ==========================================================
# Query in manageable chunks
# ==========================================================

CHUNK_SIZE = 400

all_frames = []
query_texts = []


for start in range(
    0,
    len(target_ids),
    CHUNK_SIZE
):

    stop = min(
        start + CHUNK_SIZE,
        len(target_ids)
    )

    chunk = target_ids[
        start:stop
    ]

    # source_id is numeric in Gaia ADQL,
    # therefore do not quote values.
    id_list = ",".join(chunk)

    query = f"""
SELECT
    source_id,
    phot_variable_flag,
    non_single_star,
    has_epoch_photometry
FROM gaiadr3.gaia_source
WHERE source_id IN ({id_list})
"""

    query_texts.append(
        query.strip()
    )

    print()
    print(
        f"Chunk {start:,}-{stop:,} "
        f"({len(chunk)} IDs)"
    )

    job = Gaia.launch_job_async(
        query,
        dump_to_file=False
    )

    tab = job.get_results()

    df = tab.to_pandas()

    if len(df) > 0:
        df["source_id"] = (
            df["source_id"]
            .astype(str)
        )

    print(
        "Rows returned:",
        f"{len(df):,}"
    )

    all_frames.append(df)

    # Small courtesy pause between TAP jobs.
    time.sleep(0.2)


# ==========================================================
# Combine
# ==========================================================

labels = pd.concat(
    all_frames,
    ignore_index=True
)

labels["source_id"] = (
    labels["source_id"]
    .astype(str)
)

labels = (
    labels
    .drop_duplicates(
        subset=["source_id"],
        keep="first"
    )
    .reset_index(drop=True)
)


# ==========================================================
# Audit exact coverage
# ==========================================================

returned_ids = set(
    labels["source_id"]
)

target_id_set = set(
    target_ids
)

missing = (
    target_id_set
    -
    returned_ids
)

extra = (
    returned_ids
    -
    target_id_set
)


# ==========================================================
# Construct external labels
# ==========================================================

labels["label_variable"] = (
    labels["phot_variable_flag"]
    .fillna("")
    .astype(str)
    .str.upper()
    .eq("VARIABLE")
    .astype(int)
)


labels["non_single_star"] = (
    pd.to_numeric(
        labels["non_single_star"],
        errors="coerce"
    )
    .fillna(0)
)


labels["label_nss"] = (
    labels["non_single_star"] > 0
).astype(int)


labels["label_variable_or_nss"] = (
    (
        labels["label_variable"] == 1
    )
    |
    (
        labels["label_nss"] == 1
    )
).astype(int)


labels.to_csv(
    LABEL_FILE,
    index=False
)


QUERY_LOG.write_text(
    "\n\n"
    + ("\n\n" + "="*80 + "\n\n").join(
        query_texts
    ),
    encoding="utf-8"
)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b""
        ):
            h.update(block)

    return h.hexdigest()


# ==========================================================
# Summary
# ==========================================================

print()
print("=" * 84)
print("EXACT GAIA LABEL AUDIT")
print("=" * 84)

print(
    "Definitive targets:",
    f"{len(target_id_set):,}"
)

print(
    "Unique returned:",
    f"{len(returned_ids):,}"
)

print(
    "Missing:",
    len(missing)
)

print(
    "Extra:",
    len(extra)
)

if missing:
    print()
    print(
        "First missing source_ids:"
    )

    print(
        sorted(missing)[:20]
    )


print()
print("-" * 84)

n = len(labels)

n_var = int(
    labels[
        "label_variable"
    ].sum()
)

n_nss = int(
    labels[
        "label_nss"
    ].sum()
)

n_union = int(
    labels[
        "label_variable_or_nss"
    ].sum()
)


print(
    "Variable:",
    n_var,
    f"({100*n_var/n:.3f}%)"
)

print(
    "NSS:",
    n_nss,
    f"({100*n_nss/n:.3f}%)"
)

print(
    "Variable OR NSS:",
    n_union,
    f"({100*n_union/n:.3f}%)"
)


print()
print(
    "phot_variable_flag values:"
)

print(
    labels[
        "phot_variable_flag"
    ]
    .fillna("<NA>")
    .value_counts(
        dropna=False
    )
    .to_string()
)


print()
print(
    "non_single_star > 0:",
    int(
        (
            labels[
                "non_single_star"
            ] > 0
        ).sum()
    )
)


print()
print(
    "SHA256:",
    sha256(LABEL_FILE)
)

print(
    "Saved:",
    LABEL_FILE
)

print("DONE.")
