from pathlib import Path
import hashlib

import pandas as pd
from astroquery.gaia import Gaia


ROOT = Path.home() / "universe_revision" / "revision_v2"

OUT = ROOT / "data" / "external_validation" / "gaia"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_FILE = (
    ROOT
    / "data"
    / "processed"
    / "gaia_features_reference.csv"
)

LABEL_FILE = OUT / "gaia_external_labels.csv"
QUERY_FILE = OUT / "gaia_external_labels.adql"


QUERY = """
SELECT
    source_id,
    phot_variable_flag,
    non_single_star,
    has_epoch_photometry
FROM gaiadr3.gaia_source
WHERE
    1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', 180.0, 30.0, 3.0)
    )
"""

QUERY_FILE.write_text(
    QUERY.strip() + "\n",
    encoding="utf-8"
)


print("=" * 82)
print("GAIA EXTERNAL VALIDATION LABEL DOWNLOAD")
print("=" * 82)

job = Gaia.launch_job_async(
    QUERY,
    dump_to_file=False
)

tab = job.get_results()

df = tab.to_pandas()

df["source_id"] = (
    df["source_id"]
    .astype(str)
)

print(
    "Rows returned:",
    f"{len(df):,}"
)


# ==========================================================
# Restrict to the definitive 5,205 targets
# ==========================================================

targets = pd.read_csv(
    TARGET_FILE,
    dtype={"source_id": str}
)

target_ids = set(
    targets["source_id"]
)

df = (
    df[
        df["source_id"].isin(
            target_ids
        )
    ]
    .drop_duplicates(
        subset=["source_id"]
    )
    .copy()
)


# ==========================================================
# Binary validation labels
# ==========================================================

df["label_variable"] = (
    df["phot_variable_flag"]
    .astype(str)
    .str.upper()
    .eq("VARIABLE")
    .astype(int)
)

df["non_single_star"] = pd.to_numeric(
    df["non_single_star"],
    errors="coerce"
).fillna(0)

df["label_nss"] = (
    df["non_single_star"] > 0
).astype(int)

df["label_variable_or_nss"] = (
    (
        df["label_variable"] == 1
    )
    |
    (
        df["label_nss"] == 1
    )
).astype(int)


df.to_csv(
    LABEL_FILE,
    index=False
)


missing = (
    target_ids
    -
    set(df["source_id"])
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


print()
print("=" * 82)
print("GAIA LABEL AUDIT")
print("=" * 82)

print(
    "Definitive Gaia targets:",
    f"{len(target_ids):,}"
)

print(
    "Targets with downloaded label row:",
    f"{len(df):,}"
)

print(
    "Missing:",
    len(missing)
)

print()
print(
    "Variable:",
    int(
        df["label_variable"].sum()
    ),
    f"({100*df['label_variable'].mean():.2f}%)"
)

print(
    "NSS:",
    int(
        df["label_nss"].sum()
    ),
    f"({100*df['label_nss'].mean():.2f}%)"
)

print(
    "Variable OR NSS:",
    int(
        df[
            "label_variable_or_nss"
        ].sum()
    ),
    f"({100*df['label_variable_or_nss'].mean():.2f}%)"
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
