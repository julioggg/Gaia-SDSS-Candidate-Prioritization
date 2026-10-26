from pathlib import Path
import hashlib
import time
import pandas as pd
from astroquery.gaia import Gaia

ROOT = Path.home() / "universe_revision" / "revision_v2"

TARGET_FILE = ROOT / "data/processed/gaia_features_reference.csv"
OUT = ROOT / "data/external_validation/gaia"
OUT.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT / "gaia_extended_external_labels_exact.csv"
QUERY_LOG = OUT / "gaia_extended_external_labels_queries.txt"

targets = pd.read_csv(
    TARGET_FILE,
    dtype={"source_id": str}
)

ids = (
    targets["source_id"]
    .drop_duplicates()
    .astype(str)
    .tolist()
)

CHUNK = 400
frames = []
queries = []

print("="*90)
print("GAIA EXTENDED EXTERNAL LABEL RETRIEVAL")
print("="*90)
print("Targets:", len(ids))

for start in range(0, len(ids), CHUNK):
    stop = min(start + CHUNK, len(ids))
    chunk = ids[start:stop]

    id_list = ",".join(chunk)

    query = f"""
SELECT
    source_id,
    phot_variable_flag,
    non_single_star,
    in_qso_candidates,
    in_galaxy_candidates,
    has_epoch_photometry,
    has_xp_continuous,
    has_xp_sampled,
    has_rvs,
    classprob_dsc_combmod_quasar,
    classprob_dsc_combmod_galaxy,
    classprob_dsc_combmod_star
FROM gaiadr3.gaia_source
WHERE source_id IN ({id_list})
"""

    queries.append(query.strip())

    print(f"Chunk {start:,}-{stop:,}")

    job = Gaia.launch_job_async(
        query,
        dump_to_file=False
    )

    df = job.get_results().to_pandas()

    if len(df):
        df["source_id"] = df["source_id"].astype(str)

    print("  returned:", len(df))

    frames.append(df)
    time.sleep(0.2)

labels = pd.concat(
    frames,
    ignore_index=True
)

labels["source_id"] = labels["source_id"].astype(str)

labels = (
    labels
    .drop_duplicates(
        subset=["source_id"]
    )
    .reset_index(drop=True)
)

# -----------------------------
# Binary labels
# -----------------------------

def to_bool01(series):
    s = (
        series
        .fillna(False)
    )
    return s.astype(bool).astype(int)

labels["label_qso_candidate"] = to_bool01(
    labels["in_qso_candidates"]
)

labels["label_galaxy_candidate"] = to_bool01(
    labels["in_galaxy_candidates"]
)

labels["label_extragalactic_candidate"] = (
    (
        labels["label_qso_candidate"] == 1
    )
    |
    (
        labels["label_galaxy_candidate"] == 1
    )
).astype(int)

labels["label_variable"] = (
    labels["phot_variable_flag"]
    .fillna("")
    .astype(str)
    .str.upper()
    .eq("VARIABLE")
    .astype(int)
)

labels["non_single_star"] = pd.to_numeric(
    labels["non_single_star"],
    errors="coerce"
).fillna(0)

labels["label_nss"] = (
    labels["non_single_star"] > 0
).astype(int)

# -----------------------------
# DSC numeric probabilities
# -----------------------------

prob_cols = [
    "classprob_dsc_combmod_quasar",
    "classprob_dsc_combmod_galaxy",
    "classprob_dsc_combmod_star",
]

for c in prob_cols:
    labels[c] = pd.to_numeric(
        labels[c],
        errors="coerce"
    )

# High-confidence DSC labels.
labels["label_dsc_quasar_050"] = (
    labels["classprob_dsc_combmod_quasar"] >= 0.50
).astype(int)

labels["label_dsc_galaxy_050"] = (
    labels["classprob_dsc_combmod_galaxy"] >= 0.50
).astype(int)

labels["label_dsc_extragalactic_050"] = (
    (
        labels["label_dsc_quasar_050"] == 1
    )
    |
    (
        labels["label_dsc_galaxy_050"] == 1
    )
).astype(int)

labels["label_dsc_nonstar_050"] = (
    labels["classprob_dsc_combmod_star"] < 0.50
).fillna(False).astype(int)

labels.to_csv(
    OUT_FILE,
    index=False
)

QUERY_LOG.write_text(
    ("\n\n" + "="*80 + "\n\n").join(queries),
    encoding="utf-8"
)

target_set = set(ids)
returned_set = set(labels["source_id"])

missing = target_set - returned_set
extra = returned_set - target_set

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

print()
print("="*90)
print("GAIA EXTENDED LABEL AUDIT")
print("="*90)

print("Targets:", len(target_set))
print("Returned:", len(returned_set))
print("Missing:", len(missing))
print("Extra:", len(extra))

label_cols = [
    "label_variable",
    "label_nss",
    "label_qso_candidate",
    "label_galaxy_candidate",
    "label_extragalactic_candidate",
    "label_dsc_quasar_050",
    "label_dsc_galaxy_050",
    "label_dsc_extragalactic_050",
    "label_dsc_nonstar_050",
]

print()
for c in label_cols:
    n = int(labels[c].sum())
    print(
        f"{c:35s}",
        f"{n:5d}",
        f"({100*n/len(labels):7.3f}%)"
    )

print()
print("DSC probability non-null counts:")

for c in prob_cols:
    print(
        f"{c:35s}",
        labels[c].notna().sum()
    )

print()
print("SHA256:", sha256(OUT_FILE))
print("Saved:", OUT_FILE)
print("DONE.")
