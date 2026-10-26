from pathlib import Path
import hashlib
import time
import pandas as pd
from astroquery.sdss import SDSS


ROOT = Path.home() / "universe_revision" / "revision_v2"

TARGET_FILE = (
    ROOT / "data/processed/sdss_features_reference.csv"
)

OUT = (
    ROOT / "data/external_validation/sdss"
)

CHUNKS = OUT / "chunks_labels"

OUT.mkdir(parents=True, exist_ok=True)
CHUNKS.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT / "sdss_external_labels_exact.csv"

targets = pd.read_csv(
    TARGET_FILE,
    dtype={
        "objid": str,
        "specobjid": str,
        "class": str,
    }
)

target_ids = set(
    targets["specobjid"].astype(str)
)

target_meta = (
    targets[
        [
            "objid",
            "specobjid",
            "class",
        ]
    ]
    .drop_duplicates("specobjid")
    .copy()
)

print("=" * 100)
print("SDSS DR17 EXACT EXTERNAL LABEL RETRIEVAL — RA CHUNKS")
print("=" * 100)

print("Targets:", f"{len(target_ids):,}")
print(
    "GALAXY:",
    f"{(targets['class']=='GALAXY').sum():,}"
)
print(
    "QSO:",
    f"{(targets['class']=='QSO').sum():,}"
)


# ==========================================================
# Same target sky region as definitive SDSS sample
# ==========================================================

RA_MIN = 120.0
RA_MAX = 240.0
STEP = 5.0

edges = []

x = RA_MIN

while x < RA_MAX:

    hi = min(
        x + STEP,
        RA_MAX
    )

    edges.append(
        (x, hi)
    )

    x = hi


def build_query(lo, hi, last=False):

    upper_op = "<=" if last else "<"

    return f"""
SELECT
    p.objid,
    p.ra,
    p.dec,
    s.specobjid,
    s.class,
    s.subClass,
    s.survey,
    s.programname,
    s.sourceType,
    s.z,
    s.zErr,
    s.zWarning,
    s.class_noqso,
    s.subClass_noqso,
    s.z_noqso,
    s.zWarning_noqso,
    s.class_person,
    s.z_person,
    s.snMedian,
    s.rChi2,
    s.rChi2Diff,
    s.velDisp,
    s.velDispErr
FROM PhotoPrimary AS p
JOIN SpecObj AS s
    ON p.objid = s.bestobjid
WHERE
    p.ra >= {lo:.6f}
    AND p.ra {upper_op} {hi:.6f}
    AND p.dec BETWEEN 0.0 AND 60.0
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


matched_frames = []


for i, (lo, hi) in enumerate(
    edges,
    start=1
):

    last = (
        i == len(edges)
    )

    chunk_file = (
        CHUNKS
        / f"labels_{i:03d}_{lo:.1f}_{hi:.1f}.csv"
    )

    query_file = (
        CHUNKS
        / f"labels_{i:03d}_{lo:.1f}_{hi:.1f}.sql"
    )

    query = build_query(
        lo,
        hi,
        last=last
    )

    query_file.write_text(
        query.strip() + "\n",
        encoding="utf-8"
    )

    # ------------------------------------------
    # Resume support
    # ------------------------------------------

    if chunk_file.exists():

        try:

            d = pd.read_csv(
                chunk_file,
                dtype={
                    "objid": str,
                    "specobjid": str,
                }
            )

            print(
                f"[{i:02d}/{len(edges)}] "
                f"EXISTS RA {lo:.1f}-{hi:.1f}: "
                f"{len(d):,} matched targets"
            )

            matched_frames.append(d)

            continue

        except Exception:

            print(
                "Invalid existing chunk; "
                "redownloading:",
                chunk_file.name
            )


    print(
        f"[{i:02d}/{len(edges)}] "
        f"Query RA {lo:.1f}-{hi:.1f}"
    )

    table = None
    last_error = None

    for attempt in range(
        1,
        4
    ):

        try:

            table = SDSS.query_sql(
                query,
                data_release=17,
                timeout=900
            )

            if table is None:
                raise RuntimeError(
                    "SDSS returned None."
                )

            break

        except Exception as e:

            last_error = e

            print(
                f"  attempt {attempt}/3 "
                f"failed: {e}"
            )

            if attempt < 3:
                time.sleep(
                    10 * attempt
                )


    if table is None:

        raise RuntimeError(
            f"Chunk {i} failed: "
            f"{last_error}"
        )


    df = table.to_pandas()

    df["objid"] = (
        df["objid"]
        .astype(str)
    )

    df["specobjid"] = (
        df["specobjid"]
        .astype(str)
    )


    # ======================================================
    # Critical step:
    # retain ONLY exact definitive target IDs
    # ======================================================

    matched = (
        df[
            df["specobjid"].isin(
                target_ids
            )
        ]
        .copy()
    )

    matched.to_csv(
        chunk_file,
        index=False
    )

    print(
        "  SQL rows:",
        f"{len(df):,}"
    )

    print(
        "  exact targets:",
        f"{len(matched):,}"
    )

    matched_frames.append(
        matched
    )

    # free memory
    del df
    del matched


# ==========================================================
# Combine only exact targets
# ==========================================================

labels = pd.concat(
    matched_frames,
    ignore_index=True
)

labels["specobjid"] = (
    labels["specobjid"]
    .astype(str)
)

labels = (
    labels
    .drop_duplicates(
        subset=["specobjid"],
        keep="first"
    )
    .reset_index(drop=True)
)


# ==========================================================
# Merge against frozen definitive target list
# ==========================================================

labels = target_meta.merge(
    labels,
    on="specobjid",
    how="left",
    suffixes=(
        "_target",
        "_spec"
    ),
    validate="one_to_one"
)


string_cols = [
    "class_target",
    "class_spec",
    "subClass",
    "survey",
    "programname",
    "sourceType",
    "class_noqso",
    "subClass_noqso",
    "class_person",
]

for c in string_cols:

    if c in labels.columns:

        labels[c] = (
            labels[c]
            .fillna("")
            .astype(str)
            .str.strip()
        )


labels.to_csv(
    OUT_FILE,
    index=False
)


# ==========================================================
# Audit
# ==========================================================

matched = (
    labels["class_spec"]
    .ne("")
)

print()
print("=" * 100)
print("SDSS EXACT LABEL AUDIT")
print("=" * 100)

print(
    "Targets:",
    f"{len(labels):,}"
)

print(
    "Matched:",
    f"{matched.sum():,}"
)

print(
    "Missing:",
    f"{(~matched).sum():,}"
)


if matched.any():

    agreement = (
        labels.loc[
            matched,
            "class_target"
        ]
        ==
        labels.loc[
            matched,
            "class_spec"
        ]
    )

    print(
        "Class agreement:",
        f"{100*agreement.mean():.4f}%"
    )


def print_distribution(
    df,
    col,
    n=40
):

    s = (
        df[col]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace(
            "",
            "<EMPTY>"
        )
        .value_counts(
            dropna=False
        )
    )

    print(
        s.head(n).to_string()
    )


for cls in [
    "GALAXY",
    "QSO",
]:

    d = labels[
        labels[
            "class_target"
        ] == cls
    ].copy()

    print()
    print("=" * 100)
    print(
        f"{cls} SUBCLASS DISTRIBUTION"
    )
    print("=" * 100)

    print(
        "N:",
        f"{len(d):,}"
    )

    nonempty = (
        d["subClass"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    print(
        "Non-empty subclass:",
        f"{nonempty.sum():,}",
        f"({100*nonempty.mean():.3f}%)"
    )

    print()

    print_distribution(
        d,
        "subClass"
    )


for col in [
    "sourceType",
    "survey",
    "programname",
    "class_noqso",
    "subClass_noqso",
    "class_person",
]:

    print()
    print("=" * 100)
    print(
        f"{col} DISTRIBUTION"
    )
    print("=" * 100)

    print_distribution(
        labels,
        col,
        n=30
    )


print()
print("=" * 100)
print("NUMERIC FIELD COVERAGE")
print("=" * 100)

for c in [
    "z",
    "zErr",
    "snMedian",
    "rChi2",
    "rChi2Diff",
    "velDisp",
    "velDispErr",
]:

    x = pd.to_numeric(
        labels[c],
        errors="coerce"
    )

    print(
        f"{c:15s}",
        f"{x.notna().sum():,}"
    )


def sha256(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        for block in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b""
        ):

            h.update(block)

    return h.hexdigest()


print()
print(
    "SHA256:",
    sha256(OUT_FILE)
)

print(
    "Saved:",
    OUT_FILE
)

print("DONE.")
