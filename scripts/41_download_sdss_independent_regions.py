from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from astroquery.sdss import SDSS


ROOT = Path.home() / "universe_revision" / "revision_v2"

OUT = (
    ROOT
    / "data/independent_regions/sdss"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


REGIONS = [
    {
        "name": "region_A",
        "ra_min": 120.0,
        "ra_max": 150.0,
        "dec_min": 0.0,
        "dec_max": 30.0,
    },
    {
        "name": "region_G",
        "ra_min": 200.0,
        "ra_max": 240.0,
        "dec_min": 30.0,
        "dec_max": 60.0,
    },
]


N_FROZEN = 50000

# Reference buffer:
# 600 arcsec = 0.1667 deg.
# Use 0.25 deg conservatively.
BUFFER_DEG = 0.25

RA_TILE_DEG = 5.0
DEC_TILE_DEG = 5.0


SELECT_COLUMNS = """
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
"""


FILTERS = """
    AND p.ra IS NOT NULL
    AND p.dec IS NOT NULL
    AND p.u BETWEEN 10 AND 30
    AND p.g BETWEEN 10 AND 30
    AND p.r BETWEEN 10 AND 30
    AND p.i BETWEEN 10 AND 30
    AND p.z BETWEEN 10 AND 30
    AND p.clean = 1
    AND s.z IS NOT NULL
    AND s.zWarning = 0
    AND s.z > 0.01
    AND s.z < 5.0
    AND s.class IN ('GALAXY', 'QSO')
"""


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):

            h.update(chunk)

    return h.hexdigest()


def query_with_retry(
    sql,
    attempts=5,
):

    last = None

    for attempt in range(
        1,
        attempts + 1
    ):

        try:

            t = SDSS.query_sql(
                sql,
                data_release=17,
                timeout=600
            )

            if t is None:
                raise RuntimeError(
                    "SDSS returned None."
                )

            return t.to_pandas()

        except Exception as e:

            last = e

            print(
                f"  attempt {attempt}/{attempts} failed:",
                repr(e)
            )

            if attempt < attempts:
                time.sleep(
                    5 * attempt
                )

    raise last


def count_region(
    ra_min,
    ra_max,
    dec_min,
    dec_max,
):

    sql = f"""
    SELECT COUNT(*) AS n
    FROM PhotoPrimary AS p
    JOIN SpecObj AS s
        ON p.objid = s.bestobjid
    WHERE
        p.ra >= {ra_min}
        AND p.ra < {ra_max}
        AND p.dec >= {dec_min}
        AND p.dec < {dec_max}
    {FILTERS}
    """

    df = query_with_retry(
        sql
    )

    return int(
        df.iloc[0]["n"]
    )


def make_tiles(
    ra_min,
    ra_max,
    dec_min,
    dec_max,
):

    ra_edges = list(
        np.arange(
            ra_min,
            ra_max,
            RA_TILE_DEG
        )
    )

    if ra_edges[-1] < ra_max:
        ra_edges.append(
            ra_max
        )


    dec_edges = list(
        np.arange(
            dec_min,
            dec_max,
            DEC_TILE_DEG
        )
    )

    if dec_edges[-1] < dec_max:
        dec_edges.append(
            dec_max
        )


    tiles = []

    for i in range(
        len(ra_edges) - 1
    ):

        for j in range(
            len(dec_edges) - 1
        ):

            tiles.append({
                "i": i,
                "j": j,
                "ra_lo":
                    float(
                        ra_edges[i]
                    ),
                "ra_hi":
                    float(
                        ra_edges[i+1]
                    ),
                "dec_lo":
                    float(
                        dec_edges[j]
                    ),
                "dec_hi":
                    float(
                        dec_edges[j+1]
                    ),
            })

    return tiles


def download_box_tiled(
    region_name,
    ra_min,
    ra_max,
    dec_min,
    dec_max,
    query_dir,
    prefix,
):

    expected = count_region(
        ra_min,
        ra_max,
        dec_min,
        dec_max,
    )


    print(
        "Expected count:",
        f"{expected:,}"
    )


    tiles = make_tiles(
        ra_min,
        ra_max,
        dec_min,
        dec_max,
    )


    print(
        "Tiles:",
        len(tiles)
    )


    frames = []
    manifest = []


    for t in tiles:

        sql = f"""
        SELECT
        {SELECT_COLUMNS}
        FROM PhotoPrimary AS p
        JOIN SpecObj AS s
            ON p.objid = s.bestobjid
        WHERE
            p.ra >= {t['ra_lo']}
            AND p.ra < {t['ra_hi']}
            AND p.dec >= {t['dec_lo']}
            AND p.dec < {t['dec_hi']}
        {FILTERS}
        """


        query_file = (
            query_dir
            / (
                f"{prefix}_"
                f"ra{t['i']:02d}_"
                f"dec{t['j']:02d}.sql"
            )
        )


        query_file.write_text(
            sql.strip() + "\n",
            encoding="utf-8"
        )


        print(
            f"  tile "
            f"RA[{t['ra_lo']},{t['ra_hi']}) "
            f"Dec[{t['dec_lo']},{t['dec_hi']})"
        )


        part = query_with_retry(
            sql
        )


        print(
            "    rows:",
            f"{len(part):,}"
        )


        if len(part) >= 50000:

            raise RuntimeError(
                "A tile returned >=50,000 rows; "
                "reduce tile size."
            )


        frames.append(
            part
        )


        manifest.append({
            **t,
            "rows":
                int(
                    len(part)
                ),
            "query_file":
                str(
                    query_file
                ),
        })


    df = pd.concat(
        frames,
        ignore_index=True
    )


    # Preserve IDs safely.
    for c in [
        "objid",
        "specobjid",
    ]:

        df[c] = (
            df[c]
            .astype(str)
        )


    before = len(
        df
    )


    df = (
        df
        .drop_duplicates(
            subset=[
                "objid"
            ],
            keep="first"
        )
        .reset_index(drop=True)
    )


    print(
        "Rows before dedup:",
        f"{before:,}"
    )

    print(
        "Unique objid:",
        f"{len(df):,}"
    )


    if len(df) != expected:

        raise RuntimeError(
            f"{region_name}/{prefix}: "
            f"assembled {len(df):,}, "
            f"COUNT(*) says {expected:,}"
        )


    manifest_file = (
        query_dir
        / f"{prefix}_manifest.json"
    )


    manifest_file.write_text(
        json.dumps(
            {
                "expected":
                    expected,
                "assembled":
                    int(
                        len(df)
                    ),
                "ra_tile_deg":
                    RA_TILE_DEG,
                "dec_tile_deg":
                    DEC_TILE_DEG,
                "tiles":
                    manifest,
            },
            indent=2
        )
        +
        "\n",
        encoding="utf-8"
    )


    return (
        df,
        expected,
        manifest_file
    )


summary_rows = []


print("=" * 120)
print("SDSS INDEPENDENT-REGION TILED DOWNLOAD")
print("=" * 120)


for region in REGIONS:

    name = region["name"]

    print()
    print("=" * 120)
    print(name)
    print("=" * 120)


    region_dir = (
        OUT
        / name
    )

    region_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    query_dir = (
        region_dir
        / "queries"
    )

    query_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ======================================================
    # Core
    # ======================================================

    print()
    print("CORE")
    print("-" * 120)


    core, core_expected, core_manifest = (
        download_box_tiled(
            region_name=name,
            ra_min=region["ra_min"],
            ra_max=region["ra_max"],
            dec_min=region["dec_min"],
            dec_max=region["dec_max"],
            query_dir=query_dir,
            prefix="core",
        )
    )


    # Deterministic pseudo-random sample.
    #
    # Same principle as definitive SDSS sample:
    # hash-like key derived from HTM identifier.
    core["sample_key"] = (
        (
            pd.to_numeric(
                core["htmid"],
                errors="coerce"
            ).astype("int64")
            *
            37
        )
        &
        0xFFFF
    )


    core = (
        core
        .sort_values(
            [
                "sample_key",
                "objid",
            ],
            kind="mergesort"
        )
        .reset_index(drop=True)
    )


    if len(core) < N_FROZEN:

        raise RuntimeError(
            f"{name}: fewer than "
            f"{N_FROZEN:,} eligible objects."
        )


    frozen = (
        core
        .iloc[:N_FROZEN]
        .copy()
    )


    print(
        "Frozen sample:",
        f"{len(frozen):,}"
    )


    # ======================================================
    # Buffered reference
    # ======================================================

    print()
    print("BUFFERED REFERENCE")
    print("-" * 120)


    ref_ra_min = max(
        0.0,
        region["ra_min"]
        -
        BUFFER_DEG
    )

    ref_ra_max = min(
        360.0,
        region["ra_max"]
        +
        BUFFER_DEG
    )

    ref_dec_min = max(
        -90.0,
        region["dec_min"]
        -
        BUFFER_DEG
    )

    ref_dec_max = min(
        90.0,
        region["dec_max"]
        +
        BUFFER_DEG
    )


    reference, ref_expected, ref_manifest = (
        download_box_tiled(
            region_name=name,
            ra_min=ref_ra_min,
            ra_max=ref_ra_max,
            dec_min=ref_dec_min,
            dec_max=ref_dec_max,
            query_dir=query_dir,
            prefix="reference",
        )
    )


    target_ids = set(
        frozen["objid"]
    )

    ref_ids = set(
        reference["objid"]
    )


    missing = (
        target_ids
        -
        ref_ids
    )


    print(
        "Frozen IDs present in reference:",
        f"{len(target_ids)-len(missing):,}",
        "/",
        f"{len(target_ids):,}"
    )


    if missing:

        raise RuntimeError(
            f"{name}: "
            f"{len(missing)} frozen targets "
            "missing from reference."
        )


    # ======================================================
    # Save
    # ======================================================

    core_file = (
        region_dir
        / "eligible_population.csv"
    )

    frozen_file = (
        region_dir
        / "raw_frozen.csv"
    )

    reference_file = (
        region_dir
        / "reference_buffered.csv"
    )

    ids_file = (
        region_dir
        / "objids.txt"
    )


    core.to_csv(
        core_file,
        index=False
    )

    frozen.to_csv(
        frozen_file,
        index=False
    )

    reference.to_csv(
        reference_file,
        index=False
    )

    frozen[
        "objid"
    ].to_csv(
        ids_file,
        index=False,
        header=False
    )


    hashes = {
        "eligible_population_sha256":
            sha256(core_file),

        "raw_frozen_sha256":
            sha256(frozen_file),

        "reference_buffered_sha256":
            sha256(reference_file),

        "objids_sha256":
            sha256(ids_file),

        "core_manifest_sha256":
            sha256(core_manifest),

        "reference_manifest_sha256":
            sha256(ref_manifest),
    }


    metadata = {
        **region,
        "buffer_deg":
            BUFFER_DEG,
        "core_count":
            int(
                core_expected
            ),
        "frozen_n":
            int(
                len(frozen)
            ),
        "reference_count":
            int(
                ref_expected
            ),
        "hashes":
            hashes,
    }


    (
        region_dir
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2
        )
        +
        "\n",
        encoding="utf-8"
    )


    summary_rows.append({
        "region":
            name,

        "core_count":
            core_expected,

        "frozen_n":
            len(frozen),

        "reference_count":
            ref_expected,

        **hashes,
    })


summary = pd.DataFrame(
    summary_rows
)


summary_file = (
    OUT
    / "download_summary.csv"
)


summary.to_csv(
    summary_file,
    index=False
)


print()
print("=" * 140)
print("SDSS INDEPENDENT REGION DOWNLOAD SUMMARY")
print("=" * 140)

print(
    summary[
        [
            "region",
            "core_count",
            "frozen_n",
            "reference_count",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "Saved:",
    summary_file
)

print("DONE.")
