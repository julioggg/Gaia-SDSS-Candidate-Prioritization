from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from astroquery.gaia import Gaia


ROOT = Path.home() / "universe_revision" / "revision_v2"

OUT = (
    ROOT
    / "data/independent_regions/gaia"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


CORE_RADIUS_DEG = 3.0
BUFFER_DEG = 0.10
REFERENCE_RADIUS_DEG = (
    CORE_RADIUS_DEG
    +
    BUFFER_DEG
)

N_FROZEN = 10000

# Small enough that every query remains well below
# the 50,000-row archive truncation observed previously.
DEC_TILE_WIDTH_DEG = 0.50


FIELDS = [
    {
        "name": "field_C",
        "ra": 180.0,
        "dec": -30.0,
    },
    {
        "name": "field_D",
        "ra": 60.0,
        "dec": -30.0,
    },
]


SELECT_COLUMNS = """
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
"""


BASE_FILTER = """
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


def sha256(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b""
        ):

            h.update(chunk)

    return h.hexdigest()


def launch_with_retry(
    query,
    attempts=5,
):

    last_error = None

    for attempt in range(
        1,
        attempts + 1
    ):

        try:

            job = Gaia.launch_job_async(
                query,
                dump_to_file=False
            )

            return (
                job
                .get_results()
                .to_pandas()
            )

        except Exception as e:

            last_error = e

            print(
                f"  TAP attempt "
                f"{attempt}/{attempts} failed:"
            )

            print(
                " ",
                repr(e)
            )

            if attempt < attempts:
                time.sleep(
                    5 * attempt
                )

    raise last_error


def make_count_query(
    ra,
    dec,
    radius,
):

    return f"""
SELECT COUNT(*) AS n
FROM gaiadr3.gaia_source
WHERE
    1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE(
            'ICRS',
            {ra},
            {dec},
            {radius}
        )
    )
{BASE_FILTER}
""".strip()


def expected_count(
    ra,
    dec,
    radius,
):

    query = make_count_query(
        ra,
        dec,
        radius
    )

    df = launch_with_retry(
        query
    )

    return int(
        df.iloc[0]["n"]
    )


def make_tile_query(
    ra,
    dec,
    radius,
    dec_lo,
    dec_hi,
    is_last,
):

    if is_last:

        dec_clause = (
            f"AND dec >= {dec_lo} "
            f"AND dec <= {dec_hi}"
        )

    else:

        dec_clause = (
            f"AND dec >= {dec_lo} "
            f"AND dec < {dec_hi}"
        )


    return f"""
SELECT
{SELECT_COLUMNS}
FROM gaiadr3.gaia_source
WHERE
    1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE(
            'ICRS',
            {ra},
            {dec},
            {radius}
        )
    )
    {dec_clause}
{BASE_FILTER}
""".strip()


def download_circle_tiled(
    name,
    ra,
    dec,
    radius,
    query_dir,
    prefix,
):

    expected = expected_count(
        ra,
        dec,
        radius
    )

    print(
        "Expected from COUNT(*):",
        f"{expected:,}"
    )


    dec_min = (
        dec
        -
        radius
    )

    dec_max = (
        dec
        +
        radius
    )


    edges = list(
        np.arange(
            dec_min,
            dec_max,
            DEC_TILE_WIDTH_DEG
        )
    )

    if (
        not edges
        or
        edges[0] != dec_min
    ):
        edges = [
            dec_min
        ] + edges

    if edges[-1] < dec_max:
        edges.append(
            dec_max
        )


    frames = []

    tile_manifest = []


    print(
        "Number of Dec tiles:",
        len(edges) - 1
    )


    for i in range(
        len(edges) - 1
    ):

        lo = float(
            edges[i]
        )

        hi = float(
            edges[i + 1]
        )

        is_last = (
            i
            ==
            len(edges) - 2
        )


        query = make_tile_query(
            ra,
            dec,
            radius,
            lo,
            hi,
            is_last,
        )


        query_file = (
            query_dir
            /
            f"{prefix}_tile_{i:02d}.adql"
        )

        query_file.write_text(
            query + "\n",
            encoding="utf-8"
        )


        print(
            f"  tile {i:02d}: "
            f"Dec [{lo:.3f}, {hi:.3f}"
            +
            (
                "]"
                if is_last
                else ")"
            )
        )


        part = launch_with_retry(
            query
        )


        print(
            "    rows:",
            f"{len(part):,}"
        )


        if len(part) >= 50000:

            raise RuntimeError(
                f"{name}/{prefix}/tile{i}: "
                "tile reached 50,000 rows; "
                "reduce DEC_TILE_WIDTH_DEG."
            )


        frames.append(
            part
        )


        tile_manifest.append({
            "tile": i,
            "dec_lo": lo,
            "dec_hi": hi,
            "last_tile": is_last,
            "rows_returned":
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


    df[
        "source_id"
    ] = (
        df[
            "source_id"
        ]
        .astype(str)
    )


    before_dedup = len(
        df
    )


    df = (
        df
        .drop_duplicates(
            subset=[
                "source_id"
            ],
            keep="first"
        )
        .reset_index(
            drop=True
        )
    )


    print(
        "Rows before dedup:",
        f"{before_dedup:,}"
    )

    print(
        "Unique rows after dedup:",
        f"{len(df):,}"
    )


    if len(df) != expected:

        raise RuntimeError(
            f"{name}/{prefix}: "
            f"assembled {len(df):,} "
            f"but COUNT(*) reports "
            f"{expected:,}."
        )


    manifest_file = (
        query_dir
        /
        f"{prefix}_tile_manifest.json"
    )

    manifest_file.write_text(
        json.dumps(
            {
                "expected_count":
                    expected,
                "assembled_count":
                    int(
                        len(df)
                    ),
                "tile_width_dec_deg":
                    DEC_TILE_WIDTH_DEG,
                "tiles":
                    tile_manifest,
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


# ==========================================================
# Main
# ==========================================================

summary_rows = []


print("=" * 110)
print(
    "GAIA INDEPENDENT-REGION "
    "TILED DOWNLOAD"
)
print("=" * 110)


for field in FIELDS:

    name = field[
        "name"
    ]

    ra = field[
        "ra"
    ]

    dec = field[
        "dec"
    ]


    field_dir = (
        OUT
        /
        name
    )

    field_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    query_dir = (
        field_dir
        /
        "queries"
    )

    query_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    print()
    print("=" * 110)
    print(
        name,
        f"| RA={ra}",
        f"| Dec={dec}"
    )
    print("=" * 110)


    # ======================================================
    # Complete core population
    # ======================================================

    print()
    print(
        "CORE POPULATION"
    )
    print("-" * 110)


    core, core_expected, core_manifest = (
        download_circle_tiled(
            name=name,
            ra=ra,
            dec=dec,
            radius=CORE_RADIUS_DEG,
            query_dir=query_dir,
            prefix="core",
        )
    )


    # Deterministic ordering exactly as baseline.
    core = (
        core
        .sort_values(
            [
                "random_index",
                "source_id",
            ],
            kind="mergesort"
        )
        .reset_index(
            drop=True
        )
    )


    if len(core) < N_FROZEN:

        raise RuntimeError(
            f"{name}: fewer than "
            f"{N_FROZEN:,} eligible sources."
        )


    frozen = (
        core
        .iloc[
            :N_FROZEN
        ]
        .copy()
    )


    print(
        "Frozen deterministic sample:",
        f"{len(frozen):,}"
    )


    core_file = (
        field_dir
        /
        "eligible_population.csv"
    )

    frozen_file = (
        field_dir
        /
        "raw_frozen.csv"
    )

    ids_file = (
        field_dir
        /
        "source_ids.txt"
    )


    core.to_csv(
        core_file,
        index=False
    )

    frozen.to_csv(
        frozen_file,
        index=False
    )

    frozen[
        "source_id"
    ].to_csv(
        ids_file,
        index=False,
        header=False
    )


    # ======================================================
    # Complete buffered reference
    # ======================================================

    print()
    print(
        "BUFFERED REFERENCE POPULATION"
    )
    print("-" * 110)


    reference, ref_expected, ref_manifest = (
        download_circle_tiled(
            name=name,
            ra=ra,
            dec=dec,
            radius=REFERENCE_RADIUS_DEG,
            query_dir=query_dir,
            prefix="reference",
        )
    )


    # Stable physical ordering for reproducible file.
    reference = (
        reference
        .sort_values(
            [
                "source_id"
            ],
            kind="mergesort"
        )
        .reset_index(
            drop=True
        )
    )


    reference_file = (
        field_dir
        /
        "reference_buffered.csv"
    )


    reference.to_csv(
        reference_file,
        index=False
    )


    # ======================================================
    # Integrity
    # ======================================================

    target_ids = set(
        frozen[
            "source_id"
        ]
    )

    reference_ids = set(
        reference[
            "source_id"
        ]
    )


    missing = (
        target_ids
        -
        reference_ids
    )


    print()
    print(
        "Frozen IDs present in reference:",
        f"{len(target_ids)-len(missing):,}",
        "/",
        f"{len(target_ids):,}"
    )


    if missing:

        raise RuntimeError(
            f"{name}: "
            f"{len(missing)} frozen IDs "
            "missing from complete reference."
        )


    # ======================================================
    # Hashes
    # ======================================================

    hashes = {
        "eligible_population_sha256":
            sha256(
                core_file
            ),

        "raw_frozen_sha256":
            sha256(
                frozen_file
            ),

        "source_ids_sha256":
            sha256(
                ids_file
            ),

        "reference_buffered_sha256":
            sha256(
                reference_file
            ),

        "core_tile_manifest_sha256":
            sha256(
                core_manifest
            ),

        "reference_tile_manifest_sha256":
            sha256(
                ref_manifest
            ),
    }


    metadata = {
        "field":
            name,

        "center_ra_deg":
            ra,

        "center_dec_deg":
            dec,

        "core_radius_deg":
            CORE_RADIUS_DEG,

        "buffer_deg":
            BUFFER_DEG,

        "reference_radius_deg":
            REFERENCE_RADIUS_DEG,

        "tile_width_dec_deg":
            DEC_TILE_WIDTH_DEG,

        "core_count_tap":
            int(
                core_expected
            ),

        "core_count_assembled":
            int(
                len(core)
            ),

        "frozen_n":
            int(
                len(frozen)
            ),

        "reference_count_tap":
            int(
                ref_expected
            ),

        "reference_count_assembled":
            int(
                len(reference)
            ),

        "random_index_min":
            int(
                frozen[
                    "random_index"
                ].min()
            ),

        "random_index_max":
            int(
                frozen[
                    "random_index"
                ].max()
            ),

        "hashes":
            hashes,
    }


    metadata_file = (
        field_dir
        /
        "metadata.json"
    )


    metadata_file.write_text(
        json.dumps(
            metadata,
            indent=2
        )
        +
        "\n",
        encoding="utf-8"
    )


    print()
    print("SHA256:")

    for k, v in hashes.items():

        print(
            f"  {k}: {v}"
        )


    summary_rows.append({
        "field":
            name,

        "ra_deg":
            ra,

        "dec_deg":
            dec,

        "core_count_tap":
            core_expected,

        "core_count_assembled":
            len(core),

        "frozen_n":
            len(frozen),

        "reference_count_tap":
            ref_expected,

        "reference_count_assembled":
            len(reference),

        **hashes,
    })


summary = pd.DataFrame(
    summary_rows
)


summary_file = (
    OUT
    /
    "download_summary.csv"
)


summary.to_csv(
    summary_file,
    index=False
)


print()
print("=" * 150)
print(
    "INDEPENDENT REGION DOWNLOAD SUMMARY"
)
print("=" * 150)


print(
    summary[
        [
            "field",
            "ra_deg",
            "dec_deg",
            "core_count_tap",
            "core_count_assembled",
            "frozen_n",
            "reference_count_tap",
            "reference_count_assembled",
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
