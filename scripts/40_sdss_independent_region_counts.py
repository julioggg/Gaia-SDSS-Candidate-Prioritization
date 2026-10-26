from pathlib import Path
import pandas as pd
from astroquery.sdss import SDSS


ROOT = Path.home() / "universe_revision" / "revision_v2"

OUT = (
    ROOT
    / "outputs/independent_regions/sdss"
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
        "name": "region_B",
        "ra_min": 150.0,
        "ra_max": 180.0,
        "dec_min": 0.0,
        "dec_max": 30.0,
    },
    {
        "name": "region_C",
        "ra_min": 180.0,
        "ra_max": 210.0,
        "dec_min": 0.0,
        "dec_max": 30.0,
    },
    {
        "name": "region_D",
        "ra_min": 210.0,
        "ra_max": 240.0,
        "dec_min": 0.0,
        "dec_max": 30.0,
    },
    {
        "name": "region_E",
        "ra_min": 120.0,
        "ra_max": 160.0,
        "dec_min": 30.0,
        "dec_max": 60.0,
    },
    {
        "name": "region_F",
        "ra_min": 160.0,
        "ra_max": 200.0,
        "dec_min": 30.0,
        "dec_max": 60.0,
    },
    {
        "name": "region_G",
        "ra_min": 200.0,
        "ra_max": 240.0,
        "dec_min": 30.0,
        "dec_max": 60.0,
    },
]


rows = []


print("=" * 120)
print("SDSS DR17 INDEPENDENT-REGION ELIGIBILITY AUDIT")
print("=" * 120)


for r in REGIONS:

    query = f"""
    SELECT
        COUNT(*) AS n
    FROM PhotoPrimary AS p
    JOIN SpecObj AS s
        ON p.objid = s.bestobjid
    WHERE
        p.ra >= {r['ra_min']}
        AND p.ra < {r['ra_max']}
        AND p.dec >= {r['dec_min']}
        AND p.dec < {r['dec_max']}
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


    print()
    print("-" * 120)

    print(
        r["name"],
        f"RA=[{r['ra_min']},{r['ra_max']})",
        f"Dec=[{r['dec_min']},{r['dec_max']})"
    )


    table = SDSS.query_sql(
        query,
        data_release=17,
        timeout=600
    )


    if table is None:
        raise RuntimeError(
            f"No result for {r['name']}"
        )


    n = int(
        table["n"][0]
    )


    print(
        "Eligible objects:",
        f"{n:,}"
    )


    rows.append({
        **r,
        "eligible_objects": n,
    })


df = pd.DataFrame(
    rows
)


print()
print("=" * 120)
print("SUMMARY")
print("=" * 120)

print(
    df.to_string(
        index=False
    )
)


out = (
    OUT
    / "candidate_region_counts.csv"
)

df.to_csv(
    out,
    index=False
)


print()
print(
    "Saved:",
    out
)

print("DONE.")
