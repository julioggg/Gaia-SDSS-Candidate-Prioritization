from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd


RADIUS_DEG = 3.0


FIELDS = [
    ("baseline", 180.0,  30.0),
    ("field_A",    60.0,  30.0),
    ("field_B",   300.0,  30.0),
    ("field_C",   180.0, -30.0),
    ("field_D",    60.0, -30.0),
    ("field_E",   300.0, -30.0),
]


rows = []


print("=" * 110)
print("GAIA DR3 INDEPENDENT-REGION ELIGIBILITY AUDIT")
print("=" * 110)


for name, ra, dec in FIELDS:

    c = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        frame="icrs"
    )

    gal = c.galactic

    query = f"""
    SELECT COUNT(*) AS n
    FROM gaiadr3.gaia_source
    WHERE
        1 = CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {RADIUS_DEG})
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


    print()
    print("-" * 110)

    print(
        f"{name}: "
        f"RA={ra:.1f}, Dec={dec:.1f} | "
        f"l={gal.l.deg:.2f}, b={gal.b.deg:.2f}"
    )


    job = Gaia.launch_job_async(
        query,
        dump_to_file=False
    )

    result = job.get_results()

    n = int(
        result["n"][0]
    )


    print(
        "Eligible Gaia sources:",
        f"{n:,}"
    )


    rows.append({
        "field": name,
        "ra_deg": ra,
        "dec_deg": dec,
        "gal_l_deg": float(
            gal.l.deg
        ),
        "gal_b_deg": float(
            gal.b.deg
        ),
        "eligible_sources": n,
    })


df = pd.DataFrame(
    rows
)


print()
print("=" * 110)
print("SUMMARY")
print("=" * 110)

print(
    df.to_string(
        index=False
    )
)


out = (
    "outputs/independent_regions/"
    "gaia_candidate_region_counts.csv"
)

from pathlib import Path

Path(
    "outputs/independent_regions"
).mkdir(
    parents=True,
    exist_ok=True
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
