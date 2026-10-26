import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


ARCSEC_PER_RAD = (180.0 / np.pi) * 3600.0


def _coords_rad(df):
    """
    BallTree(haversine) expects [latitude, longitude] in radians.
    """
    dec = np.deg2rad(
        pd.to_numeric(df["dec"], errors="coerce").to_numpy(dtype=float)
    )
    ra = np.deg2rad(
        pd.to_numeric(df["ra"], errors="coerce").to_numpy(dtype=float)
    )
    return np.column_stack([dec, ra])


def add_reference_neighbor_features(
    targets,
    reference,
    radius_arcsec=300.0,
    kth_neighbor=5,
    redshift_radius_arcsec=600.0,
    dz=0.05,
    batch_size=2000,
):
    """
    Compute neighborhood quantities for TARGET objects against a larger
    REFERENCE population.

    Self-matches are excluded by objid, not merely by array position.

    Outputs:
        nn_sep_arcsec
        kth_sep_arcsec
        local_density_knn
        neighbor_count_radius
        redshift_neighbor_count
    """

    targets = targets.copy()
    reference = reference.copy()

    # Identifiers are identifiers, never numeric predictors.
    targets["objid"] = targets["objid"].astype(str)
    reference["objid"] = reference["objid"].astype(str)

    ref_ids = reference["objid"].to_numpy(dtype=str)
    target_ids = targets["objid"].to_numpy(dtype=str)

    target_coords = _coords_rad(targets)
    ref_coords = _coords_rad(reference)

    if not np.isfinite(target_coords).all():
        raise ValueError("Non-finite target coordinates detected.")

    if not np.isfinite(ref_coords).all():
        raise ValueError("Non-finite reference coordinates detected.")

    print("Building BallTree...")
    print(f"  targets:   {len(targets):,}")
    print(f"  reference: {len(reference):,}")

    tree = BallTree(
        ref_coords,
        metric="haversine",
        leaf_size=40
    )

    n = len(targets)

    nn_sep = np.full(n, np.nan, dtype=float)
    kth_sep = np.full(n, np.nan, dtype=float)
    count_radius = np.zeros(n, dtype=np.int32)
    count_redshift = np.zeros(n, dtype=np.int32)

    target_z = pd.to_numeric(
        targets["redshift"],
        errors="coerce"
    ).to_numpy(dtype=float)

    ref_z = pd.to_numeric(
        reference["redshift"],
        errors="coerce"
    ).to_numpy(dtype=float)

    radius_rad = float(radius_arcsec) / ARCSEC_PER_RAD
    redshift_radius_rad = (
        float(redshift_radius_arcsec) / ARCSEC_PER_RAD
    )

    kth_neighbor = int(kth_neighbor)

    # Query one extra position because normally the target itself
    # is also present in the reference catalog.
    query_k = kth_neighbor + 1

    print()
    print("Nearest-neighbor / kNN pass...")

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)

        dist, ind = tree.query(
            target_coords[start:stop],
            k=query_k,
            return_distance=True,
            sort_results=True
        )

        for j in range(stop - start):
            global_i = start + j

            candidate_idx = ind[j]
            candidate_dist = dist[j] * ARCSEC_PER_RAD

            # Remove self by stable astronomical identifier.
            keep = (
                ref_ids[candidate_idx] != target_ids[global_i]
            )

            candidate_dist = candidate_dist[keep]

            if len(candidate_dist) < kth_neighbor:
                raise RuntimeError(
                    f"Insufficient non-self neighbors for target "
                    f"{target_ids[global_i]}"
                )

            nn_sep[global_i] = candidate_dist[0]
            kth_sep[global_i] = candidate_dist[kth_neighbor - 1]

        print(
            f"\r  processed {stop:,}/{n:,}",
            end="",
            flush=True
        )

    print()

    print()
    print("Fixed-radius + redshift-neighborhood pass...")

    # One 600" search is sufficient for both:
    #   - angular neighbor count within 300"
    #   - redshift-local neighbor count within 600"
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)

        inds, dists = tree.query_radius(
            target_coords[start:stop],
            r=redshift_radius_rad,
            return_distance=True,
            sort_results=False
        )

        for j in range(stop - start):
            global_i = start + j

            idx = inds[j]
            sep_arcsec = dists[j] * ARCSEC_PER_RAD

            nonself = (
                ref_ids[idx] != target_ids[global_i]
            )

            idx_ns = idx[nonself]
            sep_ns = sep_arcsec[nonself]

            # Angular density count.
            count_radius[global_i] = int(
                np.sum(sep_ns <= float(radius_arcsec))
            )

            # Redshift-local count.
            z0 = target_z[global_i]

            if np.isfinite(z0):
                zr = ref_z[idx_ns]

                valid = np.isfinite(zr)

                count_redshift[global_i] = int(
                    np.sum(
                        valid
                        & (sep_ns <= float(redshift_radius_arcsec))
                        & (np.abs(zr - z0) <= float(dz))
                    )
                )

        print(
            f"\r  processed {stop:,}/{n:,}",
            end="",
            flush=True
        )

    print()

    targets["nn_sep_arcsec"] = nn_sep
    targets["kth_sep_arcsec"] = kth_sep

    kth_sep_arcmin = np.maximum(
        kth_sep / 60.0,
        1e-9
    )

    area_arcmin2 = np.pi * kth_sep_arcmin**2

    targets["local_density_knn"] = (
        float(kth_neighbor) / area_arcmin2
    )

    targets["neighbor_count_radius"] = count_radius
    targets["redshift_neighbor_count"] = count_redshift

    return targets
