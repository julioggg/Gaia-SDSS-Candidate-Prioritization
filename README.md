# Reproducible Multivariable Candidate Prioritization in Gaia DR3 and SDSS DR17

This repository provides the analysis code, exact database queries, frozen
configurations, derived numerical results, figures, tables, and
reproducibility metadata associated with the manuscript:

**Regularized Multivariable Ranking for Reproducible Candidate Prioritization
in Gaia DR3 and SDSS DR17**

## Scientific scope

The proposed framework is an interpretable and reproducible multivariable
candidate-prioritization method for large astronomical catalogs.

It is not presented as a universal anomaly detector.

External catalog labels are used only for post hoc validation and are not
used during optimization of the ranking weights.

## Data sources

The analyses use:

- Gaia Data Release 3 (Gaia DR3)
- Sloan Digital Sky Survey Data Release 17 (SDSS DR17)

Large reconstructed reference catalogs are not stored directly in this
GitHub repository. Exact ADQL/SQL queries, configurations, identifiers,
metadata, and SHA256 hashes are provided to support reconstruction.

## Repository structure

- `scripts/` — analysis and peer-review revision scripts
- `configs/` — frozen analysis configurations
- `queries/` — exact Gaia ADQL and SDSS SQL queries
- `results/` — definitive numerical outputs
- `figures/` — manuscript and diagnostic figures
- `tables/` — manuscript and supplementary tables
- `metadata/` — IDs, manifests, hashes, and environment information
- `docs/` — supporting documentation

## Analyses included

The repository contains:

- deterministic catalog reconstruction;
- buffered reference populations;
- multi-seed optimizer stability;
- top-K sensitivity;
- clipping sensitivity;
- effective-feature regularization sensitivity;
- neighborhood-scale sensitivity;
- Gaia quality-cut sensitivity;
- SDSS quality-cut sensitivity;
- SDSS GALAXY/QSO class-specific analyses;
- Isolation Forest comparison;
- Local Outlier Factor comparison;
- PCA-Mahalanobis comparison;
- Gaia catalog-based external validation;
- Gaia DSC non-stellar validation;
- SDSS spectroscopic validation;
- independent Gaia sky-region validation;
- independent SDSS region validation.

## Interpretation

The proposed method is interpreted as a transparent multivariable
candidate-prioritization framework rather than as a universal anomaly
detector.

External validation is performed independently of the optimization objective.

## Reproducibility

Exact package versions are provided in `requirements.txt`.

SHA256 checksums and reproducibility metadata are provided under
`metadata/`.

## Large data

Large reconstructed catalog reference populations are intentionally excluded
from GitHub. They are intended for separate DOI-bearing archival deposition.

## License

MIT License.

## Citation

Please cite the associated article and the archived software/data DOI once
available.
