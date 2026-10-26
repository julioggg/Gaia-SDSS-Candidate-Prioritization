#!/usr/bin/env bash

set -euo pipefail

ROOT="$HOME/universe_revision/revision_v2"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"

FINAL="$ROOT/final_submission_package"
STAGE="$FINAL/staging"
ZIPDIR="$FINAL/zips"

FIG_STAGE="$STAGE/01_manuscript_figures"
TAB_STAGE="$STAGE/02_manuscript_tables"
DATA_STAGE="$STAGE/03_new_revision_data"
CODE_STAGE="$STAGE/04_supplementary_code"
META_STAGE="$STAGE/05_reproducibility_metadata"

echo "============================================================"
echo "FINAL SUBMISSION PACKAGE BUILDER"
echo "============================================================"
echo "ROOT: $ROOT"
echo "Timestamp: $STAMP"
echo

rm -rf "$STAGE"

mkdir -p \
    "$FIG_STAGE" \
    "$TAB_STAGE" \
    "$DATA_STAGE" \
    "$CODE_STAGE" \
    "$META_STAGE" \
    "$ZIPDIR"


# ============================================================
# Helpers
# ============================================================

copy_file() {

    local src="$1"
    local dst="$2"

    if [[ -f "$src" ]]; then

        mkdir -p "$(dirname "$dst")"

        cp -a \
            "$src" \
            "$dst"

        echo "COPIED: $src"

    else

        echo "WARNING missing file: $src"

    fi
}


copy_dir() {

    local src="$1"
    local dst="$2"

    if [[ -d "$src" ]]; then

        mkdir -p "$(dirname "$dst")"

        cp -a \
            "$src" \
            "$dst"

        echo "COPIED DIR: $src"

    else

        echo "WARNING missing dir: $src"

    fi
}


copy_pattern() {

    local base="$1"
    local pattern="$2"
    local dst="$3"

    mkdir -p "$dst"

    find "$base" \
        -type f \
        -name "$pattern" \
        -exec cp -a {} "$dst/" \; \
        2>/dev/null || true
}


# ============================================================
# 01. MANUSCRIPT FIGURES
# ============================================================

echo
echo "============================================================"
echo "01. MANUSCRIPT FIGURES"
echo "============================================================"

mkdir -p \
    "$FIG_STAGE/current_final" \
    "$FIG_STAGE/gaia" \
    "$FIG_STAGE/sdss" \
    "$FIG_STAGE/baseline_reference"


# Existing final figures
copy_dir \
    "$ROOT/outputs/figures/final" \
    "$FIG_STAGE/current_final/final"

copy_dir \
    "$ROOT/outputs/figures/gaia" \
    "$FIG_STAGE/gaia/final"

copy_dir \
    "$ROOT/outputs/figures/sdss" \
    "$FIG_STAGE/sdss/final"


# Existing baseline/reference figures
copy_dir \
    "$ROOT/outputs/baseline_reference/gaia/figures" \
    "$FIG_STAGE/baseline_reference/gaia"

copy_dir \
    "$ROOT/outputs/baseline_v2/sdss/figures" \
    "$FIG_STAGE/baseline_reference/sdss"


# Include any new manuscript figures if generated later
if [[ -d "$ROOT/outputs/final_manuscript/figures" ]]; then

    copy_dir \
        "$ROOT/outputs/final_manuscript/figures" \
        "$FIG_STAGE/final_manuscript"

fi


# Inventory
find "$FIG_STAGE" \
    -type f \
    | sort \
    > "$FIG_STAGE/FIGURE_INVENTORY.txt"


# README
cat > "$FIG_STAGE/README.txt" <<'EOF'
MANUSCRIPT FIGURES

This directory contains publication figures generated directly from the
revision_v2 HPC analysis.

Recommended publication formats:
- TIFF: raster submission
- PNG: high-resolution preview / submission when accepted
- PDF/SVG: vector archival version

Figures should be generated at >=300 dpi; 600 dpi is preferred for final
raster output.
EOF



# ============================================================
# 02. MANUSCRIPT TABLES
# ============================================================

echo
echo "============================================================"
echo "02. MANUSCRIPT TABLES"
echo "============================================================"

mkdir -p \
    "$TAB_STAGE/manuscript" \
    "$TAB_STAGE/gaia" \
    "$TAB_STAGE/sdss" \
    "$TAB_STAGE/validation" \
    "$TAB_STAGE/robustness" \
    "$TAB_STAGE/spatial_generalization"


# Existing validated candidate tables
copy_dir \
    "$ROOT/outputs/tables" \
    "$TAB_STAGE/manuscript/candidate_tables"


# Baseline metrics
copy_dir \
    "$ROOT/outputs/baseline_reference/gaia/metrics" \
    "$TAB_STAGE/gaia/baseline"

copy_dir \
    "$ROOT/outputs/baseline_v2/sdss/metrics" \
    "$TAB_STAGE/sdss/baseline"


# Class-specific SDSS
copy_dir \
    "$ROOT/outputs/class_specific/sdss" \
    "$TAB_STAGE/sdss/class_specific"


# External validation
copy_dir \
    "$ROOT/outputs/external_validation/gaia" \
    "$TAB_STAGE/validation/gaia"

copy_dir \
    "$ROOT/outputs/external_validation/sdss" \
    "$TAB_STAGE/validation/sdss"


# Sensitivity
copy_dir \
    "$ROOT/outputs/sensitivity_reference/gaia" \
    "$TAB_STAGE/robustness/gaia"

copy_dir \
    "$ROOT/outputs/sensitivity/sdss_quality" \
    "$TAB_STAGE/robustness/sdss_quality"

copy_dir \
    "$ROOT/outputs/sensitivity/sdss_neighbors" \
    "$TAB_STAGE/robustness/sdss_neighbors"

copy_dir \
    "$ROOT/outputs/sensitivity/sdss_neighbors_fixed_weights" \
    "$TAB_STAGE/robustness/sdss_neighbors_fixed_weights"

copy_dir \
    "$ROOT/outputs/sensitivity/sdss/clipping" \
    "$TAB_STAGE/robustness/sdss_clipping"

copy_dir \
    "$ROOT/outputs/sensitivity/sdss/top_k" \
    "$TAB_STAGE/robustness/sdss_top_k"

copy_dir \
    "$ROOT/outputs/sensitivity/sdss/min_effective_features" \
    "$TAB_STAGE/robustness/sdss_min_effective_features"


# Seed stability
copy_dir \
    "$ROOT/outputs/stability_reference/gaia" \
    "$TAB_STAGE/robustness/gaia_seed_stability"

copy_dir \
    "$ROOT/outputs/stability/sdss" \
    "$TAB_STAGE/robustness/sdss_seed_stability"


# Independent regions
copy_dir \
    "$ROOT/outputs/independent_regions/gaia" \
    "$TAB_STAGE/spatial_generalization/gaia"

copy_dir \
    "$ROOT/outputs/independent_regions/sdss" \
    "$TAB_STAGE/spatial_generalization/sdss"


# If final manuscript tables have already been generated
if [[ -d "$ROOT/outputs/final_manuscript/tables" ]]; then

    copy_dir \
        "$ROOT/outputs/final_manuscript/tables" \
        "$TAB_STAGE/final_manuscript"

fi


find "$TAB_STAGE" \
    -type f \
    | sort \
    > "$TAB_STAGE/TABLE_INVENTORY.txt"


cat > "$TAB_STAGE/README.txt" <<'EOF'
MANUSCRIPT AND SUPPLEMENTARY TABLE DATA

These files are copied directly from definitive revision_v2 analysis outputs.

Important:
- External validation tables are independent of model fitting.
- Internal score-based comparisons should not be presented as external
  validation.
- SDSS GALAXY and QSO class-specific results should remain distinguishable.
EOF



# ============================================================
# 03. NEW REVISION DATA
# ============================================================

echo
echo "============================================================"
echo "03. NEW REVISION DATA"
echo "============================================================"

mkdir -p \
    "$DATA_STAGE/external_validation" \
    "$DATA_STAGE/independent_regions" \
    "$DATA_STAGE/frozen_inputs" \
    "$DATA_STAGE/reference_metadata" \
    "$DATA_STAGE/configs" \
    "$DATA_STAGE/results"


# ------------------------------------------------------------
# External-validation datasets
# ------------------------------------------------------------

copy_dir \
    "$ROOT/data/external_validation" \
    "$DATA_STAGE/external_validation/data"

copy_dir \
    "$ROOT/outputs/external_validation" \
    "$DATA_STAGE/external_validation/results"


# ------------------------------------------------------------
# Independent-region datasets
# ------------------------------------------------------------

copy_dir \
    "$ROOT/data/independent_regions" \
    "$DATA_STAGE/independent_regions/data"

copy_dir \
    "$ROOT/outputs/independent_regions" \
    "$DATA_STAGE/independent_regions/results"


# ------------------------------------------------------------
# Frozen deterministic input identifiers / queries
# ------------------------------------------------------------

for f in \
    "$ROOT/data/raw/frozen/gaia_source_ids.txt" \
    "$ROOT/data/raw/frozen/gaia_query.adql" \
    "$ROOT/data/raw/frozen/gaia_raw_frozen.csv" \
    "$ROOT/data/raw/frozen/sdss_raw_frozen.csv" \
    "$ROOT/data/raw/frozen/sdss_query.sql"
do

    if [[ -f "$f" ]]; then

        copy_file \
            "$f" \
            "$DATA_STAGE/frozen_inputs/$(basename "$f")"

    fi

done


# ------------------------------------------------------------
# Definitive processed feature tables
# ------------------------------------------------------------

for f in \
    "$ROOT/data/processed/gaia_features_reference.csv" \
    "$ROOT/data/processed/sdss_features_reference.csv" \
    "$ROOT/data/processed/gaia_feature_columns_reference.json"
do

    if [[ -f "$f" ]]; then

        copy_file \
            "$f" \
            "$DATA_STAGE/results/$(basename "$f")"

    fi

done


# ------------------------------------------------------------
# Reference queries + metadata, but NOT huge full reference CSVs
# ------------------------------------------------------------

if [[ -d "$ROOT/data/reference/gaia" ]]; then

    find "$ROOT/data/reference/gaia" \
        -maxdepth 1 \
        -type f \
        \( \
            -name "*.json" \
            -o -name "*.adql" \
            -o -name "*.txt" \
        \) \
        -exec cp -a {} "$DATA_STAGE/reference_metadata/" \;

fi


if [[ -d "$ROOT/data/reference/sdss" ]]; then

    find "$ROOT/data/reference/sdss" \
        -maxdepth 1 \
        -type f \
        \( \
            -name "*.json" \
            -o -name "*.sql" \
            -o -name "*.txt" \
        \) \
        -exec cp -a {} "$DATA_STAGE/reference_metadata/" \;

fi


# ------------------------------------------------------------
# Configurations
# ------------------------------------------------------------

copy_dir \
    "$ROOT/configs" \
    "$DATA_STAGE/configs/all_configs"


# ------------------------------------------------------------
# Scientific outputs
# ------------------------------------------------------------

copy_dir \
    "$ROOT/outputs/anomaly_baselines" \
    "$DATA_STAGE/results/anomaly_baselines"

copy_dir \
    "$ROOT/outputs/class_specific" \
    "$DATA_STAGE/results/class_specific"

copy_dir \
    "$ROOT/outputs/sensitivity" \
    "$DATA_STAGE/results/sensitivity"

copy_dir \
    "$ROOT/outputs/sensitivity_reference" \
    "$DATA_STAGE/results/sensitivity_reference"

copy_dir \
    "$ROOT/outputs/stability" \
    "$DATA_STAGE/results/stability"

copy_dir \
    "$ROOT/outputs/stability_reference" \
    "$DATA_STAGE/results/stability_reference"


find "$DATA_STAGE" \
    -type f \
    | sort \
    > "$DATA_STAGE/DATA_INVENTORY.txt"


cat > "$DATA_STAGE/README.txt" <<'EOF'
NEW REVISION DATA PACKAGE

This archive contains the new datasets and definitive numerical outputs
created during peer-review revision.

Included:
- deterministic frozen inputs
- processed feature tables
- external-validation labels and outputs
- class-specific SDSS analyses
- robustness and sensitivity outputs
- independent-region data/results
- configuration files
- reference queries and metadata

Large full reference catalogs are intentionally excluded from the standard
submission ZIP because they may be very large. They can be archived
separately if required by Zenodo or another repository.
EOF



# ============================================================
# 04. SUPPLEMENTARY CODE
# ============================================================

echo
echo "============================================================"
echo "04. SUPPLEMENTARY CODE"
echo "============================================================"

mkdir -p \
    "$CODE_STAGE/scripts_revision" \
    "$CODE_STAGE/configs"


# All revision scripts
if [[ -d "$ROOT/scripts_revision" ]]; then

    find "$ROOT/scripts_revision" \
        -maxdepth 1 \
        -type f \
        -name "*.py" \
        -exec cp -a {} "$CODE_STAGE/scripts_revision/" \;

    find "$ROOT/scripts_revision" \
        -maxdepth 1 \
        -type f \
        -name "*.sh" \
        -exec cp -a {} "$CODE_STAGE/scripts_revision/" \;

fi


# Other root scientific helper scripts if present
for f in \
    "$ROOT/reference_neighbors.py" \
    "$ROOT/04_optimize_catalog.py" \
    "$ROOT/05_analyze_catalog.py" \
    "$ROOT/06_validate_catalog.py"
do

    if [[ -f "$f" ]]; then

        copy_file \
            "$f" \
            "$CODE_STAGE/$(basename "$f")"

    fi

done


copy_dir \
    "$ROOT/configs" \
    "$CODE_STAGE/configs/all_configs"


# Freeze exact Python environment
python --version \
    > "$CODE_STAGE/python_version.txt"

pip freeze \
    > "$CODE_STAGE/requirements_exact.txt"


cat > "$CODE_STAGE/README.md" <<'EOF'
# Supplementary Code – Revision v2

This directory contains the actual scripts used during the peer-review
revision.

Principal revision blocks include:

- deterministic catalog reconstruction
- buffered reference catalogs
- Gaia reference-based features
- SDSS reference-based features
- optimizer stability analyses
- top-K sensitivity
- clipping sensitivity
- effective-feature sensitivity
- neighborhood sensitivity
- fixed-weight neighborhood diagnostics
- Gaia quality-cut sensitivity
- SDSS quality-cut sensitivity
- SDSS GALAXY/QSO class-specific analyses
- Isolation Forest / LOF / PCA-Mahalanobis baselines
- exact Gaia external labels
- Gaia VARIABLE/NSS validation
- Gaia DSC validation
- exact SDSS external labels
- SDSS spectroscopic validation
- Gaia independent-region analysis
- SDSS independent-region analysis

The original score-based internal validation script is retained for
reproducibility but should not be interpreted as independent scientific
validation.
EOF


find "$CODE_STAGE" \
    -type f \
    | sort \
    > "$CODE_STAGE/CODE_INVENTORY.txt"



# ============================================================
# 05. REPRODUCIBILITY METADATA
# ============================================================

echo
echo "============================================================"
echo "05. REPRODUCIBILITY METADATA"
echo "============================================================"

mkdir -p \
    "$META_STAGE/queries" \
    "$META_STAGE/configs" \
    "$META_STAGE/manifests"


# All query files
find "$ROOT" \
    -type f \
    \( \
        -name "*.adql" \
        -o -name "*.sql" \
    \) \
    -not -path "*/final_submission_package/*" \
    -exec cp --parents -a {} "$META_STAGE/queries/" \; \
    2>/dev/null || true


# Configs
copy_dir \
    "$ROOT/configs" \
    "$META_STAGE/configs/all_configs"


# Existing JSON metadata
find "$ROOT/data" \
    -type f \
    -name "*.json" \
    -not -path "*/final_submission_package/*" \
    -exec cp --parents -a {} "$META_STAGE/manifests/" \; \
    2>/dev/null || true


# Software versions
cat > "$META_STAGE/software_versions.txt" <<EOF
Generated: $(date -Iseconds)

$(python --version 2>&1)

numpy=$(python -c 'import numpy; print(numpy.__version__)')
pandas=$(python -c 'import pandas; print(pandas.__version__)')
scipy=$(python -c 'import scipy; print(scipy.__version__)')
sklearn=$(python -c 'import sklearn; print(sklearn.__version__)')
astropy=$(python -c 'import astropy; print(astropy.__version__)')
astroquery=$(python -c 'import astroquery; print(astroquery.__version__)')
matplotlib=$(python -c 'import matplotlib; print(matplotlib.__version__)')
yaml=$(python -c 'import yaml; print(yaml.__version__)')
EOF


# ------------------------------------------------------------
# Global project checksums for important scientific files
# ------------------------------------------------------------

echo "Generating SHA256 manifest..."

find \
    "$ROOT/configs" \
    "$ROOT/scripts_revision" \
    "$ROOT/data/external_validation" \
    "$ROOT/data/independent_regions" \
    "$ROOT/data/processed" \
    "$ROOT/outputs/external_validation" \
    "$ROOT/outputs/independent_regions" \
    "$ROOT/outputs/sensitivity" \
    "$ROOT/outputs/sensitivity_reference" \
    "$ROOT/outputs/stability" \
    "$ROOT/outputs/stability_reference" \
    "$ROOT/outputs/class_specific" \
    "$ROOT/outputs/anomaly_baselines" \
    -type f \
    -print0 \
    2>/dev/null \
    | sort -z \
    | xargs -0 sha256sum \
    > "$META_STAGE/SHA256_SCIENTIFIC_FILES.txt"


# Full inventory
find "$ROOT" \
    -type f \
    -not -path "*/final_submission_package/*" \
    -printf "%s\t%p\n" \
    | sort -k2 \
    > "$META_STAGE/FULL_PROJECT_FILE_INVENTORY.tsv"



# ============================================================
# BUILD ZIPS
# ============================================================

echo
echo "============================================================"
echo "BUILDING ZIP FILES"
echo "============================================================"

rm -f \
    "$ZIPDIR/01_MANUSCRIPT_FIGURES_FINAL.zip" \
    "$ZIPDIR/02_MANUSCRIPT_TABLES_FINAL.zip" \
    "$ZIPDIR/03_NEW_REVISION_DATA_FINAL.zip" \
    "$ZIPDIR/04_SUPPLEMENTARY_CODE_FINAL.zip" \
    "$ZIPDIR/05_REPRODUCIBILITY_METADATA_FINAL.zip"


(
    cd "$FIG_STAGE"
    zip -r -9 \
        "$ZIPDIR/01_MANUSCRIPT_FIGURES_FINAL.zip" \
        .
)


(
    cd "$TAB_STAGE"
    zip -r -9 \
        "$ZIPDIR/02_MANUSCRIPT_TABLES_FINAL.zip" \
        .
)


(
    cd "$DATA_STAGE"
    zip -r -9 \
        "$ZIPDIR/03_NEW_REVISION_DATA_FINAL.zip" \
        .
)


(
    cd "$CODE_STAGE"
    zip -r -9 \
        "$ZIPDIR/04_SUPPLEMENTARY_CODE_FINAL.zip" \
        .
)


(
    cd "$META_STAGE"
    zip -r -9 \
        "$ZIPDIR/05_REPRODUCIBILITY_METADATA_FINAL.zip" \
        .
)



# ============================================================
# ZIP CHECKSUMS
# ============================================================

echo
echo "Generating ZIP SHA256..."

(
    cd "$ZIPDIR"

    sha256sum \
        *.zip \
        > SHA256SUMS.txt
)



# ============================================================
# FINAL REPORT
# ============================================================

echo
echo "============================================================"
echo "FINAL PACKAGE SUMMARY"
echo "============================================================"

du -h \
    "$ZIPDIR"/*

echo
echo "------------------------------------------------------------"
echo "SHA256"
echo "------------------------------------------------------------"

cat \
    "$ZIPDIR/SHA256SUMS.txt"


echo
echo "------------------------------------------------------------"
echo "ZIP CONTENT TEST"
echo "------------------------------------------------------------"

for z in "$ZIPDIR"/*.zip
do

    echo
    echo "Testing: $z"

    unzip -t "$z" \
        >/dev/null

    echo "OK"

done


echo
echo "============================================================"
echo "ALL FINAL PACKAGES CREATED SUCCESSFULLY"
echo "============================================================"

echo
echo "Location:"
echo "$ZIPDIR"
echo

