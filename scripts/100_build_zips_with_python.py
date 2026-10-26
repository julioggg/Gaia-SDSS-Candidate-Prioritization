from pathlib import Path
import hashlib
import zipfile

ROOT = (
    Path.home()
    / "universe_revision"
    / "revision_v2"
)

FINAL = (
    ROOT
    / "final_submission_package"
)

STAGE = (
    FINAL
    / "staging"
)

ZIPDIR = (
    FINAL
    / "zips"
)

ZIPDIR.mkdir(
    parents=True,
    exist_ok=True
)


PACKAGES = [
    (
        "01_MANUSCRIPT_FIGURES_FINAL.zip",
        STAGE
        / "01_manuscript_figures"
    ),
    (
        "02_MANUSCRIPT_TABLES_FINAL.zip",
        STAGE
        / "02_manuscript_tables"
    ),
    (
        "03_NEW_REVISION_DATA_FINAL.zip",
        STAGE
        / "03_new_revision_data"
    ),
    (
        "04_SUPPLEMENTARY_CODE_FINAL.zip",
        STAGE
        / "04_supplementary_code"
    ),
    (
        "05_REPRODUCIBILITY_METADATA_FINAL.zip",
        STAGE
        / "05_reproducibility_metadata"
    ),
]


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

            h.update(
                chunk
            )

    return h.hexdigest()


def build_zip(
    zip_path,
    source_dir,
):

    if not source_dir.exists():

        raise FileNotFoundError(
            source_dir
        )


    files = sorted(
        p
        for p in source_dir.rglob("*")
        if p.is_file()
    )


    print()
    print("=" * 100)

    print(
        "Building:",
        zip_path.name
    )

    print(
        "Source:",
        source_dir
    )

    print(
        "Files:",
        len(files)
    )

    print("=" * 100)


    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as zf:

        for i, path in enumerate(
            files,
            start=1
        ):

            arcname = path.relative_to(
                source_dir
            )

            zf.write(
                path,
                arcname=str(
                    arcname
                )
            )


            if (
                i % 100 == 0
                or
                i == len(files)
            ):

                print(
                    f"  {i:,}/{len(files):,}"
                )


    # Integrity test using Python itself.
    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zf:

        bad = zf.testzip()


    if bad is not None:

        raise RuntimeError(
            f"Corrupt member in "
            f"{zip_path.name}: "
            f"{bad}"
        )


    print(
        "Integrity: OK"
    )

    print(
        "Size:",
        f"{zip_path.stat().st_size / 1024**2:.2f} MB"
    )


checksums = []


for zip_name, source_dir in PACKAGES:

    zip_path = (
        ZIPDIR
        / zip_name
    )


    if zip_path.exists():

        zip_path.unlink()


    build_zip(
        zip_path,
        source_dir
    )


    checksums.append(
        (
            sha256(
                zip_path
            ),
            zip_name
        )
    )


checksum_file = (
    ZIPDIR
    / "SHA256SUMS.txt"
)


with open(
    checksum_file,
    "w",
    encoding="utf-8"
) as f:

    for digest, name in checksums:

        f.write(
            f"{digest}  {name}\n"
        )


print()
print("=" * 110)
print("FINAL PACKAGE SUMMARY")
print("=" * 110)

for digest, name in checksums:

    p = ZIPDIR / name

    print(
        f"{name:45s} "
        f"{p.stat().st_size / 1024**2:10.2f} MB"
    )


print()
print("=" * 110)
print("SHA256")
print("=" * 110)

print(
    checksum_file.read_text(
        encoding="utf-8"
    )
)


print(
    "Saved in:",
    ZIPDIR
)

print("DONE.")
