#!/usr/bin/env python3
"""Experiment 53: read-only frozen-sample provenance and retention audit.

Reviewer round 2, item 5. Verifies exact ACTIVE Gaia selection thresholds
against config and reconstructs sequential retention from frozen input.
Does not redownload, refit, overwrite, or modify original analysis inputs.
The former 3,842 Gaia / 38,936 GALAXY + 11,064 QSO numbers are historical
manuscript counts, NOT re-created here; causes cannot be claimed without
verified original queries/selection logic.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import yaml

ROOT=Path.home()/"universe_revision"/"revision_v2"
CFG_G=ROOT/"configs/gaia_config_reference.yaml"
CFG_S=ROOT/"configs/sdss_config_reference.yaml"
G_RAW=ROOT/"data/raw/frozen/gaia_raw_frozen.csv"
G_PROC=ROOT/"data/processed/gaia_features_reference.csv"
G_ELIG=ROOT/"data/raw/frozen/gaia_eligible_population.csv"
S_RAW=ROOT/"data/raw/frozen/sdss_raw_frozen.csv"
S_PROC=ROOT/"data/processed/sdss_features_reference.csv"
S_ELIG=ROOT/"data/raw/frozen/sdss_htm_candidate_population.csv"
OUT=ROOT/"outputs/reviewer_round2/sample_retention_audit"


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def numeric(df,columns):
    for col in columns:
        df[col]=pd.to_numeric(df[col],errors="coerce")
    return df


def stage(rows,catalog,stage_name,data,extra=None):
    row={"catalog":catalog,"stage":stage_name,"n_rows":int(len(data))}
    if extra:
        row.update(extra)
    rows.append(row)
    print(catalog,stage_name,len(data),flush=True)


def audit_gaia(cg,stages):
    d=pd.read_csv(G_RAW,dtype={"source_id":"string"})
    if d["source_id"].duplicated().any() or d["source_id"].isna().any():
        raise ValueError("Gaia frozen IDs are not unique")
    stage(stages,"Gaia","frozen_input",d)
    names=["ra","dec","parallax","parallax_error","phot_g_mean_mag",
           "bp_rp","ruwe","pmra","pmdec"]
    d=numeric(d,names)
    d=d.dropna(subset=names).copy()
    stage(stages,"Gaia","complete_required_numeric",d)
    d=d[(d["parallax"]>0)&(d["parallax_error"]>0)].copy()
    stage(stages,"Gaia","positive_parallax_and_error",d)
    d["parallax_snr"]=d["parallax"]/d["parallax_error"]
    d["pm_total"]=np.hypot(d["pmra"],d["pmdec"])
    d["vtan_kms"]=4.74047*d["pm_total"]/d["parallax"]
    d["abs_g_mag"]=d["phot_g_mean_mag"]+5*np.log10(d["parallax"])-10
    f=cg["quality_filters"]
    min_snr=float(f["min_parallax_snr"])
    max_ruwe=float(f["max_ruwe"])
    max_vtan=float(f["max_vtan_kms"])
    d=d[d["parallax_snr"]>=min_snr].copy()
    stage(stages,"Gaia",f"parallax_snr_ge_{min_snr:g}",d)
    d=d[d["ruwe"]<max_ruwe].copy()
    stage(stages,"Gaia",f"ruwe_lt_{max_ruwe:g}",d)
    d=d[(d["vtan_kms"]>=0)&(d["vtan_kms"]<=max_vtan)].copy()
    stage(stages,"Gaia",f"vtan_kms_between_0_and_{max_vtan:g}",d)
    d=d.replace([np.inf,-np.inf],np.nan)
    cols=["parallax_snr","vtan_kms","abs_g_mag"]
    d=d.dropna(subset=cols)
    stage(stages,"Gaia","finite_derived_kinematic_and_magnitude",d)
    proc=pd.read_csv(G_PROC,dtype={"source_id":"string"})
    if proc["source_id"].isna().any() or proc["source_id"].duplicated().any():
        raise ValueError("Gaia processed IDs are not unique")
    left=set(d["source_id"].astype(str))
    right=set(proc["source_id"].astype(str))
    if left!=right:
        raise RuntimeError(
            f"Gaia retention does not match processing: missing={len(left-right)}, "
            f"unexpected={len(right-left)}; inspect original pipeline before publication"
        )
    stage(stages,"Gaia","processed_reference_verified_exact_ids",proc)
    return {"raw_n":len(pd.read_csv(G_RAW,usecols=["source_id"])),
            "processed_n":len(proc),"parallax_snr_threshold":min_snr,
            "ruwe_upper_exclusive":max_ruwe,
            "vtan_kms_upper_inclusive":max_vtan,
            "all_retained_ids_verified":True}


def audit_sdss(cs,stages):
    d=pd.read_csv(S_RAW,dtype={"objid":"string","specobjid":"string"})
    if d["objid"].duplicated().any() or d["specobjid"].duplicated().any():
        raise ValueError("SDSS raw ID duplicate")
    stage(stages,"SDSS","frozen_input",d,{
        "galaxy":int((d["class"]=="GALAXY").sum()),
        "qso":int((d["class"]=="QSO").sum())})
    names=["ra","dec","u","g","r","i","z","redshift"]
    d=numeric(d,names)
    d=d.dropna(subset=names).copy()
    stage(stages,"SDSS","complete_required_numeric",d)
    d["class"]=d["class"].astype("string").str.upper().str.strip()
    d=d[d["class"].isin(["GALAXY","QSO"])].copy()
    stage(stages,"SDSS","galaxy_or_qso",d)
    zlow=float(cs["scientific_filter"]["min_redshift"])
    zhigh=float(cs["scientific_filter"]["max_redshift"])
    valid=(d["redshift"]>zlow)&(d["redshift"]<zhigh)
    for col in ("u","g","r","i","z"):
        valid &= d[col]>0
    d=d.loc[valid].copy()
    stage(stages,"SDSS","positive_magnitudes_and_redshift_range",d)
    proc=pd.read_csv(S_PROC,dtype={"objid":"string","specobjid":"string"})
    if proc["objid"].duplicated().any() or proc["specobjid"].duplicated().any():
        raise ValueError("SDSS processed ID duplicate")
    if set(d["objid"].astype(str))!=set(proc["objid"].astype(str)):
        raise RuntimeError("SDSS processed object IDs differ from filtered frozen IDs")
    stage(stages,"SDSS","processed_reference_verified_exact_ids",proc,{
        "galaxy":int((proc["class"]=="GALAXY").sum()),
        "qso":int((proc["class"]=="QSO").sum())})
    return {"raw_n":len(pd.read_csv(S_RAW,usecols=["objid"])),
            "processed_n":len(proc),"galaxy":int((proc["class"]=="GALAXY").sum()),
            "qso":int((proc["class"]=="QSO").sum()),
            "complete_id_match":True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default=str(OUT))
    args=p.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    with CFG_G.open() as f: cg=yaml.safe_load(f)
    with CFG_S.open() as f: cs=yaml.safe_load(f)
    stages=[]
    gaia=audit_gaia(cg,stages)
    sdss=audit_sdss(cs,stages)
    optional={}
    for catalog,path in (("Gaia",G_ELIG),("SDSS",S_ELIG)):
        if path.exists():
            idcol="source_id" if catalog=="Gaia" else "objid"
            source=pd.read_csv(path,usecols=[idcol],dtype={idcol:"string"})
            optional[catalog]={"eligible_rows":len(source),
                               "eligible_unique_ids":int(source[idcol].nunique()),
                               "frozen_is_subset":bool(set(
                                   (pd.read_csv(G_RAW if catalog=="Gaia" else S_RAW,
                                                usecols=[idcol],dtype={idcol:"string"}))[idcol]
                               ).issubset(set(source[idcol])))}
            print(catalog,"eligible count:",len(source),flush=True)
    pd.DataFrame(stages).to_csv(out/"sequential_retention.csv",index=False)
    report={
        "experiment":53,
        "purpose":"Reviewer item 5: sequential retention + unambiguous final filters",
        "gaia":gaia,"sdss":sdss,"eligible_pools":optional,
        "historical_manuscript_counts": {
            "gaia_previous_processed":3842,
            "sdss_previous_galaxy":38936,
            "sdss_previous_qso":11064
        },
        "historical_counts_provenance":
            "Reviewer textual report only; NOT verified from earlier frozen data",
        "important_limitations":[
            "Sequential retention here starts from present frozen files, not a previous-version data export.",
            "Previous 3842 Gaia and SDSS class counts cannot be causally explained without previous acquisition scripts, manifests, and input IDs.",
            "Hashes prove file identity, not successful clean-environment rerun.",
            "No code, analysis input, or model outputs modified by this audit."
        ],
        "files_sha256":{str(f):sha256(f) for f in (CFG_G,CFG_S,G_RAW,G_PROC,S_RAW,S_PROC)},
    }
    (out/"audit.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print("\nSUCCESS: wrote",out/"sequential_retention.csv",out/"audit.json",flush=True)
    print(pd.DataFrame(stages).to_string(index=False),flush=True)


if __name__=="__main__":
    main()
