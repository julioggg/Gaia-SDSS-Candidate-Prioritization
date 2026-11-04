#!/usr/bin/env python3
"""Experiment 49: feasibility audit for an SDSS downstream spectroscopy use case.

Before defining success thresholds, report external spectroscopic metric coverage
and distributions in the *frozen* SDSS target sample, including snMedian, rChi2,
rChi2Diff and valid subclass labels. Compare score strata DESCRIPTIVELY only.

This audit is intentionally NOT a positive validation and does not select a
success threshold after inspecting ranking outcomes. A future confirmatory
endpoint requires prespecified thresholds and, ideally, disjoint evaluation
objects and repeat-spectroscopy data.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path.home()/"universe_revision"/"revision_v2"
F=ROOT/"data/processed/sdss_features_reference.csv"
L=ROOT/"data/external_validation/sdss/sdss_external_labels_exact.csv"
W=ROOT/"outputs/baseline_v2/sdss/metrics/weights_sdss.csv"
OUT=ROOT/"outputs/reviewer_round2/downstream_feasibility"

FEATURES=[
 "f_photometric_quality","f_color_rarity","f_redshift_interest",
 "f_redshift_precision","f_local_density","f_isolation"]
EXTERNAL=["snMedian","rChi2","rChi2Diff","velDisp","velDispErr",
          "zErr","zWarning","subClass","survey","programname","sourceType"]
QUANTILES=[0,0.01,0.05,0.25,0.50,0.75,0.95,0.99,1]


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def load():
    feat=pd.read_csv(F,dtype={"objid":"string","specobjid":"string","class":"string"})
    lab=pd.read_csv(L,dtype={"objid":"string","objid_target":"string",
                             "specobjid":"string","class_target":"string"})
    if feat["objid"].isna().any() or feat["specobjid"].isna().any():
        raise ValueError("Missing target ID")
    if feat["objid"].duplicated().any() or feat["specobjid"].duplicated().any():
        raise ValueError("Nonunique frozen target identifiers")
    if lab["specobjid"].isna().any() or lab["specobjid"].duplicated().any():
        raise ValueError("Invalid label specobjid identifiers")
    if "objid_target" in lab:
        target_label_id=lab["objid_target"]
    else:
        target_label_id=lab["objid"]
    lab=lab.copy()
    lab["label_target_objid"]=target_label_id
    cols=["specobjid","label_target_objid"]+[c for c in EXTERNAL if c in lab]
    merged=feat.merge(lab[cols],on="specobjid",how="left",validate="one_to_one",
                      indicator=True)
    matched=merged["_merge"].eq("both")
    if matched.sum()!=len(feat):
        raise RuntimeError(f"Missing labels for {len(feat)-matched.sum()} targets")
    bad=merged["objid"].astype(str).ne(merged["label_target_objid"].astype(str))
    if bad.any():
        raise RuntimeError(f"Object-ID conflicts between labels/targets: {bad.sum()}")
    if not set(FEATURES).issubset(merged):
        raise ValueError("Missing baseline feature columns")
    tab=pd.read_csv(W)
    if tab["feature"].duplicated().any() or set(tab["feature"])!=set(FEATURES):
        raise ValueError("Weight metadata mismatch")
    w=tab.set_index("feature").loc[FEATURES,"weight"].to_numpy(float)
    w=w/w.sum()
    X=merged[FEATURES].to_numpy(float)
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite features")
    merged["baseline_score"]=np.exp(np.log(np.clip(X,1e-6,1))@w)
    merged["baseline_rank"]=pd.Series(merged["baseline_score"]).rank(
        method="first",ascending=False).astype(int)
    return merged


def diagnostics(df):
    numeric=[c for c in EXTERNAL if c in df and
             c in {"snMedian","rChi2","rChi2Diff","velDisp","velDispErr","zErr","zWarning"}]
    groups={"ALL":df}
    for cls in ("GALAXY","QSO"):
        d=df[df["class"]==cls]
        groups[cls]=d
        if len(d):
            k=max(1,int(round(0.1*len(d))))
            groups[cls+"_top10pct"]=d.nlargest(k,"baseline_score")
            groups[cls+"_bottom10pct"]=d.nsmallest(k,"baseline_score")
    rows=[]
    for name,g in groups.items():
        for c in numeric:
            values=pd.to_numeric(g[c],errors="coerce")
            finite=values[np.isfinite(values)]
            row={
                "group":name,"metric":c,
                "n_objects":len(g),"n_finite":len(finite),
                "finite_fraction":len(finite)/len(g) if len(g) else np.nan,
                "n_nonpositive":int((finite<=0).sum()),
                "n_negative":int((finite<0).sum()),
            }
            for q in QUANTILES:
                row["q"+str(int(q*100)).zfill(2)]=(
                    float(finite.quantile(q)) if len(finite) else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=load()
    print(f"Matched baseline targets: {len(d):,}",flush=True)
    q=diagnostics(d)
    q.to_csv(OUT/"external_metric_coverage_quantiles.csv",index=False)
    missing=[]
    for c in EXTERNAL:
        if c not in d:
            missing.append({"metric":c,"status":"NOT_PRESENT"})
        else:
            x=d[c].astype("string")
            missing.append({
                "metric":c,"status":"AVAILABLE",
                "nonnull_count":int(d[c].notna().sum()),
                "nonempty_count":int((x.fillna("").str.strip()!="").sum()),
            })
    pd.DataFrame(missing).to_csv(OUT/"external_field_availability.csv",index=False)
    subclass=d.groupby("class",dropna=False).agg(
        n=("objid","count"),
        n_subclass_available=("subClass",lambda s:int(
            (s.astype("string").fillna("").str.strip()!="").sum()))
    ).reset_index() if "subClass" in d else pd.DataFrame()
    subclass.to_csv(OUT/"subclass_coverage.csv",index=False)
    # A compact table of matched identifiers and external fields helps audit
    # joins without distributing raw spectroscopic measurements.
    audit_cols=["objid","specobjid","class","baseline_rank","baseline_score"]+[
        c for c in EXTERNAL if c in d]
    d[audit_cols].to_csv(OUT/"matched_external_metrics_audit.csv",index=False)
    (OUT/"README.json").write_text(json.dumps({
        "experiment":49,
        "purpose":"assess feasibility before pre-registering an independent SDSS decision endpoint",
        "input_sha256":{str(p):sha256(p) for p in (F,L,W)},
        "limitations":[
            "snMedian and fit diagnostics may correlate with measured inputs and survey selection",
            "No success threshold has been chosen or assessed",
            "Existing labels were retrieved for baseline only, not held-out region A/G",
            "Do not interpret score-stratified differences as discovery validation",
        ],
    },indent=2),encoding="utf8")
    print("\nFIELD AVAILABILITY",pd.DataFrame(missing).to_string(index=False),flush=True)
    print("\nCOVERAGE / QUANTILES",q.to_string(index=False),flush=True)
    print("\nCLASS COVERAGE",subclass.to_string(index=False),flush=True)
    print("\nSaved",OUT,flush=True)


if __name__=="__main__":
    main()
