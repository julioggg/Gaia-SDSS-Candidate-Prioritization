#!/usr/bin/env python3
"""Experiment 48: fixed-model, object-disjoint SDSS transfer.

Train parameters: frozen 50k baseline ONLY (no regional fitting):
  - baseline feature weights (six dimensions);
  - baseline color-index medians;
  - original raw-feature 1st/99th-percentile thresholds and imputation medians;
  - frozen neighborhood hyperparameters.
Apply to region A/G rows after excluding baseline target objid. Neighbors are
computed from the pre-existing regional reference pools, without reoptimization.

This is OBJECT-DISJOINT transfer, not independent reference-pool transfer.
Regional references are subsets of baseline reference, as experiment 47 found.
No external success labels are assessed in this experiment.
"""
import hashlib
import json
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path.home() / "universe_revision" / "revision_v2"
sys.path[:0] = [str(ROOT / "scripts_revision"), str(ROOT / "scripts")]
from reference_neighbors import add_reference_neighbor_features

BASE_RAW = ROOT / "data/raw/frozen/sdss_raw_frozen.csv"
BASE_FEATURES = ROOT / "data/processed/sdss_features_reference.csv"
BASE_REFERENCE = ROOT / "data/reference/sdss/sdss_reference_buffered.csv"
BASE_WEIGHTS = ROOT / "outputs/baseline_v2/sdss/metrics/weights_sdss.csv"
OUT = ROOT / "outputs/reviewer_round2/disjoint_fixed_transfer"

FEATURES = [
    "f_photometric_quality", "f_color_rarity", "f_redshift_interest",
    "f_redshift_precision", "f_local_density", "f_isolation",
]
RAW_VARS = [
    "mean_phot_error", "color_distance", "log_redshift", "redshift_error",
    "local_density_knn", "neighbor_count_radius",
    "redshift_neighbor_count", "nn_sep_arcsec",
]
REVERSED = {"mean_phot_error","redshift_error"}
EPS=1e-6


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def numeric(df):
    for c in [
        "ra","dec","u","g","r","i","z",
        "err_u","err_g","err_r","err_i","err_z",
        "redshift","redshift_error"
    ]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    return df


def base_color_and_static(df, medians=None):
    df=numeric(df.copy())
    df["u_g"]=df["u"]-df["g"]
    df["g_r"]=df["g"]-df["r"]
    df["r_i"]=df["r"]-df["i"]
    df["i_z"]=df["i"]-df["z"]
    colors=["u_g","g_r","r_i","i_z"]
    if medians is None:
        medians={c:float(df[c].median()) for c in colors}
    distances=np.zeros(len(df),float)
    for c in colors:
        distances += (df[c].to_numpy(float)-medians[c])**2
    df["color_distance"]=np.sqrt(distances)
    df["mean_phot_error"]=df[
        ["err_u","err_g","err_r","err_i","err_z"]
    ].replace([np.inf,-np.inf],np.nan).mean(axis=1)
    df["log_redshift"]=np.log1p(df["redshift"].to_numpy(float))
    return df,medians


def frozen_quantiles(df,cols):
    params={}
    for col in cols:
        arr=pd.to_numeric(df[col],errors="coerce").to_numpy(float)
        arr[~np.isfinite(arr)]=np.nan
        if np.isnan(arr).all():
            raise ValueError(f"No finite baseline values for {col}")
        median=float(np.nanmedian(arr))
        arr=np.where(np.isfinite(arr),arr,median)
        params[col]={
            "median":median,
            "lo":float(np.quantile(arr,0.01)),
            "hi":float(np.quantile(arr,0.99)),
        }
    return params


def apply_norm(s,param,invert=False):
    v=pd.to_numeric(s,errors="coerce").to_numpy(float)
    v=np.where(np.isfinite(v),v,param["median"])
    lo,hi=param["lo"],param["hi"]
    if hi<=lo:
        x=np.zeros(len(v),float)
    else:
        x=(np.clip(v,lo,hi)-lo)/(hi-lo)
    return 1-x if invert else x


def make_features(df,params):
    df=df.copy()
    n=lambda c,inv=False:apply_norm(df[c],params[c],inv)
    df["f_photometric_quality"]=n("mean_phot_error",True)
    df["f_color_rarity"]=n("color_distance")
    df["f_redshift_interest"]=n("log_redshift")
    df["f_redshift_precision"]=n("redshift_error",True)
    df["f_local_density"]=(
        0.50*n("local_density_knn")
        +0.25*n("neighbor_count_radius")
        +0.25*n("redshift_neighbor_count")
    )
    df["f_isolation"]=n("nn_sep_arcsec")
    return df


def read_csv(path):
    return pd.read_csv(path,dtype={"objid":"string","specobjid":"string","class":"string"})


def score(df,w):
    x=df[FEATURES].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite normalized features")
    if np.min(x)<-1e-8 or np.max(x)>1+1e-8:
        raise ValueError("Normalized features outside [0,1]")
    return np.exp(np.log(np.clip(x,EPS,1))@w)


def ranked(df,scores,k):
    order=np.lexsort((df["objid"].astype(str).to_numpy(),-scores))
    top=order[:k]
    return top,order


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--top-k",type=int,default=1000)
    p.add_argument("--batch-size",type=int,default=2000)
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.top_k<1 or args.batch_size<1:
        p.error("top-k and batch-size must be positive")

    baseline=read_csv(BASE_FEATURES)
    raw=read_csv(BASE_RAW)
    if baseline["objid"].duplicated().any() or raw["objid"].duplicated().any():
        raise ValueError("Duplicate baseline target IDs")
    if set(baseline["objid"].astype(str))!=set(raw["objid"].astype(str)):
        raise ValueError("Baseline processed and raw IDs mismatch")
    base_ids=set(raw["objid"].astype(str))
    wt=pd.read_csv(BASE_WEIGHTS)
    if wt["feature"].duplicated().any() or set(wt["feature"])!=set(FEATURES):
        raise ValueError("Weight feature mismatch")
    w=wt.set_index("feature").loc[FEATURES,"weight"].to_numpy(float)
    w=w/w.sum()

    raw,medians=base_color_and_static(raw)
    # Fit normalizer using baseline processed RAW/derived neighborhood variables.
    # Crucial: baseline color distance computed with baseline median; same
    # medians must then be used on regional held-out objects.
    merged=baseline[["objid"]+[
        c for c in RAW_VARS if c in baseline.columns
    ]].copy()
    merged["objid"]=merged["objid"].astype(str)
    merge_color=raw[["objid","color_distance"]].copy()
    merge_color["objid"]=merge_color["objid"].astype(str)
    if "color_distance" not in merged.columns:
        merged=merged.merge(merge_color,on="objid",validate="one_to_one")
    else:
        merged=merged.drop(columns=["color_distance"]).merge(
            merge_color,on="objid",validate="one_to_one"
        )
    params=frozen_quantiles(merged,RAW_VARS)
    reconstructed=make_features(merged,params).set_index("objid")
    baseline_byid=baseline.copy()
    baseline_byid["objid"]=baseline_byid["objid"].astype(str)
    baseline_byid=baseline_byid.set_index("objid")
    differences={}
    for name in FEATURES:
        a=reconstructed[name].reindex(baseline_byid.index).to_numpy(float)
        b=baseline_byid[name].to_numpy(float)
        differences[name]=float(np.nanmax(np.abs(a-b)))
    print("Baseline feature reconstruction max-absolute errors:",
          json.dumps(differences,indent=2),flush=True)
    # Halt rather than silently present wrong "frozen" preprocessing.
    if any(v>1e-5 for v in differences.values()):
        raise RuntimeError(
            "Baseline reconstruction does not match existing normalized features; "
            "do not claim fixed-preprocessing transfer. Inspect feature construction."
        )

    rows=[]
    for region in ["region_A","region_G"]:
        directory=ROOT/"data/independent_regions/sdss"/region
        targets=read_csv(directory/"raw_frozen.csv")
        reference=read_csv(directory/"reference_buffered.csv")
        if targets["objid"].duplicated().any() or reference["objid"].duplicated().any():
            raise ValueError(f"Duplicate ID in {region}")
        before=len(targets)
        targets=targets.loc[~targets["objid"].astype(str).isin(base_ids)].copy()
        if len(targets)==0:
            raise ValueError(f"No held-out targets in {region}")
        # Keep the REGION reference context intact; any self-match must be
        # excluded by reference_neighbors (same as original baseline).
        targets,regional_medians=base_color_and_static(targets,medians)
        reference=numeric(reference)
        targets=add_reference_neighbor_features(
            targets=targets,reference=reference,
            radius_arcsec=300.0,kth_neighbor=5,
            redshift_radius_arcsec=600.0,dz=0.05,
            batch_size=args.batch_size
        )
        targets=make_features(targets,params)
        valid=targets[FEATURES].notna().all(axis=1)
        targets=targets.loc[valid].reset_index(drop=True)
        values=score(targets,w)
        k=min(args.top_k,len(targets))
        top,all_indices=ranked(targets,values,k)
        output=targets.iloc[top][
            ["objid","specobjid","class","ra","dec","redshift"]+FEATURES
        ].copy()
        output.insert(0,"rank",range(1,k+1))
        output["score"]=values[top]
        output.to_csv(OUT/f"{region}_heldout_fixed_model_topk.csv",index=False)
        pd.DataFrame({
            "objid":targets["objid"],"fixed_baseline_score":values
        }).to_csv(OUT/f"{region}_heldout_scores.csv",index=False)
        rows.append({
            "region":region,"initial_targets":before,
            "excluded_baseline_targets":before-len(targets),
            "retained_targets":len(targets),
            "top_k":k,
            "mean_score_all":float(np.mean(values)),
            "mean_score_topk":float(np.mean(values[top])),
            "qso_fraction_all":float((targets["class"]=="QSO").mean()),
            "qso_fraction_topk":float((targets.iloc[top]["class"]=="QSO").mean()),
            "reoptimized":False,
            "baseline_parameters_frozen":True,
            "regional_reference_overlaps_baseline_reference":True,
        })
        print("Finished",region,rows[-1],flush=True)

    pd.DataFrame(rows).to_csv(OUT/"fixed_transfer_summary.csv",index=False)
    (OUT/"frozen_preprocessing.json").write_text(json.dumps({
        "experiment":48,"color_medians":medians,"frozen_1_99":params,
        "frozen_weights":dict(zip(FEATURES,map(float,w))),
        "baseline_reconstruction_max_abs_error":differences,
        "reference_buffer_note":
          "Reference pool overlap is retained and declared; no target overlap",
        "files":{
            str(x):digest(x)
            for x in [BASE_RAW,BASE_FEATURES,BASE_REFERENCE,BASE_WEIGHTS]
        },
        "limitations":[
            "No reoptimization, no refitting normalization on region",
            "No external success endpoint assessed",
            "Buffered reference pools overlap baseline",
            "Regional target cuts exclude only original baseline target IDs",
        ]
    },indent=2),encoding="utf8")
    print("\nRESULTS",pd.DataFrame(rows).to_string(index=False),flush=True)
    print("Saved",OUT,flush=True)


if __name__=="__main__":
    main()
