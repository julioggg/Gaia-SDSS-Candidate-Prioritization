#!/usr/bin/env python3
"""Experiment 56: auditable Gaia external label counts and uncertainty intervals.

Uses exact archived Gaia DR3 labels and frozen original Gaia ranking. Treat
missing DSC probability as ineligible (NOT negative); audits the pre-existing
VARIABLE, NSS and union coding and reports each separately. Bootstrap AUC
and average-precision CIs, Wilson top-k intervals, and multiplicity-adjusted
exploratory Fisher p-values across label/cutoff comparisons.
Read-only; no data acquisition or fitting.
"""
from pathlib import Path
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from scipy.stats import fisher_exact

ROOT=Path.home()/"universe_revision"/"revision_v2"
L1=ROOT/"data/external_validation/gaia/gaia_external_labels_exact.csv"
L2=ROOT/"data/external_validation/gaia/gaia_extended_external_labels_exact.csv"
RANK=ROOT/"outputs/baseline_reference/gaia/metrics/ranked_gaia.csv"
OUT=ROOT/"outputs/reviewer_round2/gaia_external_label_audit"
KS=(50,100,250,500,1000)


def hashfile(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):h.update(c)
    return h.hexdigest()


def wilson(pos,n):
    z=1.959963984540054
    p=pos/n
    d=1+z*z/n
    center=(p+z*z/(2*n))/d
    err=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return float(max(0,center-err)),float(min(1,center+err))


def boot_ci(y,s,reps,rng):
    aucs,aps=[],[]
    for _ in range(reps):
        ind=rng.integers(0,len(y),len(y))
        yy=y[ind]
        if yy.min()==yy.max():continue
        aucs.append(roc_auc_score(yy,s[ind]))
        aps.append(average_precision_score(yy,s[ind]))
    if not aucs:return (np.nan,)*4
    return (
        *[float(z) for z in np.percentile(aucs,[2.5,97.5])],
        *[float(z) for z in np.percentile(aps,[2.5,97.5])]
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bootstrap-reps",type=int,default=500)
    p.add_argument("--seed",type=int,default=20261105)
    a=p.parse_args()
    if a.bootstrap_reps<50:p.error("at least 50 bootstrap resamples")
    OUT.mkdir(parents=True,exist_ok=True)
    lab=pd.read_csv(L1,dtype={"source_id":"string"})
    ext=pd.read_csv(L2,dtype={"source_id":"string"})
    ranked=pd.read_csv(RANK,dtype={"source_id":"string"})
    for name,d in [("exact",lab),("extended",ext),("ranked",ranked)]:
        if d["source_id"].isna().any() or d["source_id"].duplicated().any():
            raise ValueError(f"{name}: missing/duplicate IDs")
    if len(ranked)!=5205 or set(lab["source_id"])!=set(ranked["source_id"]) or set(ext["source_id"])!=set(ranked["source_id"]):
        raise ValueError("Exact Gaia object coverage is not 5205, refusing partial joins")
    x=ranked[["source_id","score"]].merge(
        lab[["source_id","phot_variable_flag","non_single_star",
             "label_variable","label_nss","label_variable_or_nss"]],
        on="source_id",validate="one_to_one"
    ).merge(ext[["source_id","classprob_dsc_combmod_star",
                 "label_dsc_nonstar_050"]],on="source_id",validate="one_to_one")
    flag=x["phot_variable_flag"].astype("string").fillna("").str.strip().str.upper().eq("VARIABLE").astype(int)
    nss_raw=pd.to_numeric(x["non_single_star"],errors="coerce")
    nss=(nss_raw.fillna(0)>0).astype(int)
    union=((flag==1)|(nss==1)).astype(int)
    if not np.array_equal(flag,x["label_variable"].to_numpy(int)):
        raise ValueError("VARIABLE label reconstruction mismatch")
    if not np.array_equal(nss,x["label_nss"].to_numpy(int)):
        raise ValueError("NSS label reconstruction mismatch")
    if not np.array_equal(union,x["label_variable_or_nss"].to_numpy(int)):
        raise ValueError("VARIABLE_OR_NSS reconstruction mismatch")
    pr=pd.to_numeric(x["classprob_dsc_combmod_star"],errors="coerce")
    valid_dsc=pr.notna()&np.isfinite(pr)&pr.between(0,1,inclusive="both")
    if not np.array_equal((pr[valid_dsc]<0.5).to_numpy(int),
                          x.loc[valid_dsc,"label_dsc_nonstar_050"].to_numpy(int)):
        raise ValueError("DSC valid label reconstruction mismatch")
    labeldefs={
        "VARIABLE":(np.ones(len(x),dtype=bool),flag.to_numpy(int)),
        "NSS":(np.ones(len(x),dtype=bool),nss.to_numpy(int)),
        "VARIABLE_OR_NSS":(np.ones(len(x),dtype=bool),union.to_numpy(int)),
        "DSC_NONSTAR_050":(valid_dsc.to_numpy(bool),(pr.fillna(1)>=0).to_numpy(int)),
    }
    # Fix DSC label explicitly; absent probabilities remain excluded.
    labeldefs["DSC_NONSTAR_050"]=(
        valid_dsc.to_numpy(bool),(pr.fillna(1)<0.5).to_numpy(int))
    rng=np.random.default_rng(a.seed)
    results,toprows,counts=[],[],[]
    for name,(valid,y_all) in labeldefs.items():
        m=x.loc[valid].copy()
        m["y"]=y_all[valid]
        m=m.sort_values(["score","source_id"],ascending=[False,True])
        y=m["y"].to_numpy(int)
        scores=m["score"].to_numpy(float)
        n=len(m);pos=int(y.sum())
        counts.append({"label":name,"n_eligible":n,"positive":pos,
                       "negative":n-pos,"missing_or_ineligible":len(x)-n,
                       "prevalence":pos/n if n else np.nan})
        if 0<pos<n:
            auc=float(roc_auc_score(y,scores))
            ap=float(average_precision_score(y,scores))
            lo,hi,alo,ahi=boot_ci(y,scores,a.bootstrap_reps,rng)
        else:auc=ap=lo=hi=alo=ahi=np.nan
        results.append({"label":name,"n":n,"positive":pos,
                        "roc_auc":auc,"roc_auc_ci95_low":lo,
                        "roc_auc_ci95_high":hi,
                        "average_precision":ap,"ap_ci95_low":alo,
                        "ap_ci95_high":ahi})
        for k in KS:
            if k>n:continue
            hit=int(y[:k].sum())
            pval=float(fisher_exact([[hit,k-hit],[pos-hit,n-k-(pos-hit)]],
                                    alternative="greater").pvalue) if pos else np.nan
            cil,cih=wilson(hit,k)
            toprows.append({"label":name,"top_k":k,"n_eligible":n,
                "positives":pos,"observed":hit,"expected":k*pos/n,
                "precision":hit/k,"precision_wilson_ci95_low":cil,
                "precision_wilson_ci95_high":cih,
                "enrichment":hit/k/(pos/n) if pos else np.nan,
                "fisher_p_exploratory":pval})
    top=pd.DataFrame(toprows)
    # Family of 4 labels x 5 cutoffs (20 exploratory tests), conservative correction.
    nt=int(top["fisher_p_exploratory"].notna().sum())
    top["fisher_p_bonferroni_over_all_evaluable_tests"]=np.minimum(
        top["fisher_p_exploratory"]*nt,1.0)
    pd.DataFrame(counts).to_csv(OUT/"gaia_label_coverage.csv",index=False)
    pd.DataFrame(results).to_csv(OUT/"gaia_auc_ap_bootstrap.csv",index=False)
    top.to_csv(OUT/"gaia_topk_multiple_testing.csv",index=False)
    diagnostics={
        "phot_variable_flag_values":x["phot_variable_flag"].fillna("<MISSING>").astype(str).value_counts().to_dict(),
        "non_single_star_null_count":int(nss_raw.isna().sum()),
        "variable_and_nss_both":int(((flag==1)&(nss==1)).sum()),
        "dsc_starprob_nonnull":int(pr.notna().sum()),
        "dsc_starprob_invalid_outside_0_1":int((pr.notna()&~pr.between(0,1)).sum()),
        "missing_dsc_excluded":int((~valid_dsc).sum()),
        "input_sha256":{str(p):hashfile(p) for p in [L1,L2,RANK]},
        "method":"positive VARIABLE iff phot_variable_flag equals VARIABLE; NSS iff non_single_star>0; union by OR; DSC nonstar iff valid probability star<0.5",
        "limitations":[
            "Null NSS was mapped to zero in the original label pipeline; report this treatment rather than interpreting null as physical absence",
            "DSC nulls excluded for DSC validation, unlike original extended precomputed binary indicator",
            "Bootstrap intervals assume iid sources and do not account for spatial selection",
            "Positive astrophysical labels are archive indicators, not exhaustive ground truth",
            "All tests conditional on baseline quality-selected Gaia sample",
            "Bonferroni family covers only four labels and five tested top-K cutoffs in this script; wider exploratory benchmark family may require stricter adjustment"
        ],"bootstrap_reps":a.bootstrap_reps,"seed":a.seed,"n_bonferroni":nt,
    }
    (OUT/"metadata.json").write_text(json.dumps(diagnostics,indent=2),encoding="utf8")
    print(pd.DataFrame(counts).to_string(index=False),flush=True)
    print(pd.DataFrame(results).to_string(index=False),flush=True)
    print(top[(top["label"]=="DSC_NONSTAR_050") & (top["top_k"]<=100)].to_string(index=False),flush=True)
    print("OUTPUT",OUT,flush=True)


if __name__=="__main__":
    main()
