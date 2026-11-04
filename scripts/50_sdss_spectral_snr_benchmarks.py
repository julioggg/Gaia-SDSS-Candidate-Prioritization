#!/usr/bin/env python3
"""Experiment 50: exploratory SDSS high-spectral-S/N shortlist evaluation.

Compares baseline-optimized ranking with equal-weight means, three-feature
fixed rules and single-feature rules against external SDSS snMedian.

IMPORTANT LIMITATIONS
- Thresholds >=5 and >=10 are specified here after seeing EXP49 distributions.
  Report as EXPLORATORY, not independently preregistered or confirmatory.
- Spectroscopic S/N is external to the six ranking feature columns, but
  observationally correlated with redshift measurement precision/photometry.
- Weights were optimized on the original sample's feature matrix, not on labels.
  This is an external-measurement assessment on the training objects; validate
  on region-held-out targets with external spectroscopic measurements later.
- Photometric quality and redshift precision are model inputs. Do not validate
  with those same inputs as outcomes. No thresholds are optimized to performance.
- Bootstrap percentile CIs are exploratory; multiplicity is not accounted for.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score
from statsmodels.stats.proportion import proportion_confint

ROOT=Path.home()/"universe_revision"/"revision_v2"
FEATURE_FILE=ROOT/"data/processed/sdss_features_reference.csv"
LABEL_FILE=ROOT/"data/external_validation/sdss/sdss_external_labels_exact.csv"
WEIGHT_FILE=ROOT/"outputs/baseline_v2/sdss/metrics/weights_sdss.csv"
OUT=ROOT/"outputs/reviewer_round2/spectral_snr_outcome"
EPS=1e-6

FEATURES=[
"f_photometric_quality","f_color_rarity","f_redshift_interest",
"f_redshift_precision","f_local_density","f_isolation"]
THREE=["f_photometric_quality","f_redshift_precision","f_isolation"]
CUTS=(5.0,10.0)


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()


def load_data():
    d=pd.read_csv(FEATURE_FILE,dtype={"objid":"string","specobjid":"string","class":"string"})
    lab=pd.read_csv(LABEL_FILE,dtype={"specobjid":"string","objid_target":"string"})
    if "snMedian" not in lab:
        raise ValueError("Spectroscopic snMedian column not available")
    if lab["specobjid"].isna().any() or lab["specobjid"].duplicated().any():
        raise ValueError("Missing/duplicate specobjid in external labels")
    if d["objid"].isna().any() or d["objid"].duplicated().any() or d["specobjid"].duplicated().any():
        raise ValueError("Missing/duplicate SDSS target identifiers")
    cols=["specobjid","snMedian"]
    if "objid_target" in lab: cols.append("objid_target")
    d=d.merge(lab[cols],on="specobjid",how="left",validate="one_to_one",indicator=True)
    if not (d["_merge"]=="both").all():
        raise ValueError("Missing exact external-label coverage")
    if "objid_target" in d and d["objid"].astype(str).ne(d["objid_target"].astype(str)).any():
        raise ValueError("objid mismatch between exact labels and frozen targets")
    d["snMedian"]=pd.to_numeric(d["snMedian"],errors="coerce")
    if d["snMedian"].notna().sum()!=len(d) or not np.isfinite(d["snMedian"]).all() or (d["snMedian"]<=0).any():
        raise ValueError("Invalid/missing spectral S/N")
    return d


def scores_for(x,weights):
    logx=np.log(np.clip(x,EPS,1.0))
    weights=np.asarray(weights,float);weights=weights/weights.sum()
    return np.exp(logx@weights)


def methods(x):
    wtab=pd.read_csv(WEIGHT_FILE)
    if wtab["feature"].duplicated().any() or set(wtab["feature"])!=set(FEATURES):
        raise ValueError("Mismatch with baseline weight table")
    w=wtab.set_index("feature").loc[FEATURES,"weight"].to_numpy(float)
    x=np.clip(x,EPS,1)
    uniform=np.ones(len(FEATURES))/len(FEATURES)
    three=np.array([int(f in THREE) for f in FEATURES],float)
    three/=three.sum()
    out={
        "optimized_geometric":scores_for(x,w),
        "equal_six_geometric":scores_for(x,uniform),
        "equal_six_arithmetic":x@uniform,
        "fixed_three_geometric":scores_for(x,three),
        "fixed_three_arithmetic":x@three,
    }
    for j,f in enumerate(FEATURES):
        out[f"single_{f}"]=x[:,j].copy()
    return out,w


def ci_for_precision(hits,k):
    lo,hi=proportion_confint(hits,k,alpha=0.05,method="wilson")
    return float(lo),float(hi)


def metrics(df,score,cut,top_frac,method,cls):
    eligible=df["snMedian"].to_numpy(float)
    positive=(eligible>=cut).astype(np.int8)
    n=len(df);n_pos=int(positive.sum())
    if n_pos==0 or n_pos==n:
        raise ValueError(f"{cls}: no positive or negative at SNR cut={cut}")
    fraction=min(1.0,top_frac)
    k=max(1,min(n,int(round(n*fraction))))
    ids=df["objid"].astype(str).to_numpy()
    order=np.lexsort((ids,-score))
    hits=int(positive[order[:k]].sum())
    lo,hi=ci_for_precision(hits,k)
    prev=n_pos/n
    rho,pval=spearmanr(score,eligible)
    return {
        "class":cls,"method":method,"snr_threshold":cut,
        "n":n,"n_positive":n_pos,"prevalence":prev,
        "top_fraction":fraction,"top_k":k,
        "positives_at_k":hits,"precision_at_k":hits/k,
        "precision_wilson_ci95_low":lo,
        "precision_wilson_ci95_high":hi,
        "enrichment_over_prevalence":(hits/k)/prev,
        "average_precision":float(average_precision_score(positive,score)),
        "ap_over_prevalence":float(average_precision_score(positive,score)/prev),
        "roc_auc":float(roc_auc_score(positive,score)),
        "spearman_vs_spectral_SNR":float(rho),
        "spearman_naive_p_unadjusted":float(pval),
        "median_spectral_snr_top_k":float(np.median(eligible[order[:k]])),
    }


def paired_bootstrap_auc(y,a,b,reps,seed):
    rng=np.random.default_rng(seed)
    n=len(y);dist=[]
    for _ in range(reps):
        idx=rng.integers(0,n,n)
        if y[idx].min()==y[idx].max():
            continue
        dist.append(roc_auc_score(y[idx],a[idx])-roc_auc_score(y[idx],b[idx]))
    return (
        float(np.mean(dist)) if dist else np.nan,
        float(np.percentile(dist,2.5)) if dist else np.nan,
        float(np.percentile(dist,97.5)) if dist else np.nan,
        len(dist)
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--top-fraction",type=float,default=0.1)
    p.add_argument("--bootstrap-reps",type=int,default=500)
    p.add_argument("--seed",type=int,default=20261104)
    args=p.parse_args()
    if not 0<args.top_fraction<=1 or args.bootstrap_reps<0:
        p.error("Top fraction must be in (0,1] and bootstrap reps >= 0")
    OUT.mkdir(parents=True,exist_ok=True)
    d=load_data()
    if len(d)!=50000:
        raise ValueError("Unexpected number of frozen SDSS objects")
    x=d[FEATURES].to_numpy(float)
    if not np.isfinite(x).all() or np.min(x)<-1e-8 or np.max(x)>1+1e-8:
        raise ValueError("Invalid features")
    all_scores,w=methods(x)
    rows=[]
    comparisons=[]
    for cls in ("GALAXY","QSO"):
        sel=d["class"].astype(str).eq(cls).to_numpy()
        dc=d.loc[sel].reset_index(drop=True)
        score_sub={name:s[sel] for name,s in all_scores.items()}
        print(f"\n{cls}: n={len(dc)}",flush=True)
        for cut in CUTS:
            y=(dc["snMedian"].to_numpy(float)>=cut).astype(np.int8)
            for method,s in score_sub.items():
                rows.append(metrics(dc,s,cut,args.top_fraction,method,cls))
            for comparator in ("fixed_three_geometric","equal_six_arithmetic",
                               "equal_six_geometric"):
                delta=float(
                    roc_auc_score(y,score_sub["optimized_geometric"])-
                    roc_auc_score(y,score_sub[comparator])
                )
                bmean,blo,bhi,actual=paired_bootstrap_auc(
                    y,score_sub["optimized_geometric"],score_sub[comparator],
                    args.bootstrap_reps,args.seed
                ) if args.bootstrap_reps else (np.nan,np.nan,np.nan,0)
                comparisons.append({
                    "class":cls,"snr_threshold":cut,
                    "method_a":"optimized_geometric","method_b":comparator,
                    "auc_a_minus_b":delta,
                    "paired_bootstrap_mean":bmean,
                    "paired_bootstrap_ci95_low":blo,
                    "paired_bootstrap_ci95_high":bhi,
                    "successful_bootstrap_reps":actual,
                })
            print(f"  threshold SNR >= {cut}, prevalence={y.mean():.3f}",flush=True)

    result=pd.DataFrame(rows)
    result.to_csv(OUT/"snr_rank_benchmarks.csv",index=False)
    pd.DataFrame(comparisons).to_csv(OUT/"paired_auc_differences.csv",index=False)
    summary=result[result["method"].isin([
        "optimized_geometric","equal_six_geometric","equal_six_arithmetic",
        "fixed_three_geometric","fixed_three_arithmetic"
    ])][["class","snr_threshold","method","prevalence","top_k",
        "precision_at_k","enrichment_over_prevalence","roc_auc",
        "average_precision","median_spectral_snr_top_k"]]
    summary.to_csv(OUT/"main_comparison.csv",index=False)
    (OUT/"metadata.json").write_text(json.dumps({
        "experiment":50,
        "type":"EXPLORATORY external spectroscopic signal-to-noise audit",
        "input_sha256":{str(f):sha256(f) for f in (FEATURE_FILE,LABEL_FILE,WEIGHT_FILE)},
        "outcome":"SDSS DR17 external snMedian (spectroscopic S/N)",
        "thresholds_exploratory_after_experiment49_distribution_review":list(CUTS),
        "top_fraction":args.top_fraction,
        "paired_bootstrap_reps":args.bootstrap_reps,
        "bootstrap_seed":args.seed,
        "model_features":FEATURES,
        "canonical_weights":dict(zip(FEATURES,map(float,w))),
        "limitations":[
            "S/N is not a ranking feature, but is observationally correlated with features.",
            "Same objects as unsupervised feature-weight optimization; not external object validation.",
            "Thresholds were defined after inspecting spectral S/N distributions in experiment 49.",
            "No multiplicity correction; bootstrap CI exploratory.",
            "Do not claim value over fixed-three comparator unless outcome metrics support it.",
        ]
    },indent=2),encoding="utf8")
    print("\nMAIN COMPARISON\n",summary.to_string(index=False),flush=True)
    print("\nPAIRED AUC DIFFERENCES\n",
          pd.DataFrame(comparisons).to_string(index=False),flush=True)
    print("\nSaved to",OUT,flush=True)


if __name__=="__main__":
    main()
