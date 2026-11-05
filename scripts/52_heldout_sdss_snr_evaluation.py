#!/usr/bin/env python3
"""Experiment 52: SDSS held-out spectroscopy, frozen model vs simple rules.

Sources: exp48 frozen normalized regional features and exp51 DR17 spectroscopic
S/N, joined using (region, objid, specobjid), with exact-match verification.

Compare frozen weighted geometric with equal-six geometric/arithmetic and
fixed-three geometric/arithmetic. Report SNR>=5 and >=10 exploratorily, with
class-stratified AUC/AP, precision@10% and paired bootstrap differences.
No training, fitting, tuning or refitting in this script. Cross-region duplicate
IDs are audited; regions are shown separately, NOT pooled as independent draws.
Reference background overlaps with baseline and spectroscopic SNR is correlated
with ranking inputs; neither external reference independence nor causality is
claimed. Thresholds chosen after experiment49; results exploratory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path.home()/"universe_revision"/"revision_v2"
INPUT=ROOT/"outputs/reviewer_round2/disjoint_fixed_transfer"
LABELS=ROOT/"outputs/reviewer_round2/heldout_spectroscopy/region_external_spectroscopic_labels.csv"
WEIGHTS=ROOT/"outputs/baseline_v2/sdss/metrics/weights_sdss.csv"
BASE=ROOT/"data/raw/frozen/sdss_raw_frozen.csv"
OUT=ROOT/"outputs/reviewer_round2/heldout_snr_evaluation"
REGIONS=("region_A","region_G")
FEATURES=[
    "f_photometric_quality","f_color_rarity","f_redshift_interest",
    "f_redshift_precision","f_local_density","f_isolation"
]
THREE={"f_photometric_quality","f_redshift_precision","f_isolation"}
THRESHOLDS=(5.0,10.0)
EPS=1e-6


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def binary_metrics(y,score,k):
    if len(np.unique(y))!=2:
        return np.nan,np.nan,np.nan,np.nan,0
    n=len(y)
    # numeric ordinal sort with unique objid tie-order handled by upstream
    rank=np.argsort(-score,kind="stable")[:k]
    hits=int(y[rank].sum())
    return (float(roc_auc_score(y,score)),
            float(average_precision_score(y,score)),
            hits/k, float(np.median(score[rank])), hits)


def wilson(hits,n):
    z=1.959963984540054
    p=hits/n
    den=1+z*z/n
    cen=(p+z*z/(2*n))/den
    rad=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.0,float(cen-rad)),min(1.0,float(cen+rad))


def compare_bootstrap(y,score_a,score_b,fraction,reps,rng):
    n=len(y)
    da,dp=[],[]
    for _ in range(reps):
        ind=rng.integers(0,n,n)
        yy=y[ind]
        if yy.min()==yy.max():
            continue
        aa=score_a[ind]
        bb=score_b[ind]
        da.append(float(roc_auc_score(yy,aa)-roc_auc_score(yy,bb)))
        k=max(1,int(round(len(yy)*fraction)))
        pos_a=int(yy[np.argsort(-aa,kind="stable")[:k]].sum())
        pos_b=int(yy[np.argsort(-bb,kind="stable")[:k]].sum())
        dp.append((pos_a-pos_b)/k)
    if not da:
        raise RuntimeError("Bootstrap could not produce evaluable resamples")
    return {
        "auc_diff_bootstrap_ci95_low":float(np.percentile(da,2.5)),
        "auc_diff_bootstrap_ci95_high":float(np.percentile(da,97.5)),
        "precision_diff_bootstrap_ci95_low":float(np.percentile(dp,2.5)),
        "precision_diff_bootstrap_ci95_high":float(np.percentile(dp,97.5)),
        "bootstrap_valid_reps":len(da),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bootstrap-reps",type=int,default=200)
    ap.add_argument("--top-fraction",type=float,default=0.1)
    ap.add_argument("--seed",type=int,default=20261104)
    args=ap.parse_args()
    if not 0<args.top_fraction<=1 or args.bootstrap_reps<0:
        ap.error("Invalid fraction/reps")
    OUT.mkdir(parents=True,exist_ok=True)
    labels=pd.read_csv(
        LABELS,dtype={"region":"string","objid":"string","specobjid":"string"}
    )
    required={"region","objid","specobjid","snMedian","exact_spectroscopic_match","snr_valid"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Missing external columns: {required-set(labels.columns)}")
    if labels[["region","objid","specobjid"]].isna().any().any():
        raise ValueError("Missing label identifiers")
    if labels[["region","objid"]].duplicated().any():
        raise ValueError("Duplicated label region/object records")
    wt=pd.read_csv(WEIGHTS)
    if set(wt["feature"])!=set(FEATURES) or wt["feature"].duplicated().any():
        raise ValueError("Wrong canonical weights")
    w=wt.set_index("feature").loc[FEATURES,"weight"].to_numpy(float)
    w=w/w.sum()
    baseline_ids=set(pd.read_csv(BASE,usecols=["objid"],dtype={"objid":"string"})["objid"])
    data=[]
    manifest=[]
    for region in REGIONS:
        file=INPUT/f"{region}_heldout_all_features.csv"
        if not file.exists():
            raise FileNotFoundError(f"Run updated exp48: missing {file}")
        feat=pd.read_csv(
            file,dtype={"objid":"string","specobjid":"string","class":"string"}
        )
        if feat[["objid","specobjid"]].isna().any().any() or feat["objid"].duplicated().any():
            raise ValueError(f"{region}: invalid object IDs")
        if bool(feat["objid"].isin(baseline_ids).any()):
            raise ValueError(f"{region} contains baseline target objects!")
        reg_label=labels.loc[labels["region"]==region]
        d=feat.merge(
            reg_label[["objid","specobjid","snMedian",
                       "exact_spectroscopic_match","snr_valid"]],
            on=["objid","specobjid"],how="left",validate="one_to_one",
            indicator=True
        )
        if len(d)!=len(feat) or not d["_merge"].eq("both").all():
            raise ValueError(f"{region}: external label coverage is incomplete")
        d["region"]=region
        data.append(d)
        manifest.append({"region":region,"input_file":str(file),
                         "input_sha256":sha256(file),
                         "total_rows":len(d)})
    union=pd.concat(data,ignore_index=True)
    duplicated_obj=int(union["objid"].duplicated().sum())
    unique_obj=int(union["objid"].nunique())
    dup_report=pd.DataFrame([{
        "regional_rows":len(union),
        "unique_object_ids_across_regions":unique_obj,
        "duplicate_region_object_occurrences":duplicated_obj,
        "baseline_target_overlap":int(union["objid"].isin(baseline_ids).sum()),
        "note":"Separate region evaluations; do not treat pooled regional rows as independent."
    }])
    dup_report.to_csv(OUT/"cross_region_id_audit.csv",index=False)

    rows,delta_rows,coverage=[],[],[]
    rng=np.random.default_rng(args.seed)
    for d in data:
        region=str(d["region"].iloc[0])
        valid=d["snr_valid"].astype("string").str.lower().eq("true")
        exact=d["exact_spectroscopic_match"].astype("string").str.lower().eq("true")
        snr=pd.to_numeric(d["snMedian"],errors="coerce")
        valid=valid & exact & np.isfinite(snr) & (snr>0)
        coverage.append({"region":region,"n_total":len(d),
                         "n_valid":int(valid.sum()),"n_invalid":int((~valid).sum()),
                         "fraction_valid":float(valid.mean())})
        d=d.loc[valid].copy()
        d["snMedian"]=snr.loc[valid].to_numpy(float)
        X=d[FEATURES].to_numpy(float)
        if not np.isfinite(X).all() or X.min() < -1e-8 or X.max()>1+1e-8:
            raise ValueError(f"{region}: invalid frozen features")
        X=np.clip(X,EPS,1.0)
        lg=np.log(X)
        equal=np.full(len(FEATURES),1/len(FEATURES))
        three=np.array([float(c in THREE) for c in FEATURES])
        three/=three.sum()
        methods={
            "optimized_geometric_frozen":np.exp(lg@w),
            "fixed_three_geometric":np.exp(lg@three),
            "fixed_three_arithmetic":X@three,
            "equal_six_geometric":np.exp(lg@equal),
            "equal_six_arithmetic":X@equal,
        }
        # Check the score file captured by experiment48; exact numerical
        # equivalence ensures no silent change in the weighted ranking.
        maxerr=float(np.max(np.abs(methods["optimized_geometric_frozen"]-
                                     d["fixed_baseline_score"].to_numpy(float))))
        if maxerr>1e-8:
            raise RuntimeError(f"{region} model-score reconstruction error {maxerr}")
        print(f"{region}: valid={len(d):,}; max score reconstruction error={maxerr:.2g}",
              flush=True)
        for cls in ("GALAXY","QSO"):
            subset=d["class"].astype(str).eq(cls).to_numpy()
            objids=d.loc[subset,"objid"].astype(str).to_numpy()
            order=np.argsort(objids,kind="stable")
            ysnr=d.loc[subset,"snMedian"].to_numpy(float)[order]
            if len(ysnr)<100:
                raise ValueError(f"{region}/{cls}: too few targets")
            s={name:values[subset][order] for name,values in methods.items()}
            n=len(ysnr)
            k=max(1,min(n,int(round(n*args.top_fraction))))
            for cut in THRESHOLDS:
                y=(ysnr>=cut).astype(np.int8)
                if y.min()==y.max():
                    raise RuntimeError(f"No label variation in {region}/{cls} SNR>={cut}")
                prevalence=float(y.mean())
                submetrics={}
                for name,score in s.items():
                    auc,avg,prec,_,hits=binary_metrics(y,score,k)
                    low,high=wilson(hits,k)
                    submetrics[name]={"auc":auc,"precision":prec}
                    rows.append({
                        "region":region,"class":cls,"snr_threshold":cut,
                        "method":name,"n_valid":n,"n_positive":int(y.sum()),
                        "prevalence":prevalence,"top_k":k,
                        "top_hits":hits,"precision_at_k":prec,
                        "precision_wilson_ci95_low":low,
                        "precision_wilson_ci95_high":high,
                        "enrichment":prec/prevalence,
                        "roc_auc":auc,"average_precision":avg,
                        "median_snr_topk":float(np.median(
                            ysnr[np.argsort(-score,kind="stable")[:k]]))
                    })
                a="optimized_geometric_frozen"
                for b in ("fixed_three_geometric","fixed_three_arithmetic",
                          "equal_six_arithmetic","equal_six_geometric"):
                    cmp={
                        "region":region,"class":cls,"snr_threshold":cut,
                        "method_a":a,"method_b":b,
                        "auc_a_minus_b":submetrics[a]["auc"]-submetrics[b]["auc"],
                        "precision_a_minus_b":submetrics[a]["precision"]-submetrics[b]["precision"]
                    }
                    if args.bootstrap_reps>0:
                        cmp.update(compare_bootstrap(
                            y,s[a],s[b],args.top_fraction,args.bootstrap_reps,rng
                        ))
                    delta_rows.append(cmp)
                print(
                    f" {cls}, SNR>={cut}: optimized AUC={submetrics[a]['auc']:.4f}, "
                    f"3-mean AUC={submetrics['fixed_three_arithmetic']['auc']:.4f}",
                    flush=True
                )
        pd.DataFrame(rows).to_csv(OUT/"heldout_snr_main.csv",index=False)
        pd.DataFrame(delta_rows).to_csv(OUT/"heldout_paired_differences.csv",index=False)
    pd.DataFrame(coverage).to_csv(OUT/"coverage_audit.csv",index=False)
    (OUT/"metadata.json").write_text(json.dumps({
        "experiment":52,
        "model":"original SDSS optimized geometric score with baseline weights and normalization frozen",
        "weights":dict(zip(FEATURES,map(float,w))),
        "thresholds_exploratory_after_exp49":list(THRESHOLDS),
        "top_fraction":args.top_fraction,
        "bootstrap_reps":args.bootstrap_reps,
        "seed":args.seed,
        "label_file":str(LABELS),"label_sha256":sha256(LABELS),
        "weights_sha256":sha256(WEIGHTS),
        "input_features":manifest,
        "duplicate_object_occurrences_across_regions":duplicated_obj,
        "limitations":[
            "No region-level refitting or weight optimization",
            "Region targets exclude all 50k baseline target IDs",
            "Spatial reference pools overlap baseline reference",
            "SNR external to ranking feature columns but observationally correlated",
            "Cutoffs were selected after inspecting experiment49 distributions",
            "Region A and G may share objects and are evaluated separately",
            "Bootstrap assumes iid objects within each region; sky correlations remain possible",
        ],
    },indent=2),encoding="utf8")
    display=pd.DataFrame(rows)
    print("\nSUMMARY",flush=True)
    print(display[
        display["method"].isin(["optimized_geometric_frozen","fixed_three_arithmetic",
                                "fixed_three_geometric"])
    ][["region","class","snr_threshold","method","n_valid","top_k",
       "precision_at_k","roc_auc","average_precision"]].to_string(index=False),
       flush=True)
    print("\nCross-region object audit:",dup_report.to_string(index=False),
          flush=True)
    print("Output:",OUT,flush=True)


if __name__=="__main__":
    main()
