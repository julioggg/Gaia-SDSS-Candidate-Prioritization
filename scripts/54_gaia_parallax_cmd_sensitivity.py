#!/usr/bin/env python3
"""Experiment 54: Gaia rank sensitivity to astrometric/photometric perturbations.

READ-ONLY analysis on the 5,205 baseline-quality Gaia targets.
- Freeze baseline weights, feature-normalization percentiles and medians.
- Reconstruct all six original Gaia features (must match to tolerance).
- Perturb parallax (illustrative additive offsets and Gaussian 1-sigma errors).
- Perturb apparent G and BP-RP using explicitly illustrative corrections.
- Recompute derived absolute-magnitude, tangential-velocity and astrometric
  quality features, keeping precomputed reference-neighborhood metrics fixed.
- Track overall rank/Spearman, top-50/100/1000 overlap, selection by parallax SNR,
  and CMD parent/selected plot. No optimization or retraining.
This is sensitivity analysis, NOT a calibrated zero-point/extinction correction.
It cannot establish physically plausible correction ranges for every object.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path.home()/"universe_revision"/"revision_v2"
DATA=ROOT/"data/processed/gaia_features_reference.csv"
WEIGHTS=[
    ROOT/"outputs/baseline_reference/gaia/metrics/weights_gaia.csv",
    ROOT/"outputs/baseline_v2/gaia/metrics/weights_gaia.csv",
]
OUT=ROOT/"outputs/reviewer_round2/gaia_measurement_sensitivity"
FEATURES=[
    "f_astrometric_quality","f_photometric_color_rarity",
    "f_absolute_magnitude_rarity","f_kinematic_interest",
    "f_local_density","f_isolation",
]
EPS=1e-6
RAW_INPUTS=[
    "ruwe","parallax_snr","bp_rp","abs_g_mag","log_vtan",
    "local_density_knn","neighbor_count_radius","nn_sep_arcsec",
]
def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def fit_params(series):
    a=pd.to_numeric(series,errors="coerce").to_numpy(float)
    a=np.where(np.isfinite(a),a,np.nan)
    med=float(np.nanmedian(a))
    a=np.where(np.isfinite(a),a,med)
    return {"median":med,"lo":float(np.quantile(a,0.01)),
            "hi":float(np.quantile(a,0.99))}

def normalize(series,params,reverse=False):
    # Accept both pandas Series and NumPy arrays (color/magnitude deviations).
    a=np.asarray(pd.to_numeric(series,errors="coerce"),dtype=float)
    a=np.where(np.isfinite(a),a,params["median"])
    lo,hi=params["lo"],params["hi"]
    x=np.clip((np.clip(a,lo,hi)-lo)/(hi-lo),0,1) if hi>lo else np.zeros(len(a))
    return 1-x if reverse else x

def build_features(d,params,medians):
    x=pd.DataFrame(index=d.index)
    n=lambda col:normalize(d[col],params[col])
    x["f_astrometric_quality"]=0.5*normalize(d["ruwe"],params["ruwe"],True)+0.5*n("parallax_snr")
    x["f_photometric_color_rarity"]=normalize(
        np.abs(d["bp_rp"].to_numpy(float)-medians["bp_rp"]),
        params["color_deviation"])
    x["f_absolute_magnitude_rarity"]=normalize(
        np.abs(d["abs_g_mag"].to_numpy(float)-medians["abs_g_mag"]),
        params["mag_deviation"])
    x["f_kinematic_interest"]=n("log_vtan")
    x["f_local_density"]=0.7*n("local_density_knn")+0.3*n("neighbor_count_radius")
    x["f_isolation"]=n("nn_sep_arcsec")
    return x

def rerender_derived(d,parallax,delta_g=0,delta_color=0):
    temp=d.copy()
    p=np.maximum(parallax,1e-6)
    temp["parallax"]=p
    temp["parallax_snr"]=p/temp["parallax_error"].to_numpy(float)
    temp["bp_rp"]=temp["bp_rp"].to_numpy(float)+delta_color
    temp["phot_g_mean_mag"]=temp["phot_g_mean_mag"].to_numpy(float)+delta_g
    temp["abs_g_mag"]=temp["phot_g_mean_mag"].to_numpy(float)+5*np.log10(p)-10
    vtan=4.74047*np.hypot(temp["pmra"].to_numpy(float),
                           temp["pmdec"].to_numpy(float))/p
    temp["log_vtan"]=np.log1p(vtan)
    return temp

def ranked(scores,ids):
    return np.lexsort((ids,-scores))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--draws",type=int,default=20)
    ap.add_argument("--seed",type=int,default=20261105)
    args=ap.parse_args()
    if args.draws<1:
        ap.error("draws must be positive")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA,dtype={"source_id":"string"})
    if df["source_id"].isna().any() or df["source_id"].duplicated().any():
        raise ValueError("Gaia source identifiers missing or duplicated")
    req=RAW_INPUTS+FEATURES+["parallax","parallax_error",
        "phot_g_mean_mag","pmra","pmdec"]
    missing=set(req)-set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for col in req:
        df[col]=pd.to_numeric(df[col],errors="coerce")
    if not np.isfinite(df[req].to_numpy(float)).all():
        raise ValueError("Nonfinite original numeric feature values")
    if (df["parallax"]<=0).any() or (df["parallax_error"]<=0).any():
        raise ValueError("Invalid parallax")
    ids=df["source_id"].astype(str).to_numpy()
    chosen=next((p for p in WEIGHTS if p.exists()),None)
    if chosen is None:
        raise FileNotFoundError("Gaia weight file not found")
    wt=pd.read_csv(chosen)
    if set(wt["feature"])!=set(FEATURES):
        raise ValueError("Gaia weight feature mismatch")
    w=wt.set_index("feature").loc[FEATURES,"weight"].to_numpy(float)
    w=w/w.sum()
    medians={"bp_rp":float(df["bp_rp"].median()),
             "abs_g_mag":float(df["abs_g_mag"].median())}
    params={key:fit_params(df[key]) for key in [
        "ruwe","parallax_snr","log_vtan","local_density_knn",
        "neighbor_count_radius","nn_sep_arcsec",
    ]}
    params["color_deviation"]=fit_params(np.abs(df["bp_rp"]-medians["bp_rp"]))
    params["mag_deviation"]=fit_params(np.abs(df["abs_g_mag"]-medians["abs_g_mag"]))
    orig=build_features(df,params,medians)
    difference={
        f:float(np.max(np.abs(orig[f].to_numpy(float)-
                               df[f].to_numpy(float)))) for f in FEATURES}
    print("Original-feature reconstruction maximum errors:",difference,flush=True)
    if any(e>1e-6 for e in difference.values()):
        raise RuntimeError("Feature definitions cannot reproduce baseline. Stop without claiming sensitivity.")
    originals=df[FEATURES].to_numpy(float)
    scores0=np.exp(np.log(np.clip(originals,EPS,1))@w)
    order0=ranked(scores0,ids)
    ranks0=np.empty(len(df),int);ranks0[order0]=np.arange(1,len(df)+1)
    rng=np.random.default_rng(args.seed)
    scenarios=[
        ("parallax_plus_0p02mas",lambda:rerender_derived(df,df["parallax"].to_numpy(float)+0.02)),
        ("parallax_minus_0p02mas",lambda:rerender_derived(df,df["parallax"].to_numpy(float)-0.02)),
        ("parallax_plus_0p05mas",lambda:rerender_derived(df,df["parallax"].to_numpy(float)+0.05)),
        ("parallax_minus_0p05mas",lambda:rerender_derived(df,df["parallax"].to_numpy(float)-0.05)),
        ("G_minus_0p10mag",lambda:rerender_derived(df,df["parallax"].to_numpy(float),delta_g=-0.10)),
        ("color_minus_0p05mag",lambda:rerender_derived(df,df["parallax"].to_numpy(float),delta_color=-0.05)),
        ("G_minus_0p10_color_minus_0p05",lambda:rerender_derived(
            df,df["parallax"].to_numpy(float),delta_g=-0.10,delta_color=-0.05)),
    ]
    for draw in range(args.draws):
        n=rng.normal(0,1,len(df))
        scenarios.append((f"parallax_one_sigma_draw_{draw+1:03d}",
            lambda n=n:rerender_derived(
                df,df["parallax"].to_numpy(float)+n*df["parallax_error"].to_numpy(float))
        ))
    rows=[]
    for name, fn in scenarios:
        d=fn()
        z=build_features(d,params,medians)[FEATURES].to_numpy(float)
        s=np.exp(np.log(np.clip(z,EPS,1))@w)
        order=ranked(s,ids)
        rank=np.empty(len(df),int);rank[order]=np.arange(1,len(df)+1)
        overlaps={f"top{k}_overlap":len(set(ids[order0[:k]])&set(ids[order[:k]]))
                  for k in (50,100,1000)}
        # Conditioning: original 5,205 kept; report crossings not refilter.
        vtan=4.74047*np.hypot(d["pmra"].to_numpy(float),
                             d["pmdec"].to_numpy(float))/d["parallax"].to_numpy(float)
        rows.append({
            "scenario":name,
            "n_targets":len(df),
            "full_rank_spearman":float(spearmanr(ranks0,rank).statistic),
            "median_absolute_rank_shift":float(np.median(np.abs(ranks0-rank))),
            "max_absolute_rank_shift":int(np.max(np.abs(ranks0-rank))),
            "n_parallax_snr_below_5_after_perturbation":int((d["parallax_snr"]<5).sum()),
            "n_vtan_above_1500_after_perturbation":int((vtan>1500).sum()),
            "parallax_floored_count":int((d["parallax"]<=1.0000001e-6).sum()),
            **overlaps
        })
        print(name,rows[-1],flush=True)
    res=pd.DataFrame(rows)
    res.to_csv(OUT/"gaia_perturbation_rank_summary.csv",index=False)
    base=pd.DataFrame({
        "source_id":ids,"baseline_rank":ranks0,"baseline_score":scores0,
        "bp_rp":df["bp_rp"],"abs_g_mag":df["abs_g_mag"],
        "parallax_snr":df["parallax_snr"],"ruwe":df["ruwe"],
        "vtan_kms":df["vtan_kms"],
    })
    base.to_csv(OUT/"gaia_rank_and_cmd_data.csv",index=False)
    selected=np.zeros(len(df),bool);selected[order0[:1000]]=True
    fig,ax=plt.subplots(figsize=(7.6,5.7))
    ax.scatter(df.loc[~selected,"bp_rp"],df.loc[~selected,"abs_g_mag"],
               s=6,alpha=0.16,label=f"Other quality-selected sources (n={(~selected).sum():,})")
    ax.scatter(df.loc[selected,"bp_rp"],df.loc[selected,"abs_g_mag"],
               s=12,alpha=0.75,label="Top 1,000 frozen ranking")
    ax.set_xlabel("Gaia BP–RP (mag, uncorrected)")
    ax.set_ylabel("Approximate absolute G magnitude (mag, uncorrected)")
    ax.set_title("Gaia baseline quality-selected sample: color–magnitude diagram")
    ax.invert_yaxis()
    ax.legend(loc="best",fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT/"gaia_cmd_parent_top1000.png",dpi=230)
    fig.savefig(OUT/"gaia_cmd_parent_top1000.pdf")
    plt.close(fig)
    (OUT/"metadata.json").write_text(json.dumps({
        "experiment":54,"data":str(DATA),"weights":str(chosen),
        "input_sha256":sha256(DATA),"weights_sha256":sha256(chosen),
        "feature_reconstruction_max_abs_error":difference,
        "normalization_frozen":True,"weights_frozen":True,
        "original_processed_target_ids_frozen":True,
        "random_seed":args.seed,"one_sigma_draws":args.draws,
        "limitations":[
            "Perturbation amplitudes are illustrative, not a source-specific Gaia zero-point or extinction model",
            "Uniform G and color shifts illustrate systematics; real extinction varies by distance/sightline",
            "Gaussian parallax perturbations assume independent errors and ignore correlations",
            "Nonpositive perturbed parallax is floored to 1e-6 mas, not a physical distance inference",
            "Selection is conditional on original quality-filtered 5205 sources; no replacement with excluded sources",
            "Neighborhood metrics kept frozen to isolate photometric/astrometric changes",
            "CMD is uncorrected and not an astrophysical classification diagnostic",
        ],
        "weights":dict(zip(FEATURES,map(float,w))),
        "feature_normalization":params,"feature_medians":medians,
    },indent=2),encoding="utf8")
    print("\nSaved",OUT,flush=True)

if __name__=="__main__":
    main()
