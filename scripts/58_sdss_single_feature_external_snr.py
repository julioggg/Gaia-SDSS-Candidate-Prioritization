#!/usr/bin/env python3
"""Experiment 58: single-feature SDSS held-out spectroscopy + scoring times.

Object-disjoint frozen SDSS regions A and G; no fitting/reoptimization.
Uses exact DR17 snMedian from experiment 51 and frozen normalized features
from experiment 48, comparing the optimized mixed SDSS geometric score,
three-feature fixed arithmetic/geometric, all six individual features,
equal-six arithmetic/geometric. Measure scoring-only wall time (excludes
feature extraction and differential evolution). Evaluate SNR>=5, >=10
separately within region and GALAXY/QSO; no pooling.
"""
from pathlib import Path
from time import perf_counter
import argparse, hashlib, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path.home()/"universe_revision/revision_v2"
DIR=ROOT/"outputs/reviewer_round2/disjoint_fixed_transfer"
LABELS=ROOT/"outputs/reviewer_round2/heldout_spectroscopy/region_external_spectroscopic_labels.csv"
WEIGHTS=ROOT/"outputs/baseline_v2/sdss/metrics/weights_sdss.csv"
OUT=ROOT/"outputs/reviewer_round2/simple_snr_external"
FEATURES=[
 "f_photometric_quality","f_color_rarity","f_redshift_interest",
 "f_redshift_precision","f_local_density","f_isolation"]
THREE={"f_photometric_quality","f_redshift_precision","f_isolation"}
EPS=1e-6

def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def timed(fn,n):
 samples=[]
 for _ in range(n):
  t=perf_counter()
  result=fn()
  samples.append((perf_counter()-t)*1000)
 return np.asarray(result,float),float(np.median(samples)),float(np.min(samples))

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--timing-repeats",type=int,default=15)
 p.add_argument("--top-fraction",type=float,default=0.1)
 a=p.parse_args()
 if a.timing_repeats<3 or not 0<a.top_fraction<1:p.error("Invalid args")
 OUT.mkdir(parents=True,exist_ok=True)
 label=pd.read_csv(LABELS,dtype={"region":"string","objid":"string","specobjid":"string"})
 required={"region","objid","specobjid","snMedian","exact_spectroscopic_match","snr_valid"}
 if not required.issubset(label):raise ValueError("Wrong experiment51 label schema")
 if label[["region","objid","specobjid"]].isna().any().any() or label[["region","objid"]].duplicated().any():
  raise ValueError("Label IDs invalid")
 weights=pd.read_csv(WEIGHTS)
 if weights["feature"].duplicated().any() or set(weights["feature"])!=set(FEATURES):
  raise ValueError("Frozen canonical SDSS weights mismatch")
 w=weights.set_index("feature").loc[FEATURES,"weight"].to_numpy(float)
 w=w/w.sum()
 fixed=np.array([float(f in THREE) for f in FEATURES])
 fixed/=fixed.sum()
 eq=np.full(6,1/6)
 methods={
  "optimized_geometric_frozen":("geometric",w),
  "fixed_three_arithmetic":("arithmetic",fixed),
  "fixed_three_geometric":("geometric",fixed),
  "equal_six_arithmetic":("arithmetic",eq),
  "equal_six_geometric":("geometric",eq)
 }
 for j,f in enumerate(FEATURES):
  one=np.zeros(6);one[j]=1
  methods["single_"+f]=("arithmetic",one)
 rows=[];runtime=[];meta_files=[LABELS,WEIGHTS]
 for region in ("region_A","region_G"):
  source=DIR/f"{region}_heldout_all_features.csv"
  meta_files.append(source)
  d=pd.read_csv(source,dtype={"objid":"string","specobjid":"string","class":"string"})
  if d["objid"].duplicated().any():raise ValueError("Duplicate objects")
  sub=label[label["region"].eq(region)]
  d=d.merge(sub[["objid","specobjid","snMedian","exact_spectroscopic_match","snr_valid"]],
            on=["objid","specobjid"],how="left",validate="one_to_one",indicator=True)
  if not d["_merge"].eq("both").all():raise ValueError("Incomplete exact labels")
  valid=d["snr_valid"].astype("string").str.lower().eq("true")
  exact=d["exact_spectroscopic_match"].astype("string").str.lower().eq("true")
  snr=pd.to_numeric(d["snMedian"],errors="coerce")
  d=d.loc[valid&exact&np.isfinite(snr)&(snr>0)].copy()
  d["snMedian"]=pd.to_numeric(d["snMedian"],errors="coerce")
  X=d[FEATURES].to_numpy(float)
  if not np.isfinite(X).all() or X.min() < -1e-8 or X.max()>1+1e-8:
   raise ValueError("Feature domain")
  X=np.clip(X,EPS,1)
  lg=np.log(X)
  scoremap={}
  for name,(family,mw) in methods.items():
   fn=(lambda mw=mw:np.exp(lg@mw)) if family=="geometric" else (lambda mw=mw:X@mw)
   s,med,mn=timed(fn,a.timing_repeats)
   scoremap[name]=s
   runtime.append({"region":region,"method":name,"n_scored":len(d),
                   "timing_repeats":a.timing_repeats,
                   "scoring_ms_median":med,"scoring_ms_min":mn,
                   "excludes_training_feature_engineering_and_io":True})
  err=float(np.max(np.abs(scoremap["optimized_geometric_frozen"]-
       d["fixed_baseline_score"].to_numpy(float))))
  if err>1e-8:raise RuntimeError("Frozen baseline score not reconstructed")
  for cls in ("GALAXY","QSO"):
   sub=d[d["class"].eq(cls)].copy()
   if len(sub)<100:raise ValueError("Too few samples")
   pos=np.flatnonzero(d["class"].eq(cls).to_numpy())
   sid=sub["objid"].astype(str).to_numpy()
   for cut in (5.0,10.0):
    y=(sub["snMedian"].to_numpy(float)>=cut).astype(int)
    if y.min()==y.max():raise ValueError("Degenerate label")
    n=len(y);k=max(1,int(round(n*a.top_fraction)))
    for name,s_all in scoremap.items():
     s=s_all[pos]
     ix=np.lexsort((sid,-s))[:k]
     hits=int(y[ix].sum())
     rows.append({"region":region,"class":cls,"snr_threshold":cut,
                  "method":name,"n":n,"positive":int(y.sum()),
                  "prevalence":float(y.mean()),"top_k":k,"top_hits":hits,
                  "top_precision":hits/k,"top_enrichment":hits/k/y.mean(),
                  "roc_auc":float(roc_auc_score(y,s)),
                  "average_precision":float(average_precision_score(y,s))})
  print(region,"completed, n_valid=",len(d),"max_frozen_score_error=",err,flush=True)
 pd.DataFrame(rows).to_csv(OUT/"single_feature_and_simple_rules_snr.csv",index=False)
 pd.DataFrame(runtime).to_csv(OUT/"scoring_only_runtime.csv",index=False)
 (OUT/"metadata.json").write_text(json.dumps({
  "experiment":58,"top_fraction":a.top_fraction,
  "timing_repeats":a.timing_repeats,
  "feature_names":FEATURES,
  "source_sha256":{str(f):digest(f) for f in meta_files},
  "limitations":[
   "SNR thresholds exploratory and selected after analysis 49",
   "All evaluation restricted to observed SDSS spectra; not physical discovery",
   "Regional objects disjoint but reference populations overlap",
   "Scoring times do not include preprocessing, data acquisition or evolutionary optimization",
   "No tuning of comparators to external SNR labels",
   "Arithmetic rules can be externally superior without justifying the original geometric objective",
   "Comparisons are performed separately for class and region; no pooled independence claim"]
 },indent=2),encoding="utf8")
 tab=pd.DataFrame(rows)
 print("\nROC AUC SUMMARY SNR >= 10\n"+tab.loc[tab.snr_threshold.eq(10),
  ["region","class","method","roc_auc","top_precision"]].to_string(index=False),flush=True)
 print("OUTPUT",OUT,flush=True)

if __name__=="__main__":
 main()
