#!/usr/bin/env python3
"""Experiment 57: exact SDSS subclass labels, QSO score orientation, 95% bootstrap CIs.

Read-only. Class-specific original rankings, exact frozen IDs and original
external subclass columns. Missing subclasses are excluded, never negatives.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

R=Path.home()/"universe_revision/revision_v2"
LABEL=R/"data/external_validation/sdss/sdss_external_labels_exact.csv"
FEAT=R/"data/processed/sdss_features_reference.csv"
RANK={
 "GALAXY":R/"outputs/class_specific/sdss/GALAXY/ranked_galaxy.csv",
 "QSO":R/"outputs/class_specific/sdss/QSO/ranked_qso.csv"}
OUT=R/"outputs/reviewer_round2/sdss_external_label_audit"
FIELDS=["f_photometric_quality","f_color_rarity","f_redshift_interest",
        "f_redshift_precision","f_local_density","f_isolation"]

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def bootstrap(y,s,reps,rng):
 auc,ap=[],[]
 for i in range(reps):
  ix=rng.integers(len(y),size=len(y))
  yy=y[ix]
  if yy.min()==yy.max():continue
  auc.append(roc_auc_score(yy,s[ix]))
  ap.append(average_precision_score(yy,s[ix]))
 if len(auc)<max(30,int(reps*.7)):raise ValueError("Bootstrap insufficient")
 return [float(v) for v in np.percentile(auc,[2.5,97.5])],[float(v) for v in np.percentile(ap,[2.5,97.5])]

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--bootstrap-reps",type=int,default=500)
 parser.add_argument("--seed",type=int,default=20261105)
 a=parser.parse_args()
 if a.bootstrap_reps<50:parser.error("Use at least 50 bootstrap replicates")
 OUT.mkdir(parents=True,exist_ok=True)
 lab=pd.read_csv(LABEL,dtype={"objid_target":"string","objid":"string","specobjid":"string","class_target":"string"})
 feat=pd.read_csv(FEAT,dtype={"objid":"string","specobjid":"string","class":"string"})
 target_col="objid_target" if "objid_target" in lab else "objid"
 if target_col not in lab:raise ValueError("No target objid")
 for name,df,col in [("labels",lab,"specobjid"),("features",feat,"specobjid")]:
  if df[col].isna().any() or df[col].duplicated().any():raise ValueError(name+" bad IDs")
 if len(feat)!=50000 or len(lab)!=50000:raise ValueError("Wrong frozen sample size")
 lab=lab.assign(join_objid=lab[target_col])
 lab["subclass_normalized"]=lab["subClass"].astype("string").fillna("").str.strip().str.upper()
 to_merge=["join_objid","specobjid","class_target","subclass_normalized"]
 if "class_spec" in lab:to_merge.append("class_spec")
 df=feat.merge(lab[to_merge],left_on=["objid","specobjid"],
               right_on=["join_objid","specobjid"],how="left",
               validate="one_to_one",indicator=True)
 if not df["_merge"].eq("both").all():raise ValueError("Unmatched exact SDSS labels")
 if not df["class"].astype(str).eq(df["class_target"].astype(str)).all():
  raise ValueError("Target class differs")
 if "class_spec" in df:
  if not df["class_spec"].astype("string").str.strip().eq(df["class_target"]).all():
   raise ValueError("SpecObj class differs")
 coverage,summary,quantiles,top,subclasses=[],[],[],[],[]
 rng=np.random.default_rng(a.seed)
 for cls in ("GALAXY","QSO"):
  pop=df[df["class_target"].eq(cls)].copy()
  for key,val in pop["subclass_normalized"].value_counts().items():
   subclasses.append({"class":cls,"subclass":key,"n":int(val)})
  eligible=pop[pop["subclass_normalized"].ne("")].copy()
  if cls=="GALAXY":
   eligible["positive"]=eligible["subclass_normalized"].str.contains(
       "AGN|BROADLINE",regex=True).astype(int)
  else:
   eligible["positive"]=eligible["subclass_normalized"].ne("BROADLINE").astype(int)
  coverage.append({"class":cls,"n_parent":len(pop),"n_evaluable":len(eligible),
                   "n_missing_subclass":len(pop)-len(eligible),
                   "positive":int(eligible["positive"].sum()),
                   "negative":int((eligible["positive"]==0).sum())})
  ranked=pd.read_csv(RANK[cls],dtype={"objid":"string"})
  if ranked["objid"].duplicated().any() or ranked["objid"].isna().any():
   raise ValueError("Ranking duplicate object")
  d=eligible.merge(ranked[["objid","score"]],on="objid",how="left",
                   suffixes=("","_ranking"),validate="one_to_one")
  if len(d)!=len(eligible) or d["score"].isna().any():
   raise ValueError("Ranking object coverage")
  d=d.sort_values(["score","objid"],ascending=[False,True])
  s=d["score"].to_numpy(float);y=d["positive"].to_numpy(int)
  if not np.isfinite(s).all() or y.min()==y.max():
   raise ValueError("Score or label invalid")
  auc=float(roc_auc_score(y,s));auc_inv=float(roc_auc_score(y,-s))
  if abs(auc+auc_inv-1)>1e-10:raise ValueError("Inverted AUC inconsistent")
  auc_ci,ap_ci=bootstrap(y,s,a.bootstrap_reps,rng)
  summary.append({"class":cls,"n":len(d),"positive":int(y.sum()),
                  "auc_high_score":auc,"auc_ci95_low":auc_ci[0],
                  "auc_ci95_high":auc_ci[1],"auc_inverted":auc_inv,
                  "average_precision":float(average_precision_score(y,s)),
                  "ap_ci95_low":ap_ci[0],"ap_ci95_high":ap_ci[1],
                  "median_positive_score":float(np.median(s[y==1])),
                  "median_negative_score":float(np.median(s[y==0]))})
  for field in FIELDS:
   if field not in d:raise ValueError("Missing "+field)
   for positive in (0,1):
    v=d.loc[d["positive"].eq(positive),field].to_numpy(float)
    quantiles.append({"class":cls,"feature":field,"positive":positive,
                      "n":len(v),"median":float(np.median(v)),
                      "q10":float(np.quantile(v,.1)),
                      "q90":float(np.quantile(v,.9))})
  for fraction in (.005,.01,.02,.05,.1):
   k=max(1,int(round(len(d)*fraction)))
   top.append({"class":cls,"top_fraction":fraction,"k":k,
               "positives_top":int(y[:k].sum()),
               "precision":float(y[:k].mean()),
               "prevalence":float(y.mean())})
 pd.DataFrame(coverage).to_csv(OUT/"coverage.csv",index=False)
 pd.DataFrame(summary).to_csv(OUT/"auc_orientation_ci.csv",index=False)
 pd.DataFrame(quantiles).to_csv(OUT/"feature_quantiles.csv",index=False)
 pd.DataFrame(top).to_csv(OUT/"top_fraction.csv",index=False)
 pd.DataFrame(subclasses).to_csv(OUT/"subclass_counts.csv",index=False)
 (OUT/"metadata.json").write_text(json.dumps({
  "experiment":57,
  "definitions":{
   "GALAXY":"nonempty subclass contains AGN or BROADLINE",
   "QSO":"nonempty subclass does not equal BROADLINE"},
  "missing_subclass":"excluded before score evaluation",
  "score":"descending, large means high priority",
  "bootstrap_reps":a.bootstrap_reps,"seed":a.seed,
  "source_hashes":{str(p):sha(p) for p in [LABEL,FEAT,*RANK.values()]},
  "caveats":[
   "Class-specific rankings rather than mixed SDSS ranking",
   "Noncanonical QSO is metadata-based and is not confirmed astrophysical novelty",
   "Labels missing from subclass are excluded, which induces possible selection bias",
   "Bootstrap does not correct spatial dependence or catalog selection",
   "Inverse AUC is solely a coding/orientation diagnostic, not a proposed classifier"]
 },indent=2),encoding="utf8")
 print("COVERAGE\n",pd.DataFrame(coverage).to_string(index=False),flush=True)
 print("AUC\n",pd.DataFrame(summary).to_string(index=False),flush=True)
 print("QSO QUANTILES\n",pd.DataFrame(quantiles).loc[
    pd.DataFrame(quantiles)["class"].eq("QSO")].to_string(index=False),flush=True)
 print("OUTPUT",OUT,flush=True)

if __name__=="__main__":
 main()
