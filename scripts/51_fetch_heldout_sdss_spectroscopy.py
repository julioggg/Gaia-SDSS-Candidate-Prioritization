#!/usr/bin/env python3
"""Experiment 51: exact SDSS DR17 spectroscopic S/N lookup for held-out objects.

Target list: frozen object-disjoint region_A and region_G outputs from EXP48.
Match by *specobjid* AND confirm SDSS SpecObj.bestobjid = target objid.
Preserve chunk files; rerunning resumes completed SQL batches.
Never silently drop missing matches; report exact coverage for each region.

Requires astroquery installed and CEDIA outgoing network access.
This script only RETRIEVES independent outcome measurements; EXP52 evaluates
rankings with frozen baseline parameters.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path.home()/"universe_revision"/"revision_v2"
TRANSFER=ROOT/"outputs/reviewer_round2/disjoint_fixed_transfer"
OUT=ROOT/"outputs/reviewer_round2/heldout_spectroscopy"
CHUNK_DIR=OUT/"chunks"
REGIONS=("region_A","region_G")
COLS=["specobjid","bestobjid","class","z","zErr","zWarning","snMedian","rChi2","rChi2Diff"]


def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def read_targets():
    subsets=[]
    manifest=[]
    for region in REGIONS:
        file=TRANSFER/f"{region}_heldout_all_features.csv"
        if not file.is_file():
            raise FileNotFoundError(
                f"{file} missing: update experiment48 from GitHub and rerun it"
            )
        df=pd.read_csv(file,dtype={"objid":"string","specobjid":"string","class":"string"})
        for idcol in ("objid","specobjid"):
            if df[idcol].isna().any() or not df[idcol].astype(str).str.fullmatch(r"\d+").all():
                raise ValueError(f"Invalid {idcol} in {region}")
            if df[idcol].duplicated().any():
                raise ValueError(f"Nonunique {idcol} in {region}")
        df.insert(0,"region",region)
        subsets.append(df[["region","objid","specobjid","class"]])
        manifest.append({"region":region,"rows":len(df),"file":str(file),"sha256":sha256(file)})
    targets=pd.concat(subsets,ignore_index=True)
    if targets[["region","objid"]].duplicated().any():
        raise ValueError("Duplicate region-object pair")
    same_spec=targets.groupby("specobjid")["objid"].nunique()
    if (same_spec>1).any():
        raise ValueError("One specobjid maps to multiple photometric objects")
    return targets,manifest


def batch_ids(ids,chunk):
    for i in range(0,len(ids),chunk):
        yield i//chunk+1,ids[i:i+chunk]


def sql_for(ids):
    joined=",".join(ids)
    return (
      "SELECT specobjid, bestobjid, class, z, zErr, zWarning, "
      "snMedian, rChi2, rChi2Diff FROM SpecObj "
      f"WHERE specobjid IN ({joined})"
    )


def query(sql,attempts,timeout):
    from astroquery.sdss import SDSS
    last=None
    for attempt in range(1,attempts+1):
        try:
            table=SDSS.query_sql(sql,data_release=17,timeout=timeout)
            if table is None:
                return pd.DataFrame(columns=COLS)
            return table.to_pandas()
        except Exception as e:
            last=e
            print(f"    failed {attempt}/{attempts}: {e}",flush=True)
            if attempt<attempts:
                time.sleep(min(30,5*attempt))
    raise RuntimeError(f"SDSS DR17 query failed: {last}")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--batch-size",type=int,default=300)
    p.add_argument("--attempts",type=int,default=3)
    p.add_argument("--timeout",type=int,default=600)
    p.add_argument("--max-batches",type=int,default=0,
                   help="Test limited first N batches; 0=all")
    args=p.parse_args()
    if args.batch_size<1 or args.attempts<1 or args.timeout<1 or args.max_batches<0:
        p.error("Invalid batch, attempt or timeout setting")
    OUT.mkdir(parents=True,exist_ok=True)
    CHUNK_DIR.mkdir(parents=True,exist_ok=True)
    targets,files=read_targets()
    unique_ids=sorted(targets["specobjid"].astype(str).unique(),key=int)
    print(f"Target rows: {len(targets):,}; unique specobjid: {len(unique_ids):,}",
          flush=True)
    chunks=[]
    for num,ids in batch_ids(unique_ids,args.batch_size):
        if args.max_batches and num>args.max_batches:
            break
        # Hash the exact requested IDs so cache never silently belongs to a
        # different batch partition after changing --batch-size.
        id_digest=hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]
        name=f"batch_{num:04d}_{id_digest}.csv"
        fpath=CHUNK_DIR/name
        sqlpath=CHUNK_DIR/name.replace(".csv",".sql")
        if not sqlpath.exists():
            sqlpath.write_text(sql_for(ids)+"\n",encoding="utf8")
        if fpath.exists():
            df=pd.read_csv(fpath,dtype={"specobjid":"string","bestobjid":"string"})
            print(f"[{num}] cached rows={len(df)}",flush=True)
        else:
            df=query(sql_for(ids),args.attempts,args.timeout)
            if len(df):
                for c in ("specobjid","bestobjid"):
                    df[c]=df[c].astype("string")
                    if not df[c].str.fullmatch(r"\d+").fillna(False).all():
                        raise ValueError(f"Invalid returned {c} in batch {num}")
                if df["specobjid"].duplicated().any():
                    raise ValueError(f"Duplicate spectroscopic rows in batch {num}")
                unknown=set(df["specobjid"].astype(str))-set(ids)
                if unknown:
                    raise ValueError(f"Unexpected spec IDs in batch {num}")
            df.to_csv(fpath,index=False)
            print(f"[{num}] queried={len(ids)} matched={len(df)}",flush=True)
        chunks.append(df)
    if args.max_batches:
        print("Limited --max-batches test; no final coverage claimed.",flush=True)
        return
    if not chunks:
        raise RuntimeError("No chunks produced")
    labels=pd.concat(chunks,ignore_index=True)
    if labels["specobjid"].duplicated().any():
        raise ValueError("Duplicate SDSS specobjid across SQL chunks")
    labels["specobjid"]=labels["specobjid"].astype("string")
    labels["bestobjid"]=labels["bestobjid"].astype("string")
    cols=[c for c in COLS if c in labels]
    if "snMedian" not in cols or "bestobjid" not in cols:
        raise ValueError("Required returned columns missing from SDSS")
    joined=targets.merge(labels[cols],on="specobjid",how="left",
                         validate="many_to_one",indicator=True,
                         suffixes=("_target","_archive"))
    found=joined["_merge"].eq("both")
    compatible=(joined["objid"].astype(str)==joined["bestobjid"].astype(str))&found
    if not compatible[found].all():
        print(f"WARNING exact ID conflicts: {(found&~compatible).sum()}",flush=True)
    joined["exact_spectroscopic_match"]=compatible
    joined["snMedian"]=pd.to_numeric(joined["snMedian"],errors="coerce")
    joined["snr_valid"]=(compatible & np.isfinite(joined["snMedian"]) &
                         (joined["snMedian"]>0))
    joined.to_csv(OUT/"region_external_spectroscopic_labels.csv",index=False)
    summary=joined.groupby("region",dropna=False).agg(
       target_rows=("objid","count"),
       retrieved_spec_rows=("exact_spectroscopic_match","sum"),
       valid_snr_rows=("snr_valid","sum"),
    ).reset_index()
    summary["valid_snr_fraction"]=summary["valid_snr_rows"]/summary["target_rows"]
    summary.to_csv(OUT/"coverage.csv",index=False)
    (OUT/"provenance.json").write_text(json.dumps({
        "experiment":51,"data_release":"SDSS DR17",
        "matching":"Exact specobjid and bestobjid = target objid",
        "n_target_rows":len(targets),"n_unique_specobjid":len(unique_ids),
        "batch_size":args.batch_size,"source_files":files,
        "caution":"Exact-match label retrieval; no outcome threshold or model fitting",
    },indent=2),encoding="utf8")
    print("\n",summary.to_string(index=False),flush=True)
    print("Saved:",OUT,flush=True)


if __name__=="__main__":
    main()
