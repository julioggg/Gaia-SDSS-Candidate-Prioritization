#!/usr/bin/env python3
"""Experiment 47: row-level SDSS/Gaia overlap and HTM spatial-sampling audit.

No external archive access, optimization, or changes to frozen files. Reports:
- object-level overlap for target and buffered reference populations;
- how many region targets would survive strict baseline target/reference exclusion;
- RA/Dec spatial structure of the 50k HTM frozen baseline relative to the
  catalog candidate population (marginal KS and 2D grid occupancy);
- baseline/reference pool sizes and exact file hashes.

Do NOT call region_A or region_G independent when baseline target overlaps.
Removing overlapping targets alone does not make the reference pools disjoint.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path.home() / "universe_revision" / "revision_v2"
OUT = ROOT / "outputs/reviewer_round2/spatial_audit"
FILES = {
    "sdss_base_target": ROOT / "data/raw/frozen/sdss_raw_frozen.csv",
    "sdss_base_reference": ROOT / "data/reference/sdss/sdss_reference_buffered.csv",
    "sdss_candidate_population": ROOT / "data/raw/frozen/sdss_htm_candidate_population.csv",
    "sdss_A_target": ROOT / "data/independent_regions/sdss/region_A/raw_frozen.csv",
    "sdss_A_reference": ROOT / "data/independent_regions/sdss/region_A/reference_buffered.csv",
    "sdss_G_target": ROOT / "data/independent_regions/sdss/region_G/raw_frozen.csv",
    "sdss_G_reference": ROOT / "data/independent_regions/sdss/region_G/reference_buffered.csv",
    "gaia_base_target": ROOT / "data/raw/frozen/gaia_raw_frozen.csv",
    "gaia_base_reference": ROOT / "data/reference/gaia/gaia_reference_buffered.csv",
    "gaia_C_target": ROOT / "data/independent_regions/gaia/field_C/raw_frozen.csv",
    "gaia_C_reference": ROOT / "data/independent_regions/gaia/field_C/reference_buffered.csv",
    "gaia_D_target": ROOT / "data/independent_regions/gaia/field_D/raw_frozen.csv",
    "gaia_D_reference": ROOT / "data/independent_regions/gaia/field_D/reference_buffered.csv",
}


def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def read_data(path,idname):
    if not path.is_file():
        raise FileNotFoundError(path)
    df=pd.read_csv(path,usecols=lambda c: c in {idname,"ra","dec","htmid","sample_key"},
                   dtype={idname:"string"},low_memory=False)
    if idname not in df or df[idname].isna().any():
        raise ValueError(f"Missing identifiers in {path}")
    duplicates=int(df[idname].duplicated().sum())
    # Reference population duplicate detection is a reproducibility red flag.
    if duplicates:
        print(f"WARNING {path}: {duplicates} duplicate IDs",flush=True)
    return df


def sets_for(prefix,keys,idname,records):
    for key in keys:
        name=prefix+key
        d=records[name]
        s=set(d[idname].astype(str))
        yield name,s


def overlaps(keys,records,idname,table):
    ids={k:set(records[k][idname].astype(str)) for k in keys}
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            na,nb=len(ids[a]),len(ids[b])
            n=len(ids[a]&ids[b])
            table.append({
                "catalog":idname,"set_a":a,"set_b":b,
                "unique_n_a":na,"unique_n_b":nb,
                "shared_ids":n,
                "pct_of_a":100*n/na if na else None,
                "pct_of_b":100*n/nb if nb else None,
            })
    return ids


def spatial_audit(parent,frozen,bins):
    for f in (parent,frozen):
        for name in ("ra","dec"):
            if name not in f:
                raise ValueError(f"Missing {name} from HTM spatial audit")
            f[name]=pd.to_numeric(f[name],errors="coerce")
        if f[["ra","dec"]].isna().any().any():
            raise ValueError("Nonfinite RA/Dec")
    xedges=np.linspace(120,240,bins+1)
    yedges=np.linspace(0,60,bins+1)
    pa=np.histogram2d(parent["ra"],parent["dec"],bins=(xedges,yedges))[0]
    fr=np.histogram2d(frozen["ra"],frozen["dec"],bins=(xedges,yedges))[0]
    expected=pa/pa.sum()*fr.sum()
    valid=pa>0
    # Compares frozen counts to eligible population proportions; because
    # frozen is selected from this population these are not independent samples.
    residual=np.full_like(expected,np.nan,dtype=float)
    residual[valid]=(fr[valid]-expected[valid])/np.sqrt(expected[valid])
    cells=[]
    for i in range(bins):
        for j in range(bins):
            cells.append({
                "ra_lo":xedges[i],"ra_hi":xedges[i+1],
                "dec_lo":yedges[j],"dec_hi":yedges[j+1],
                "candidate_count":int(pa[i,j]),
                "frozen_count":int(fr[i,j]),
                "expected_if_proportionate":float(expected[i,j]),
                "standardized_residual_descriptive":float(residual[i,j])
                  if np.isfinite(residual[i,j]) else None,
            })
    ks_ra=ks_2samp(parent["ra"],frozen["ra"])
    ks_dec=ks_2samp(parent["dec"],frozen["dec"])
    summary={
        "eligible_rows":int(len(parent)),
        "frozen_rows":int(len(frozen)),
        "bins_per_axis":bins,
        "eligible_occupied_cells":int(np.sum(pa>0)),
        "frozen_occupied_cells":int(np.sum(fr>0)),
        "empty_after_frozen_of_eligible_cells":int(np.sum((pa>0)&(fr==0))),
        "max_abs_standardized_residual_descriptive":float(
            np.nanmax(np.abs(residual))),
        "KS_RA_descriptive_statistic":float(ks_ra.statistic),
        "KS_RA_naive_pvalue_NOT_valid_for_nested_samples":float(ks_ra.pvalue),
        "KS_Dec_descriptive_statistic":float(ks_dec.statistic),
        "KS_Dec_naive_pvalue_NOT_valid_for_nested_samples":float(ks_dec.pvalue),
        "note":"Eligible and frozen are nested, so ordinary 2-sample KS p-values are not an independent-sample test. Use spatial differences descriptively; formal sampling tests must match the without-replacement design."
    }
    return summary,pd.DataFrame(cells)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bins",type=int,default=12)
    args=p.parse_args()
    if args.bins<2:
        p.error("--bins must be >= 2")
    OUT.mkdir(parents=True,exist_ok=True)
    records={}
    manifest=[]
    for key,path in FILES.items():
        idname="source_id" if key.startswith("gaia_") else "objid"
        df=read_data(path,idname)
        records[key]=df
        manifest.append({
            "dataset":key,"path":str(path),"rows":len(df),
            "unique_ids":df[idname].nunique(),"sha256":digest(path),
        })
        print(key,len(df),flush=True)
    sdss=[k for k in FILES if k.startswith("sdss_") and
          k not in ("sdss_candidate_population",)]
    gaia=[k for k in FILES if k.startswith("gaia_")]
    rows=[]
    ss=overlaps(sdss,records,"objid",rows)
    gg=overlaps(gaia,records,"source_id",rows)
    overlap_df=pd.DataFrame(rows)
    overlap_df.to_csv(OUT/"all_pairwise_overlaps.csv",index=False)
    exclusion=[]
    for name,s in [("region_A",ss["sdss_A_target"]),("region_G",ss["sdss_G_target"])]:
        baseline_targets=ss["sdss_base_target"]
        baseline_refs=ss["sdss_base_reference"]
        region_ref=ss[f"sdss_{name[-1]}_reference"]
        exclusion.append({
            "region":name,"target_n":len(s),
            "target_intersects_baseline_target":len(s&baseline_targets),
            "target_intersects_baseline_reference":len(s&baseline_refs),
            "target_remaining_after_baseline_target_exclusion":len(s-baseline_targets),
            "target_remaining_after_target_AND_reference_exclusion":len(
                s-(baseline_targets|baseline_refs)),
            "region_reference_intersects_baseline_target":len(region_ref&baseline_targets),
            "region_reference_intersects_baseline_reference":len(region_ref&baseline_refs),
            "region_reference_size":len(region_ref),
            "strict_reference_disjoint":not bool(region_ref & (baseline_targets|baseline_refs))
        })
    pd.DataFrame(exclusion).to_csv(OUT/"sdss_disjointness_feasibility.csv",index=False)
    summary,grid=spatial_audit(
        records["sdss_candidate_population"].copy(),
        records["sdss_base_target"].copy(),args.bins)
    grid.to_csv(OUT/"sdss_htm_spatial_grid.csv",index=False)
    pd.DataFrame(manifest).to_csv(OUT/"file_sha256_manifest.csv",index=False)
    (OUT/"spatial_summary.json").write_text(
        json.dumps(summary,indent=2),encoding="utf8")
    print("\nKEY SDSS OVERLAPS",flush=True)
    print(overlap_df[
        (overlap_df["catalog"]=="objid")&
        overlap_df["set_a"].isin(["sdss_base_target","sdss_base_reference"])&
        overlap_df["set_b"].isin([
            "sdss_A_target","sdss_G_target","sdss_A_reference","sdss_G_reference"])
    ].to_string(index=False),flush=True)
    print("\nDISJOINTNESS",flush=True)
    print(pd.DataFrame(exclusion).to_string(index=False),flush=True)
    print("\nHTM SPATIAL AUDIT",json.dumps(summary,indent=2),flush=True)
    print("\nSaved:",OUT,flush=True)


if __name__=="__main__":
    main()
