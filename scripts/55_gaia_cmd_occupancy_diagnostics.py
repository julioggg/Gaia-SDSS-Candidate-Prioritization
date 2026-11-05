#!/usr/bin/env python3
"""Experiment 55: Gaia CMD occupancy and marginal-rarity interpretation.

Read-only. Compare the original top-1000 vs remaining 4205 baseline-quality
Gaia sources. Evaluate 2D color-magnitude grid occupancy against an internally
defined parent baseline; distinguish marginal deviations from relative 2D
rarity. This is NOT a physical stellar-density model or a field-wide
completeness-corrected CMD. No optimizer fitting, no refiltering, no archive API.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path.home()/"universe_revision"/"revision_v2"
IN=ROOT/"outputs/reviewer_round2/gaia_measurement_sensitivity/gaia_rank_and_cmd_data.csv"
FEATURES=ROOT/"data/processed/gaia_features_reference.csv"
OUT=ROOT/"outputs/reviewer_round2/gaia_cmd_occupancy"

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rank=pd.read_csv(IN,dtype={"source_id":"string"})
    features=pd.read_csv(FEATURES,dtype={"source_id":"string"})
    cols=["source_id","f_photometric_color_rarity","f_absolute_magnitude_rarity",
          "f_kinematic_interest","f_astrometric_quality"]
    if features["source_id"].duplicated().any() or rank["source_id"].duplicated().any():
        raise ValueError("Duplicate Gaia identifiers")
    d=rank.merge(features[cols],on="source_id",how="left",validate="one_to_one",indicator=True)
    if len(d)!=5205 or not d["_merge"].eq("both").all():
        raise ValueError("Incomplete Gaia ranked-feature pairing")
    d["selected_top1000"]=d["baseline_rank"]<=1000
    c=d["bp_rp"].to_numpy(float)
    m=d["abs_g_mag"].to_numpy(float)
    if not (np.isfinite(c).all() and np.isfinite(m).all()):
        raise ValueError("Nonfinite Gaia color-magnitude values")
    # Use quantile extent to avoid distant points determining resolution.
    # Clip overflow into edge bins, and preserve separate overflow counts.
    cc=np.quantile(c,[0.01,0.99])
    mm=np.quantile(m,[0.01,0.99])
    if cc[1]<=cc[0] or mm[1]<=mm[0]:
        raise ValueError("Degenerate CMD quantile extent")
    nb=16
    xe=np.linspace(cc[0],cc[1],nb+1)
    ye=np.linspace(mm[0],mm[1],nb+1)
    ix=np.clip(np.searchsorted(xe,c,side="right")-1,0,nb-1)
    iy=np.clip(np.searchsorted(ye,m,side="right")-1,0,nb-1)
    cell=ix*nb+iy
    counts=np.bincount(cell,minlength=nb*nb)
    occupied=counts[cell]
    d["cmd_cell_occupancy_parent"]=occupied
    d["cmd_cell_occupancy_fraction"]=occupied/len(d)
    # Dense = over-represented vs random top1000. Sparse = fewer parent objects.
    # A cell with 1 object is not necessarily astrophysically rare (sampling).
    d["cmd_cell_rarity_neglogfreq"]=-np.log((occupied+0.5)/(len(d)+0.5*nb*nb))
    n_selected=int(d["selected_top1000"].sum())
    assert n_selected==1000
    d["color_deviation_from_parent_median"]=abs(d["bp_rp"]-d["bp_rp"].median())
    d["absolute_mag_deviation_from_parent_median"]=abs(
        d["abs_g_mag"]-d["abs_g_mag"].median()
    )
    cutoff=np.quantile(occupied,0.1)
    d["low_occupancy_cell"]=occupied<=cutoff
    rows=[]
    for label,subset in [
        ("parent_all",d),("top1000",d[d["selected_top1000"]]),
        ("remainder_4205",d[~d["selected_top1000"]])
    ]:
        rows.append({
            "population":label,"n":len(subset),
            "cmd_cell_occ_median":float(subset["cmd_cell_occupancy_parent"].median()),
            "cmd_cell_occ_mean":float(subset["cmd_cell_occupancy_parent"].mean()),
            "low_occ_fraction":float(subset["low_occupancy_cell"].mean()),
            "bp_rp_median":float(subset["bp_rp"].median()),
            "abs_g_mag_median":float(subset["abs_g_mag"].median()),
            "color_abs_deviation_median":float(subset["color_deviation_from_parent_median"].median()),
            "abs_mag_abs_deviation_median":float(subset["absolute_mag_deviation_from_parent_median"].median()),
            "f_color_rarity_median":float(subset["f_photometric_color_rarity"].median()),
            "f_mag_rarity_median":float(subset["f_absolute_magnitude_rarity"].median()),
            "f_kinematic_interest_median":float(subset["f_kinematic_interest"].median()),
        })
    metrics=pd.DataFrame(rows)
    metrics.to_csv(OUT/"cmd_selection_summary.csv",index=False)
    corr=[]
    for x in ("f_photometric_color_rarity","f_absolute_magnitude_rarity",
              "color_deviation_from_parent_median",
              "absolute_mag_deviation_from_parent_median","baseline_score"):
        rho=float(spearmanr(d[x],d["cmd_cell_rarity_neglogfreq"]).statistic)
        corr.append({"variable":x,"spearman_with_relative_cmd_sparse_cell":rho})
    pd.DataFrame(corr).to_csv(OUT/"cmd_marginal_vs_joint_rarity.csv",index=False)
    d[["source_id","baseline_rank","selected_top1000","bp_rp","abs_g_mag",
      "cmd_cell_occupancy_parent","cmd_cell_rarity_neglogfreq",
      "f_photometric_color_rarity","f_absolute_magnitude_rarity",
      "color_deviation_from_parent_median",
      "absolute_mag_deviation_from_parent_median"]].to_csv(
          OUT/"cmd_object_diagnostics.csv",index=False)

    grid=np.bincount(cell,minlength=nb*nb).reshape(nb,nb)
    fig,axs=plt.subplots(1,2,figsize=(13,5.3),layout="constrained")
    a=axs[0]
    im=a.pcolormesh(xe,ye,np.log1p(grid.T),shading="auto")
    fig.colorbar(im,ax=a,label="ln(1 + sampled sources / bin)")
    a.set(xlabel="BP–RP (mag, uncorrected)",
          ylabel="Approximate absolute G magnitude (uncorrected)",
          title="Quality-selected Gaia CMD: relative grid occupancy")
    a.invert_yaxis()
    b=axs[1]
    b.scatter(d.loc[~d["selected_top1000"],"bp_rp"],
              d.loc[~d["selected_top1000"],"abs_g_mag"],
              s=5,alpha=0.2,label="Other quality-selected sources")
    b.scatter(d.loc[d["selected_top1000"],"bp_rp"],
              d.loc[d["selected_top1000"],"abs_g_mag"],
              s=9,alpha=0.65,label="Top-1,000")
    b.set(xlabel="BP–RP (mag, uncorrected)",
          ylabel="Approximate absolute G magnitude (uncorrected)",
          title="Frozen-model prioritization")
    b.invert_yaxis()
    b.legend(loc="best",fontsize=8)
    fig.savefig(OUT/"gaia_cmd_density_selection.png",dpi=200)
    fig.savefig(OUT/"gaia_cmd_density_selection.pdf")
    plt.close(fig)
    (OUT/"metadata.json").write_text(json.dumps({
        "experiment":55,"method":"descriptive 16x16 equal-width grid spanning 1st–99th percentile CMD values",
        "parent":"5205 selected Gaia DR3 sources (parallax SNR >=5; RUWE<1.4; vtan<=1500)",
        "grid_boundary_rule":"1% tails clipped to edge cells for counting, not discarded",
        "input_sha256":{str(p):sha256(p) for p in [IN,FEATURES]},
        "notes":[
            "Grid density is conditional on footprint, SNR quality selection, observational completeness and bin choice.",
            "Low-count cells are descriptive relative to these 5205 objects; no physical discovery probability.",
            "A 16x16 grid is arbitrary and results must not be treated as binning-independent.",
            "Absolute magnitude uses positive-parallax approximation without extinction/zero-point correction.",
            "Not an uncertainty-propagation or calibrated stellar population analysis.",
        ]
    },indent=2),encoding="utf8")
    print("SUMMARY\n"+metrics.to_string(index=False),flush=True)
    print("RELATIVE SPARSE-CELL CORRELATIONS\n"+
          pd.DataFrame(corr).to_string(index=False),flush=True)
    print("OUTPUT",OUT,flush=True)

if __name__=="__main__":
    main()
