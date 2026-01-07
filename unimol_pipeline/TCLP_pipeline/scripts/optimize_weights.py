#!/usr/bin/env python3
"""
Optimize ensemble weights for TCPL scores using grid search.
Finds optimal weights that maximize Spearman correlation and AUC with ToxRefDB.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from itertools import product
import argparse


def load_data(base_dir: str):
    """Load required data."""
    scores_path = os.path.join(base_dir, "processed", "tcpl_chemical_scores_final.csv")
    toxref_path = os.path.join(base_dir, "source", "tox21_toxrefdb_matched_via_cas.csv")
    bridge_path = os.path.join(base_dir, "processed", "chemical_bridge_table.csv")
    
    scores = pd.read_csv(scores_path)
    toxref = pd.read_csv(toxref_path)
    bridge = pd.read_csv(bridge_path)
    
    return scores, toxref, bridge


def prepare_merged_data(scores, toxref, bridge):
    """Prepare merged dataset with all score variants and ToxRefDB POD."""
    # Map ToxRefDB to chid
    toxref_norm = toxref.rename(columns={"CAS_NORM": "casn_normalized"})
    toxref_agg = toxref_norm.groupby("casn_normalized").agg({
        "POD_MGKGDAY": "min",
        "S_global": "mean",
        "S_potency": "mean",
    }).reset_index()
    
    bridge_sub = bridge[["casn_normalized", "chid"]].drop_duplicates()
    toxref_with_chid = pd.merge(toxref_agg, bridge_sub, on="casn_normalized", how="left")
    toxref_with_chid = toxref_with_chid.dropna(subset=["chid"])
    
    # Merge with scores
    merged = pd.merge(scores[["chid", "S_c"]], toxref_with_chid, on="chid", how="inner")
    merged = merged.dropna(subset=["POD_MGKGDAY", "S_global", "S_potency", "S_c"])
    
    # Flip S_global and S_potency (negative correlation -> positive)
    merged["S_global_flip"] = 1 - merged["S_global"]
    merged["S_potency_flip"] = 1 - merged["S_potency"]
    
    # Compute -log10(POD) for Spearman
    merged["neglog10_POD"] = -np.log10(merged["POD_MGKGDAY"].clip(lower=1e-10))
    
    print(f"Prepared {len(merged)} chemicals with all scores and ToxRefDB data")
    return merged


def compute_metrics(x, y_cont, pod, tau=10.0):
    """Compute Spearman and AUC for given score."""
    rho, pval = spearmanr(x, y_cont)
    y_true = (pod <= tau).astype(int)
    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        auc = roc_auc_score(y_true, x)
    else:
        auc = np.nan
    return rho, pval, auc


def grid_search_weights(merged, step=0.05, tau=10.0):
    """Grid search for optimal ensemble weights."""
    print(f"\n=== Grid Search (step={step}, τ={tau}) ===")
    
    S_global_flip = merged["S_global_flip"].values
    S_potency_flip = merged["S_potency_flip"].values
    S_c = merged["S_c"].values
    y_cont = merged["neglog10_POD"].values
    pod = merged["POD_MGKGDAY"].values
    
    # Generate weight combinations that sum to 1
    weights_range = np.arange(0, 1 + step, step)
    best_rho = {"weights": None, "rho": -1, "auc": 0}
    best_auc = {"weights": None, "rho": 0, "auc": 0}
    best_combined = {"weights": None, "score": -1, "rho": 0, "auc": 0}
    
    results = []
    
    for w1 in weights_range:
        for w2 in weights_range:
            w3 = 1 - w1 - w2
            if w3 < -1e-6 or w3 > 1 + 1e-6:
                continue
            w3 = max(0, min(1, w3))  # clip numerical errors
            
            # Compute ensemble
            S_ensemble = w1 * S_global_flip + w2 * S_potency_flip + w3 * S_c
            
            # Compute metrics
            rho, pval, auc = compute_metrics(S_ensemble, y_cont, pod, tau)
            
            if np.isnan(rho) or np.isnan(auc):
                continue
            
            results.append({
                "w_global": w1, "w_potency": w2, "w_sc": w3,
                "spearman_rho": rho, "spearman_p": pval, "auc": auc
            })
            
            # Track best
            if rho > best_rho["rho"]:
                best_rho = {"weights": (w1, w2, w3), "rho": rho, "auc": auc}
            if auc > best_auc["auc"]:
                best_auc = {"weights": (w1, w2, w3), "rho": rho, "auc": auc}
            
            # Combined score: average of normalized rho and auc
            combined = 0.5 * rho + 0.5 * auc
            if combined > best_combined["score"]:
                best_combined = {"weights": (w1, w2, w3), "score": combined, 
                                "rho": rho, "auc": auc}
    
    print(f"\nTotal combinations tested: {len(results)}")
    print(f"\nBest by Spearman ρ:")
    print(f"  Weights: w_global={best_rho['weights'][0]:.2f}, "
          f"w_potency={best_rho['weights'][1]:.2f}, w_sc={best_rho['weights'][2]:.2f}")
    print(f"  ρ={best_rho['rho']:.4f}, AUC={best_rho['auc']:.4f}")
    
    print(f"\nBest by AUC:")
    print(f"  Weights: w_global={best_auc['weights'][0]:.2f}, "
          f"w_potency={best_auc['weights'][1]:.2f}, w_sc={best_auc['weights'][2]:.2f}")
    print(f"  ρ={best_auc['rho']:.4f}, AUC={best_auc['auc']:.4f}")
    
    print(f"\nBest combined (0.5*ρ + 0.5*AUC):")
    print(f"  Weights: w_global={best_combined['weights'][0]:.2f}, "
          f"w_potency={best_combined['weights'][1]:.2f}, w_sc={best_combined['weights'][2]:.2f}")
    print(f"  ρ={best_combined['rho']:.4f}, AUC={best_combined['auc']:.4f}")
    
    return pd.DataFrame(results), best_rho, best_auc, best_combined


def main():
    parser = argparse.ArgumentParser(description="Optimize TCPL ensemble weights")
    parser.add_argument("--base_dir", type=str, default=".",
                       help="Base directory of TCLP pipeline")
    parser.add_argument("--step", type=float, default=0.05,
                       help="Weight grid step size")
    parser.add_argument("--tau", type=float, default=10.0,
                       help="POD threshold for AUC calculation")
    parser.add_argument("--output", type=str, default=None,
                       help="Output CSV for grid search results")
    args = parser.parse_args()
    
    scores, toxref, bridge = load_data(args.base_dir)
    merged = prepare_merged_data(scores, toxref, bridge)
    
    results_df, best_rho, best_auc, best_combined = grid_search_weights(
        merged, step=args.step, tau=args.tau)
    
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\nGrid search results saved to: {args.output}")
    
    # Show top 10 by combined score
    results_df["combined"] = 0.5 * results_df["spearman_rho"] + 0.5 * results_df["auc"]
    top10 = results_df.nlargest(10, "combined")
    print("\n=== Top 10 Weight Combinations ===")
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()

