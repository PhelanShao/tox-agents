#!/usr/bin/env python3
"""
Generate ToxRefDB-aligned external validation metrics and figures for TCLP pipeline.

Outputs (created under ../validation/):
 - si_table_S16_toxrefdb_alignment_metrics.csv  (Table S16)
 - Figure_S17_Sc_vs_neglog10POD.png            (Scatter + monotone smoothing)
 - Figure_S18_ROC_thresholds_3_10_30.png       (ROC curves at τ={3,10,30})

Inputs (relative to repository):
 - Unimol/TCLP_pipeline/source/tox21_toxrefdb_matched_via_cas.csv
     Columns: CAS_NORM, POD_MGKGDAY, PUBCHEM_CID, S_global, S_potency
 - Unimol/TCLP_pipeline/processed/chemical_bridge_table.csv
     Columns: chid, casn, dsstox_substance_id, casn_normalized, PUBCHEM_CID
 - Unimol/TCLP_pipeline/processed/tcpl_chemical_scores_final.csv
     Columns include: chid, S_c

Method notes:
 - For chemicals with multiple ToxRefDB entries, POD per chemical is aggregated as min(POD).
 - S_global/S_potency per chemical are averaged if duplicates exist.
 - S_c is mapped via CAS to chid and aggregated by max(S_c) per CAS.
 - Spearman correlation is computed between score and -log10(POD_min).
 - ROC-AUC is computed for τ ∈ {3,10,30} mg/kg/day (label: POD ≤ τ as positive).
 - If a score yields AUC < 0.5 for a given τ, scores are flipped (1 - score) for that τ.
 - 95% CIs are estimated via bootstrap (percentile intervals).
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _bootstrap_ci(values: List[float], alpha: float = 0.05) -> Tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    lo = np.percentile(values, 100 * (alpha / 2))
    hi = np.percentile(values, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def compute_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y, nan_policy="omit")
        return float(r), float(p)
    except Exception:
        # Fallback: manual rank correlation without p-value
        xrank = pd.Series(x).rank()
        yrank = pd.Series(y).rank()
        r = float(pd.Series(xrank).corr(pd.Series(yrank), method="pearson"))
        return r, float("nan")


def bootstrap_stat(
    func,
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]
        try:
            stats.append(func(xb, yb))
        except Exception:
            continue
    return stats


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        # Simple trapezoidal ROC AUC fallback (approx via sorting)
        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]
        P = float(y_true_sorted.sum())
        N = float(len(y_true_sorted) - P)
        if P == 0 or N == 0:
            return float("nan")
        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)
        tpr = tps / P
        fpr = fps / N
        return float(np.trapz(tpr, fpr))


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray):
    try:
        from sklearn.metrics import roc_curve
        return roc_curve(y_true, y_score)
    except Exception:
        # Minimal fallback: compute unique thresholds
        thresholds = np.unique(y_score)[::-1]
        P = float(y_true.sum())
        N = float(len(y_true) - P)
        tpr_list, fpr_list = [0.0], [0.0]
        for thr in thresholds:
            y_pred = (y_score >= thr).astype(int)
            tp = float(((y_pred == 1) & (y_true == 1)).sum())
            fp = float(((y_pred == 1) & (y_true == 0)).sum())
            tpr_list.append(0.0 if P == 0 else tp / P)
            fpr_list.append(0.0 if N == 0 else fp / N)
        tpr_list.append(1.0)
        fpr_list.append(1.0)
        return np.array(fpr_list), np.array(tpr_list), thresholds


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_and_aggregate_sources(base_dir: str) -> pd.DataFrame:
    toxref_path = os.path.join(base_dir, "source", "tox21_toxrefdb_matched_via_cas.csv")
    bridge_path = os.path.join(base_dir, "processed", "chemical_bridge_table.csv")
    scores_path = os.path.join(base_dir, "processed", "tcpl_chemical_scores_final.csv")

    tox = pd.read_csv(toxref_path)
    tox = tox.rename(columns={"CAS_NORM": "casn_normalized"})
    tox = tox[(tox["POD_MGKGDAY"].notna()) & (tox["POD_MGKGDAY"] > 0)].copy()

    # Aggregate ToxRef per CAS: POD_min, and mean S_global/S_potency
    agg_funcs = {
        "POD_MGKGDAY": "min",
        "S_global": "mean",
        "S_potency": "mean",
        "PUBCHEM_CID": "first",
    }
    tox_chem = tox.groupby("casn_normalized", as_index=False).agg(agg_funcs)

    # Map CAS -> chid via bridge, then map to S_c
    bridge = pd.read_csv(bridge_path)
    bridge = bridge.drop_duplicates(subset=["casn_normalized", "chid"])  # avoid dup pairs
    scores = pd.read_csv(scores_path, usecols=["chid", "S_c"])  # chemical-level score
    cs = pd.merge(bridge[["casn_normalized", "chid"]], scores, on="chid", how="left")
    # Aggregate S_c per CAS: use max to be conservative
    cs_chem = cs.groupby("casn_normalized", as_index=False)["S_c"].max()

    merged = pd.merge(tox_chem, cs_chem, on="casn_normalized", how="left")
    merged = merged.rename(columns={"POD_MGKGDAY": "POD", "S_c": "S_c"})
    merged["neglog10_POD"] = -np.log10(merged["POD"].astype(float))
    return merged


def analyze_and_save(df: pd.DataFrame, out_dir: str, seed: int = 42):
    ensure_dir(out_dir)

    # Prepare metrics
    scores = {
        "S_c": df["S_c"].values.astype(float),
        "S_global": df["S_global"].values.astype(float),
        "S_potency": df["S_potency"].values.astype(float),
    }
    y_cont = df["neglog10_POD"].values.astype(float)
    pod = df["POD"].values.astype(float)

    taus = [3.0, 10.0, 30.0]

    records: List[Dict] = []
    for name, x in scores.items():
        mask = np.isfinite(x) & np.isfinite(y_cont)
        x_valid = x[mask]
        y_valid = y_cont[mask]
        pod_valid = pod[mask]
        n = int(len(x_valid))
        if n == 0:
            continue

        # Spearman
        rho, pval = compute_spearman(x_valid, y_valid)
        boot_rhos = bootstrap_stat(
            lambda xb, yb: compute_spearman(xb, yb)[0], x_valid, y_valid, seed=seed
        )
        rho_lo, rho_hi = _bootstrap_ci(boot_rhos)

        # AUCs per τ
        aucs = {}
        auc_ci = {}
        flipped = {}
        for tau in taus:
            y_true = (pod_valid <= tau).astype(int)
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                auc = float("nan")
                lo, hi = float("nan"), float("nan")
                did_flip = False
            else:
                auc = roc_auc(y_true, x_valid)
                did_flip = False
                if math.isfinite(auc) and auc < 0.5:
                    # Flip orientation
                    did_flip = True
                    x_used = 1.0 - x_valid
                    auc = roc_auc(y_true, x_used)
                    # For CI, use flipped scores as well
                    boot_aucs = bootstrap_stat(
                        lambda xb, yb: roc_auc((yb <= tau).astype(int), 1.0 - xb),
                        x_valid,
                        pod_valid,
                        seed=seed,
                    )
                else:
                    boot_aucs = bootstrap_stat(
                        lambda xb, yb: roc_auc((yb <= tau).astype(int), xb),
                        x_valid,
                        pod_valid,
                        seed=seed,
                    )
                lo, hi = _bootstrap_ci(boot_aucs)
            aucs[tau] = auc
            auc_ci[tau] = (lo, hi)
            flipped[tau] = did_flip

        rec = {
            "Score": name,
            "N": n,
            "Spearman_rho": rho,
            "Spearman_p": pval,
            "Spearman_CI_lower": rho_lo,
            "Spearman_CI_upper": rho_hi,
        }
        for tau in taus:
            rec[f"AUC_tau_{int(tau)}"] = aucs[tau]
            rec[f"AUC_tau_{int(tau)}_CI_lower"] = auc_ci[tau][0]
            rec[f"AUC_tau_{int(tau)}_CI_upper"] = auc_ci[tau][1]
            rec[f"AUC_tau_{int(tau)}_flipped"] = int(flipped[tau])
        records.append(rec)

    out_csv = os.path.join(out_dir, "si_table_S16_toxrefdb_alignment_metrics.csv")
    pd.DataFrame.from_records(records).to_csv(out_csv, index=False)

    # Figures
    try:
        import matplotlib.pyplot as plt
        from sklearn.isotonic import IsotonicRegression

        # Figure S17: Sc vs -log10(POD)
        name = "S_c"
        x = scores[name]
        mask = np.isfinite(x) & np.isfinite(y_cont)
        x_plot = x[mask]
        y_plot = y_cont[mask]
        n = len(x_plot)
        fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=200)
        ax.scatter(x_plot, y_plot, s=12, alpha=0.35, edgecolor="none", label=f"n={n}")
        # Isotonic monotone fit
        try:
            iso = IsotonicRegression(y_min=min(y_plot), y_max=max(y_plot), increasing=True)
            xi = np.linspace(np.nanmin(x_plot), np.nanmax(x_plot), 200)
            yi = iso.fit_transform(x_plot, y_plot)
            yi_sorted = iso.transform(xi)
            ax.plot(xi, yi_sorted, color="crimson", lw=2, label="Monotone fit (isotonic)")
        except Exception:
            pass
        ax.set_xlabel("S_c (tcpl-compliant chemical score)")
        ax.set_ylabel("-log10(POD) [mg·kg⁻¹·day⁻¹]")
        ax.set_title("S_c vs -log10(POD)")
        ax.grid(True, ls=":", lw=0.5, alpha=0.6)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig_path = os.path.join(out_dir, "Figure_S17_Sc_vs_neglog10POD.png")
        fig.savefig(fig_path)
        plt.close(fig)

        # Figure S18: ROC curves at τ={3,10,30} for Sc
        taus = [3.0, 10.0, 30.0]
        fig2, ax2 = plt.subplots(figsize=(5.5, 4.2), dpi=200)
        for tau, color in zip(taus, ["tab:blue", "tab:orange", "tab:green"]):
            y_true = (pod[mask] <= tau).astype(int)
            x_used = x_plot.copy()
            auc = roc_auc(y_true, x_used)
            if math.isfinite(auc) and auc < 0.5:
                x_used = 1.0 - x_used
                auc = roc_auc(y_true, x_used)
            fpr, tpr, _ = roc_curve_points(y_true, x_used)
            # Bootstrap CI for AUC
            boot_aucs = bootstrap_stat(lambda xb, yb: roc_auc((yb <= tau).astype(int), xb), x_used, pod[mask], seed=seed)
            lo, hi = _bootstrap_ci(boot_aucs)
            ax2.plot(fpr, tpr, color=color, lw=1.8, label=f"τ={int(tau)} mg/kg/day | AUC={auc:.3f} [{lo:.3f},{hi:.3f}]")
        ax2.plot([0, 1], [0, 1], "k--", lw=1)
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curves at POD thresholds")
        ax2.grid(True, ls=":", lw=0.5, alpha=0.6)
        ax2.legend(loc="lower right", frameon=False)
        fig2.tight_layout()
        fig2_path = os.path.join(out_dir, "Figure_S18_ROC_thresholds_3_10_30.png")
        fig2.savefig(fig2_path)
        plt.close(fig2)
    except Exception as e:
        # Matplotlib/sklearn not available; skip figures
        print(f"[WARN] Skipped figure generation: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate ToxRefDB SI metrics and figures")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=os.path.join("Unimol", "TCLP_pipeline"),
        help="Base directory of TCLP pipeline",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("Unimol", "TCLP_pipeline", "validation"),
        help="Output directory for validation results",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_and_aggregate_sources(args.base_dir)
    ensure_dir(args.out_dir)
    analyze_and_save(df, args.out_dir, seed=args.seed)
    print(f"Saved Table S16 and figures under: {args.out_dir}")


if __name__ == "__main__":
    main()
