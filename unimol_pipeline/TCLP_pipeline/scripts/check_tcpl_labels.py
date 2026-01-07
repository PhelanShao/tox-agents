import os
from typing import Tuple

import pandas as pd


def _md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "\n" + df.to_string(index=False) + "\n"


def series_counts(s: pd.Series) -> pd.DataFrame:
    vc = s.value_counts(dropna=False).sort_index()
    total = int(vc.sum())
    out = pd.DataFrame({"class": vc.index, "count": vc.values})
    out["pct"] = (out["count"] / total * 100.0).round(2)
    return out


def relabel_ci(df: pd.DataFrame, t_low: float, t_high: float) -> Tuple[pd.Series, pd.Series]:
    # Ternary classification (CI method): upper_bound <= t_low → low(0); lower_bound >= t_high → high(2); others → medium(1)
    def tern_ci(row):
        if row["ci_upper"] <= t_low:
            return 0
        if row["ci_lower"] >= t_high:
            return 2
        return 1

    tern = df.apply(tern_ci, axis=1)
    # Binary classification threshold consistent with ternary classification: use midpoint
    t_star = (t_low + t_high) / 2.0
    bina = (df["S_c"] >= t_star).astype(int)
    return bina, tern


def relabel_quantile(df: pd.DataFrame, q_low: float = 1 / 3, q_high: float = 2 / 3) -> Tuple[pd.Series, pd.Series, float, float]:
    # Quantile method: automatically determine low/medium/high thresholds based on distribution
    t_low = float(df["S_c"].quantile(q_low))
    t_high = float(df["S_c"].quantile(q_high))

    def tern_q(s):
        if s < t_low:
            return 0
        if s > t_high:
            return 2
        return 1

    tern = df["S_c"].apply(tern_q)
    t_star = (t_low + t_high) / 2.0
    bina = (df["S_c"] >= t_star).astype(int)
    return bina, tern, t_low, t_high


def main():
    scores_fp = os.path.join("processed", "tcpl_chemical_scores_final.csv")
    bridge_fp = os.path.join("processed", "chemical_bridge_table.csv")
    main_fp = "processed_final8k213_tcpl_labeled_final.csv"

    if not os.path.exists(scores_fp):
        raise FileNotFoundError(scores_fp)
    scores = pd.read_csv(scores_fp)
    print("Reading chemical-level scores:", scores_fp, "— rows:", len(scores))

    # Original label distribution
    bin_now = series_counts(scores["tcpl_binary_compliant"].astype(int))
    ter_now = series_counts(scores["tcpl_ternary_compliant"].astype(int))

    # Proposal 1: CI method (adjustable)
    t_low_ci, t_high_ci = 0.20, 0.40
    b_ci, t_ci = relabel_ci(scores, t_low_ci, t_high_ci)

    # Proposal 2: Quantile method
    b_q, t_q, t_low_q, t_high_q = relabel_quantile(scores, 1 / 3, 2 / 3)

    # Write proposed columns
    scores["tcpl_binary_proposed_ci"] = b_ci
    scores["tcpl_ternary_proposed_ci"] = t_ci
    scores["tcpl_binary_proposed_quantile"] = b_q
    scores["tcpl_ternary_proposed_quantile"] = t_q

    out_scores_fp = os.path.join("processed", "tcpl_chemical_scores_with_proposed.csv")
    scores.to_csv(out_scores_fp, index=False)
    print("Written chemical-level table with proposed labels:", out_scores_fp)

    # Generate report
    os.makedirs("reports", exist_ok=True)
    rep = os.path.join("reports", "tcpl_label_check_report.md")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("# TCPL Label Check and Relabeling Proposal\n\n")
        f.write(f"Data file: `{scores_fp}`  ")
        f.write(f"\nTotal chemicals: {len(scores)}\n\n")

        f.write("## Original Label Distribution (Chemical Level)\n\n")
        f.write("### Binary Classification tcpl_binary_compliant\n\n")
        f.write(_md(bin_now) + "\n\n")
        f.write("### Ternary Classification tcpl_ternary_compliant\n\n")
        f.write(_md(ter_now) + "\n\n")

        f.write("## Proposed Thresholds and Distribution\n\n")
        f.write(
            f"- CI method thresholds: t_low={t_low_ci:.2f}, t_high={t_high_ci:.2f} (low class: upper_CI≤t_low; high class: lower_CI≥t_high; others: medium class)\n"
        )
        f.write(
            f"- Quantile method thresholds: q1≈33%→t_low={t_low_q:.3f}, q2≈67%→t_high={t_high_q:.3f}\n\n"
        )

        def counts_md(title: str, s: pd.Series):
            cnt = series_counts(s)
            f.write(f"### {title}\n\n")
            f.write(_md(cnt) + "\n\n")

        counts_md("Binary Classification (CI Method Proposal)", scores["tcpl_binary_proposed_ci"])
        counts_md("Ternary Classification (CI Method Proposal)", scores["tcpl_ternary_proposed_ci"])
        counts_md("Binary Classification (Quantile Method Proposal)", scores["tcpl_binary_proposed_quantile"])
        counts_md("Ternary Classification (Quantile Method Proposal)", scores["tcpl_ternary_proposed_quantile"])

        # Consistency cross-tabulation
        f.write("## Consistency Check\n\n")
        f.write("### Original Binary vs Original Ternary\n\n")
        f.write(
            _md(pd.crosstab(scores["tcpl_binary_compliant"], scores["tcpl_ternary_compliant"]).reset_index())
            + "\n\n"
        )
        f.write("### Original Ternary vs Proposed Ternary (CI Method)\n\n")
        f.write(
            _md(pd.crosstab(scores["tcpl_ternary_compliant"], scores["tcpl_ternary_proposed_ci"]).reset_index())
            + "\n\n"
        )
        f.write("### Original Ternary vs Proposed Ternary (Quantile Method)\n\n")
        f.write(
            _md(
                pd.crosstab(
                    scores["tcpl_ternary_compliant"],
                    scores["tcpl_ternary_proposed_quantile"],
                ).reset_index()
            )
            + "\n\n"
        )

        f.write("## Should Binary Classification Be Modified?\n\n")
        f.write("- Recommendation: maintain a consistent threshold system with ternary classification:\n")
        f.write(
            "  - If using CI method: binary threshold t* = (t_low + t_high)/2, corresponding column `tcpl_binary_proposed_ci`.\n"
        )
        f.write(
            "  - If using quantile method: binary threshold t* = (t_low + t_high)/2, corresponding column `tcpl_binary_proposed_quantile`.\n"
        )
        f.write(
            "- If only ternary is modified without changing binary, inconsistency may occur (e.g., binary=1 but ternary=0). Recommend synchronous update to maintain consistency.\n\n"
        )

    print("Report generated:", rep)

    # Optional: sample-level mapping (for subsequent Sankey diagram)
    if os.path.exists(bridge_fp) and os.path.exists(main_fp):
        try:
            bridge = pd.read_csv(bridge_fp)
            main = pd.read_csv(main_fp)
            bc = {c.lower(): c for c in bridge.columns}
            chid_col = bc.get("chid")
            cid_col = bc.get("pubchem_cid")
            # Attempt to fill bridge CID using CAS→CID mapping
            try:
                cas_map_fp = os.path.join("processed", "cas_pubchem_mapping.json")
                if os.path.exists(cas_map_fp):
                    import json
                    with open(cas_map_fp, "r", encoding="utf-8") as jf:
                        cas_map = json.load(jf).get("results", {})
                    # Prefer casn_normalized, then casn
                    casn_norm = bc.get("casn_normalized")
                    casn = bc.get("casn")
                    if cid_col is None and "PUBCHEM_CID" in bridge.columns:
                        cid_col = "PUBCHEM_CID"
                    if cid_col is None:
                        # Create new CID column
                        bridge["PUBCHEM_CID"] = pd.NA
                        cid_col = "PUBCHEM_CID"
                    def fill_cid(row):
                        cid = row[cid_col]
                        if pd.isna(cid):
                            for cas_col in [casn_norm, casn]:
                                if cas_col and pd.notna(row.get(cas_col)):
                                    cid_try = cas_map.get(str(row[cas_col]).strip())
                                    if cid_try is not None:
                                        return cid_try
                        return cid
                    bridge[cid_col] = bridge.apply(fill_cid, axis=1)
            except Exception as _:
                pass

            if (
                chid_col is not None
                and cid_col is not None
                and "PUBCHEM_CID" in main.columns
            ):
                ss = scores.copy()
                ss["chid"] = pd.to_numeric(ss["chid"], errors="coerce")
                bridge[chid_col] = pd.to_numeric(bridge[chid_col], errors="coerce")
                merged = (
                    ss.merge(
                        bridge[[chid_col, cid_col]],
                        left_on="chid",
                        right_on=chid_col,
                        how="left",
                    ).merge(main, left_on=cid_col, right_on="PUBCHEM_CID", how="right")
                )
                out_main = "processed_final8k213_tcpl_labeled_final_with_proposed.csv"
                merged.to_csv(out_main, index=False)
                print("Written sample-level merged table:", out_main)
            else:
                print("[Info] Cannot map: bridge missing chid/PUBCHEM_CID or main table missing PUBCHEM_CID.")
        except Exception as e:
            print("[Info] Sample-level merge failed:", e)
    else:
        print("[Info] Bridge or main table not found, skipping sample-level merge.")


if __name__ == "__main__":
    main()
