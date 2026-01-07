import os
import shutil
import pandas as pd


IN_FP = "processed_final8k213_tcpl_labeled_final_with_proposed.csv"


def backup(fp: str) -> str:
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    bkdir = f"backups_{ts}"
    os.makedirs(bkdir, exist_ok=True)
    shutil.copy2(fp, os.path.join(bkdir, os.path.basename(fp)))
    return bkdir


def choose_first_non_null(row: pd.Series, cols: list, treat_neg1_as_nan: bool = False):
    for c in cols:
        if c not in row.index:
            continue
        v = row[c]
        if pd.isna(v):
            continue
        if treat_neg1_as_nan and v == -1:
            continue
        return int(v), c
    return None, None


def main():
    if not os.path.exists(IN_FP):
        raise FileNotFoundError(IN_FP)
    df = pd.read_csv(IN_FP)

    # Preferences: CI > quantile > agg > orig
    bin_candidates = [
        "tcpl_binary_proposed_ci",
        "tcpl_binary_proposed_quantile",
        "tcpl_binary_agg",
        "tcpl_binary_orig",
    ]
    ter_candidates = [
        "tcpl_ternary_proposed_ci",
        "tcpl_ternary_proposed_quantile",
        "tcpl_ternary_agg",
        "tcpl_ternary_orig",
    ]

    final_bin = []
    final_ter = []
    src_bin = []
    src_ter = []

    for _, row in df.iterrows():
        y = row.get("y")
        forced = False
        if pd.notna(y) and y == 0:
            final_bin.append(0)
            final_ter.append(0)
            src_bin.append("forced_by_y0")
            src_ter.append("forced_by_y0")
            continue

        # Otherwise, prefer CI -> quantile -> agg -> orig
        b_val, b_src = choose_first_non_null(row, bin_candidates, treat_neg1_as_nan=True)
        t_val, t_src = choose_first_non_null(row, ter_candidates, treat_neg1_as_nan=True)

        final_bin.append(b_val if b_val is not None else 0)
        final_ter.append(t_val if t_val is not None else 1)
        src_bin.append(b_src or "fallback_default")
        src_ter.append(t_src or "fallback_default")

    df["tcpl_binary_final"] = pd.Series(final_bin, dtype="Int64")
    df["tcpl_ternary_final"] = pd.Series(final_ter, dtype="Int64")
    df["tcpl_binary_final_source"] = src_bin
    df["tcpl_ternary_final_source"] = src_ter

    bkdir = backup(IN_FP)
    df.to_csv(IN_FP, index=False)
    print(f"✔ Updated final labels in {IN_FP} (backup in {bkdir})")

    # Print simple distributions
    for col in ["tcpl_binary_final", "tcpl_ternary_final"]:
        vc = df[col].value_counts(dropna=False).sort_index()
        print(col, "counts:\n", vc)


if __name__ == "__main__":
    main()

