import os
import ast
import shutil
import pandas as pd


IN_FP = "processed_final8k213_tcpl_labeled_final_with_proposed.csv"


def backup(fp: str) -> str:
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    bkdir = f"backups_{ts}"
    os.makedirs(bkdir, exist_ok=True)
    shutil.copy2(fp, os.path.join(bkdir, os.path.basename(fp)))
    return bkdir


def mech_has_high(mech_raw: str, thresh: float) -> bool:
    try:
        d = ast.literal_eval(mech_raw) if isinstance(mech_raw, str) else None
        if isinstance(d, dict):
            for info in d.values():
                try:
                    if float(info.get("score", 0.0)) >= thresh:
                        return True
                except Exception:
                    continue
    except Exception:
        pass
    return False


def main():
    if not os.path.exists(IN_FP):
        raise FileNotFoundError(IN_FP)
    df = pd.read_csv(IN_FP)

    # thresholds (can be overridden by env)
    t_low = float(os.environ.get("T_LOW", 0.20))          # general low gate
    t_high = float(os.environ.get("T_HIGH", 0.40))        # high gate
    # when y_class==1, require stricter low gate for assigning 0
    t_low_pos = float(os.environ.get("T_LOW_POS", 0.15))  # stricter low for positives

    # Build strict labels from scratch (no monotonic constraint on y_class)
    b_strict = []
    t_strict = []

    for _, row in df.iterrows():
        y = row.get("y")
        sc = row.get("S_c")
        ci_l = row.get("ci_lower")
        ci_u = row.get("ci_upper")
        mech_raw = row.get("mechanism_details")

        # Force 0 when y==0 (original target) OR y_class==0
        if (pd.notna(y) and y == 0) or (pd.notna(row.get("y_class")) and int(row.get("y_class")) == 0):
            t_label = 0
            b_label = 0
        else:
            # High if confident high
            if pd.notna(ci_l) and ci_l >= t_high:
                t_label = 2
            else:
                # Strict zero gate if confident low and no mechanism above t_low
                yc = row.get("y_class")
                # choose stricter low threshold for y_class==1
                low_thr = t_low_pos if pd.notna(yc) and int(yc) == 1 else t_low
                zero_conf = (pd.notna(ci_u) and ci_u <= low_thr) and (pd.notna(sc) and sc <= low_thr)
                no_mech_high = not mech_has_high(mech_raw, low_thr)
                if zero_conf and no_mech_high:
                    t_label = 0
                else:
                    t_label = 1

            b_label = 0 if t_label == 0 else 1

        b_strict.append(b_label)
        t_strict.append(t_label)

    df["tcpl_binary_final_strict"] = pd.Series(b_strict, dtype="Int64")
    df["tcpl_ternary_final_strict"] = pd.Series(t_strict, dtype="Int64")

    bkdir = backup(IN_FP)
    df.to_csv(IN_FP, index=False)
    print(f"✔ Wrote strict final labels to {IN_FP} (backup in {bkdir})")

    # print distributions
    for col in ["tcpl_binary_final_strict", "tcpl_ternary_final_strict"]:
        vc = df[col].value_counts(dropna=False).sort_index()
        print(col, "counts:\n", vc)


if __name__ == "__main__":
    main()
