import csv
import json
from pathlib import Path
from collections import OrderedDict


def load_json(p: Path):
    return json.loads(p.read_text())


def extract_multi_experiments(detail_list: list):
    exps = {}
    for entry in detail_list:
        if not isinstance(entry, dict):
            continue
        name = entry.get("experiment_name")
        if not name:
            continue
        detailed = entry.get("detailed_results", {})
        cls = detailed.get("classification_endpoints", {}) or {}
        reg = detailed.get("regression_endpoints", {}) or {}
        exps[name] = {
            "classification": {k: (v.get("auc") if isinstance(v, dict) else None) for k, v in cls.items()},
            "regression": {k: (v.get("r2") if isinstance(v, dict) else None) for k, v in reg.items()},
            "avg_auc": entry.get("summary", {}).get("classification", {}).get("avg_auc"),
            "avg_r2": entry.get("summary", {}).get("regression", {}).get("avg_r2"),
        }
    return exps


def compute_ablation_sensitivity(exps, baseline_name="toxd4c_ablation_full_model"):
    if baseline_name not in exps:
        raise RuntimeError(f"Baseline experiment '{baseline_name}' not found in details JSON.")
    base = exps[baseline_name]
    out = OrderedDict()
    for name, e in exps.items():
        if name == baseline_name:
            continue
        # Per-endpoint classification deltas
        delta_cls = {}
        for ep, v in e["classification"].items():
            bv = base["classification"].get(ep)
            if v is not None and bv is not None:
                delta_cls[ep] = v - bv
        # Per-endpoint regression deltas
        delta_reg = {}
        for ep, v in e["regression"].items():
            bv = base["regression"].get(ep)
            if v is not None and bv is not None:
                delta_reg[ep] = v - bv
        avg_da = (sum(delta_cls.values()) / len(delta_cls)) if delta_cls else None
        avg_dr = (sum(delta_reg.values()) / len(delta_reg)) if delta_reg else None
        out[name] = {
            "avg_delta_auc": avg_da,
            "avg_delta_r2": avg_dr,
            "n_cls": len(delta_cls),
            "n_reg": len(delta_reg),
            "improved_cls": sum(1 for x in delta_cls.values() if x > 0),
            "degraded_cls": sum(1 for x in delta_cls.values() if x < 0),
            "improved_reg": sum(1 for x in delta_reg.values() if x > 0),
            "degraded_reg": sum(1 for x in delta_reg.values() if x < 0),
            "per_endpoint_cls": OrderedDict(sorted(delta_cls.items(), key=lambda kv: kv[0])),
            "per_endpoint_reg": OrderedDict(sorted(delta_reg.items(), key=lambda kv: kv[0])),
        }
    return out


def extract_single_stats(comp: dict):
    singles = {}
    stats = comp.get("statistical_summary", {})
    for key, val in stats.items():
        desc = val.get("description", "")
        if key.startswith("single_cls_"):
            ep = desc.replace("Single classification: ", "").strip()
            auc = val.get("statistics", {}).get("avg_auc", {}).get("mean")
            if auc is not None:
                singles[ep] = {"type": "classification", "value": float(auc)}
        elif key.startswith("single_reg_"):
            ep = desc.replace("Single regression: ", "").strip()
            r2 = val.get("statistics", {}).get("avg_r2", {}).get("mean")
            if r2 is not None:
                singles[ep] = {"type": "regression", "value": float(r2)}
    return singles


def compare_single_vs_multi(singles, multi_full, multi_cls_only=None):
    rows = []
    for ep, sv in singles.items():
        typ = sv["type"]
        if typ == "classification":
            mf = multi_full["classification"].get(ep)
            mco = multi_cls_only["classification"].get(ep) if multi_cls_only else None
            rows.append({
                "endpoint": ep,
                "type": typ,
                "single": sv["value"],
                "multi_full": mf,
                "delta_full": (mf - sv["value"]) if (mf is not None) else None,
                "multi_cls_only": mco,
                "delta_cls_only": (mco - sv["value"]) if (mco is not None) else None,
            })
        else:
            mf = multi_full["regression"].get(ep)
            rows.append({
                "endpoint": ep,
                "type": typ,
                "single": sv["value"],
                "multi_full": mf,
                "delta_full": (mf - sv["value"]) if (mf is not None) else None,
                "multi_cls_only": None,
                "delta_cls_only": None,
            })
    return rows


def write_csv(path: Path, rows: list, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def main():
    out_dir = Path("ToxD4C/r1c3_sensitivity_results")
    comp_path = out_dir / "comprehensive_sensitivity_analysis.json"
    detail_path = Path("ToxD4C/detailed_endpoint_results_all_experiments.json")

    comp = load_json(comp_path)
    details = load_json(detail_path)
    exps = extract_multi_experiments(details)

    # Build ablation sensitivity table (Table S4)
    sens = compute_ablation_sensitivity(exps, baseline_name="toxd4c_ablation_full_model")
    summary_rows = []
    per_endpoint_rows = []
    for name, v in sens.items():
        summary_rows.append({
            "experiment": name,
            "avg_delta_auc": v.get("avg_delta_auc"),
            "avg_delta_r2": v.get("avg_delta_r2"),
            "n_cls": v.get("n_cls"),
            "n_reg": v.get("n_reg"),
            "improved_cls": v.get("improved_cls"),
            "degraded_cls": v.get("degraded_cls"),
            "improved_reg": v.get("improved_reg"),
            "degraded_reg": v.get("degraded_reg"),
        })
        for ep, d in v.get("per_endpoint_cls", {}).items():
            per_endpoint_rows.append({
                "experiment": name,
                "endpoint": ep,
                "type": "classification",
                "delta": d,
            })
        for ep, d in v.get("per_endpoint_reg", {}).items():
            per_endpoint_rows.append({
                "experiment": name,
                "endpoint": ep,
                "type": "regression",
                "delta": d,
            })

    write_csv(out_dir / "Table_S4_sensitivity_analysis.csv", summary_rows,
              ["experiment", "avg_delta_auc", "avg_delta_r2", "n_cls", "n_reg", "improved_cls", "degraded_cls", "improved_reg", "degraded_reg"]) 
    write_csv(out_dir / "Table_S4_per_endpoint_deltas.csv", per_endpoint_rows,
              ["experiment", "endpoint", "type", "delta"]) 

    # Single vs multi comparison
    singles = extract_single_stats(comp)
    full_name = "toxd4c_ablation_full_model"
    cls_only_name = "toxd4c_ablation_classification_only"
    if full_name in exps:
        multi_full = exps[full_name]
        multi_cls_only = exps.get(cls_only_name)
        svm_rows = compare_single_vs_multi(singles, multi_full, multi_cls_only)
        write_csv(out_dir / "single_vs_multi_comparison.csv", svm_rows,
                  ["endpoint", "type", "single", "multi_full", "delta_full", "multi_cls_only", "delta_cls_only"]) 


if __name__ == "__main__":
    main()

