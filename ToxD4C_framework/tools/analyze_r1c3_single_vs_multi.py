import json
from pathlib import Path
from collections import defaultdict, OrderedDict


def load_json(p: Path):
    return json.loads(p.read_text())


def extract_single_stats(comp: dict):
    singles = {}
    stats = comp.get("statistical_summary", {})
    for key, val in stats.items():
        desc = val.get("description", "")
        if key.startswith("single_cls_"):
            ep = desc.replace("Single classification: ", "").strip()
            auc = val.get("statistics", {}).get("avg_auc", {}).get("mean")
            if auc is not None:
                singles[ep] = {"type": "classification", "metric": "auc", "value": float(auc)}
        elif key.startswith("single_reg_"):
            ep = desc.replace("Single regression: ", "").strip()
            r2 = val.get("statistics", {}).get("avg_r2", {}).get("mean")
            if r2 is not None:
                singles[ep] = {"type": "regression", "metric": "r2", "value": float(r2)}
    return singles


def extract_multi_experiments(detail_list: list):
    # Map experiment_name -> per-endpoint metrics
    exps = {}
    for entry in detail_list:
        name = entry.get("experiment_name")
        if not name:
            continue
        mode = entry.get("task_mode")
        # Some files nest detailed results under 'detailed_results'
        detailed = entry.get("detailed_results", {}) if isinstance(entry, dict) else {}
        cls = detailed.get("classification_endpoints", {})
        reg = detailed.get("regression_endpoints", {})
        exps[name] = {
            "task_mode": mode,
            "classification": {k: (v.get("auc") if isinstance(v, dict) else None) for k, v in cls.items()},
            "regression": {k: (v.get("r2") if isinstance(v, dict) else None) for k, v in reg.items()},
            "avg_auc": entry.get("summary", {}).get("classification", {}).get("avg_auc"),
            "avg_r2": entry.get("summary", {}).get("regression", {}).get("avg_r2"),
        }
    return exps


def compare_single_vs_multi(singles, multi_full, multi_cls_only=None):
    rows = []
    for ep, sv in singles.items():
        if sv["type"] == "classification":
            mv = multi_full["classification"].get(ep)
            mv_cls_only = multi_cls_only["classification"].get(ep) if multi_cls_only else None
            rows.append({
                "endpoint": ep,
                "type": sv["type"],
                "single": sv["value"],
                "multi_full": mv,
                "multi_cls_only": mv_cls_only,
                "delta_multi_full": (mv - sv["value"]) if (mv is not None) else None,
                "delta_multi_cls_only": (mv_cls_only - sv["value"]) if (mv_cls_only is not None) else None,
            })
        else:
            mv = multi_full["regression"].get(ep)
            rows.append({
                "endpoint": ep,
                "type": sv["type"],
                "single": sv["value"],
                "multi_full": mv,
                "multi_cls_only": None,
                "delta_multi_full": (mv - sv["value"]) if (mv is not None) else None,
                "delta_multi_cls_only": None,
            })
    return rows


def ablation_sensitivity(exps, baseline_name="toxd4c_ablation_full_model"):
    # Compare every experiment to baseline per endpoint
    if baseline_name not in exps:
        return {}
    base = exps[baseline_name]
    results = {}
    for name, e in exps.items():
        if name == baseline_name:
            continue
        delta_cls = {}
        for ep, v in e["classification"].items():
            bv = base["classification"].get(ep)
            if bv is not None and v is not None:
                delta_cls[ep] = v - bv
        delta_reg = {}
        for ep, v in e["regression"].items():
            bv = base["regression"].get(ep)
            if bv is not None and v is not None:
                delta_reg[ep] = v - bv
        if delta_cls or delta_reg:
            results[name] = {
                "avg_delta_auc": sum(delta_cls.values()) / len(delta_cls) if delta_cls else None,
                "avg_delta_r2": sum(delta_reg.values()) / len(delta_reg) if delta_reg else None,
                "n_cls": len(delta_cls),
                "n_reg": len(delta_reg),
                "improved_cls": sum(1 for x in delta_cls.values() if x > 0),
                "degraded_cls": sum(1 for x in delta_cls.values() if x < 0),
                "improved_reg": sum(1 for x in delta_reg.values() if x > 0),
                "degraded_reg": sum(1 for x in delta_reg.values() if x < 0),
                "per_endpoint_cls": OrderedDict(sorted(delta_cls.items(), key=lambda kv: kv[1])) if delta_cls else {},
                "per_endpoint_reg": OrderedDict(sorted(delta_reg.items(), key=lambda kv: kv[1])) if delta_reg else {},
            }
    return results


def main():
    comp_path = Path("ToxD4C/r1c3_sensitivity_results/comprehensive_sensitivity_analysis.json")
    detail_path = Path("ToxD4C/detailed_endpoint_results_all_experiments.json")
    comp = load_json(comp_path)
    details = load_json(detail_path)

    singles = extract_single_stats(comp)
    exps = extract_multi_experiments(details)

    # Identify key multi experiments
    full_name = "toxd4c_ablation_full_model"
    cls_only = "toxd4c_ablation_classification_only"
    if full_name not in exps:
        print("[WARN] Baseline multi full model not found.")
        return
    multi_full = exps[full_name]
    multi_cls_only = exps.get(cls_only)

    # Compare single vs multi
    comp_rows = compare_single_vs_multi(singles, multi_full, multi_cls_only)

    # Print summary
    print("Single vs Multi (selected endpoints):")
    print("endpoint,type,single,multi_full,delta_full,multi_cls_only,delta_cls_only")
    deltas_cls = []
    deltas_reg = []
    for r in comp_rows:
        endpoint = r["endpoint"]
        typ = r["type"]
        s = r["single"]
        mf = r["multi_full"]
        d_full = r["delta_multi_full"]
        mco = r["multi_cls_only"]
        d_co = r["delta_multi_cls_only"]
        print(f"{endpoint},{typ},{s:.4f},{mf if mf is not None else 'NA'},{d_full if d_full is not None else 'NA'},{mco if mco is not None else 'NA'},{d_co if d_co is not None else 'NA'}")
        if d_full is not None:
            if typ == 'classification':
                deltas_cls.append(d_full)
            else:
                deltas_reg.append(d_full)

    if deltas_cls:
        print(f"Avg delta AUC (multi_full - single) across {len(deltas_cls)} cls endpoints: {sum(deltas_cls)/len(deltas_cls):.4f}")
    if deltas_reg:
        print(f"Avg delta R2 (multi_full - single) across {len(deltas_reg)} reg endpoints: {sum(deltas_reg)/len(deltas_reg):.4f}")

    # Ablation sensitivity relative to full model
    sens = ablation_sensitivity(exps, baseline_name=full_name)
    print("\nAblation sensitivity (vs full model):")
    print("experiment,avg_delta_auc,avg_delta_r2,improved_cls/degraded_cls,improved_reg/degraded_reg")
    # Order experiments by avg_delta_auc when available, else avg_delta_r2
    def keyfn(item):
        v = item[1]
        return (v.get("avg_delta_auc") or -1e9) + ((v.get("avg_delta_r2") or -1e9) * 0.01)
    for name, v in sorted(sens.items(), key=keyfn, reverse=True):
        print(
            f"{name},{v.get('avg_delta_auc')},{v.get('avg_delta_r2')},{v.get('improved_cls')}/{v.get('degraded_cls')},{v.get('improved_reg')}/{v.get('degraded_reg')}"
        )

    # Identify most sensitive endpoints per ablation (bottom/top 3)
    print("\nMost sensitive endpoints per ablation (classification):")
    for name, v in sens.items():
        pe = v.get("per_endpoint_cls", {})
        if not pe:
            continue
        items = list(pe.items())
        worst = items[:3]
        best = items[-3:]
        print(f"{name}: worst {worst} | best {best}")

    print("\nMost sensitive endpoints per ablation (regression):")
    for name, v in sens.items():
        pe = v.get("per_endpoint_reg", {})
        if not pe:
            continue
        items = list(pe.items())
        worst = items[:3]
        best = items[-3:]
        print(f"{name}: worst {worst} | best {best}")


if __name__ == "__main__":
    main()
