#!/usr/bin/env python3
"""
Compile head-only fine-tuning (r1c3_headft_*) single-endpoint results and
compare against multi_full and classification-only multi-task baselines.

Outputs a CSV at ToxD4C/r1c3_sensitivity_results/single_vs_multi_comparison_headft.csv
with columns: endpoint,type,single,multi_full,delta_full,multi_cls_only,delta_cls_only

This script reads:
  - ToxD4C/experiments/r1c3_headft_*/*/checkpoints/*_results.json (single)
  - ToxD4C/detailed_endpoint_results_all_experiments.json (multi baselines)
"""

import json
import csv
from pathlib import Path
import re


def main():
    repo = Path(__file__).resolve().parents[1]
    exp_dir = repo / 'experiments'
    detail_path = repo / 'detailed_endpoint_results_all_experiments.json'
    out_dir = repo / 'r1c3_sensitivity_results'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load detailed multi-task experiments
    details = json.loads(detail_path.read_text())
    exps = {}
    for entry in details:
        name = entry.get('experiment_name')
        detailed = entry.get('detailed_results', {}) or {}
        cls = detailed.get('classification_endpoints', {}) or {}
        reg = detailed.get('regression_endpoints', {}) or {}
        exps[name] = {
            'classification': {k: (v.get('auc') if isinstance(v, dict) else None) for k, v in cls.items()},
            'regression': {k: (v.get('r2') if isinstance(v, dict) else None) for k, v in reg.items()},
            'avg_auc': entry.get('summary', {}).get('classification', {}).get('avg_auc'),
            'avg_r2': entry.get('summary', {}).get('regression', {}).get('avg_r2'),
        }

    multi_full_name = 'toxd4c_ablation_full_model'
    multi_cls_only_name = 'toxd4c_ablation_classification_only'
    if multi_full_name not in exps:
        raise SystemExit("Baseline 'toxd4c_ablation_full_model' not found in detailed_endpoint_results_all_experiments.json")
    multi_full = exps[multi_full_name]
    multi_cls_only = exps.get(multi_cls_only_name)

    # Index→name mapping
    CLASSIFICATION_TASKS = [
        'Carcinogenicity','Ames Mutagenicity','Respiratory toxicity','Eye irritation','Eye corrosion',
        'Cardiotoxicity1','Cardiotoxicity10','Cardiotoxicity30','Cardiotoxicity5','CYP1A2','CYP2C19','CYP2C9',
        'CYP2D6','CYP3A4','NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD','NR-PPAR-gamma',
        'SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']
    REGRESSION_TASKS = ['Acute oral toxicity (LD50)','LC50DM','BCF','LC50','IGC50']

    # Gather newest headft results per index
    singles = {}
    re_cls = re.compile(r'^r1c3_headft_cls_(\d+)_seed_')
    re_reg = re.compile(r'^r1c3_headft_reg_(\d+)_seed_')

    for d in exp_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        m = re_cls.match(name)
        if m:
            idx = int(m.group(1))
            # result file omits the timestamp suffix in directory name
            base = re.sub(r'_[0-9]{8}_[0-9]{6}$', '', name)
            rf = d / 'checkpoints' / f'{base}_results.json'
            if rf.exists():
                prev = singles.get(('cls', idx))
                if prev is None or rf.stat().st_mtime > prev[0]:
                    try:
                        data = json.loads(rf.read_text())
                        val = data.get('final_metrics', {}).get('task_0_auc')
                        singles[('cls', idx)] = (rf.stat().st_mtime, val)
                    except Exception:
                        pass
            continue
        m = re_reg.match(name)
        if m:
            idx = int(m.group(1))
            base = re.sub(r'_[0-9]{8}_[0-9]{6}$', '', name)
            rf = d / 'checkpoints' / f'{base}_results.json'
            if rf.exists():
                prev = singles.get(('reg', idx))
                if prev is None or rf.stat().st_mtime > prev[0]:
                    try:
                        data = json.loads(rf.read_text())
                        val = data.get('final_metrics', {}).get('task_0_r2')
                        singles[('reg', idx)] = (rf.stat().st_mtime, val)
                    except Exception:
                        pass
            continue

    rows = []
    for (typ, idx), (_t, sval) in sorted(singles.items(), key=lambda x: (x[0][0], x[0][1])):
        if typ == 'cls':
            ep = CLASSIFICATION_TASKS[idx] if idx < len(CLASSIFICATION_TASKS) else f'cls_{idx}'
            mf = multi_full['classification'].get(ep)
            mco = multi_cls_only['classification'].get(ep) if multi_cls_only else None
            rows.append({
                'endpoint': ep,
                'type': 'classification',
                'single': sval,
                'multi_full': mf,
                'delta_full': (mf - sval) if (mf is not None and sval is not None) else None,
                'multi_cls_only': mco,
                'delta_cls_only': (mco - sval) if (mco is not None and sval is not None) else None,
            })
        else:
            ep = REGRESSION_TASKS[idx] if idx < len(REGRESSION_TASKS) else f'reg_{idx}'
            mf = multi_full['regression'].get(ep)
            rows.append({
                'endpoint': ep,
                'type': 'regression',
                'single': sval,
                'multi_full': mf,
                'delta_full': (mf - sval) if (mf is not None and sval is not None) else None,
                'multi_cls_only': None,
                'delta_cls_only': None,
            })

    out_csv = out_dir / 'single_vs_multi_comparison_headft.csv'
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['endpoint','type','single','multi_full','delta_full','multi_cls_only','delta_cls_only'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Print a brief summary
    cls_d = [r['delta_full'] for r in rows if r['type']=='classification' and r['delta_full'] is not None]
    reg_d = [r['delta_full'] for r in rows if r['type']=='regression' and r['delta_full'] is not None]
    cls2_d = [r['delta_cls_only'] for r in rows if r['type']=='classification' and r['delta_cls_only'] is not None]

    def stats(arr):
        if not arr:
            return 'n=0'
        import statistics
        return f"n={len(arr)}, improved={sum(1 for x in arr if x>0)}, mean={statistics.fmean(arr):.4f}, median={statistics.median(arr):.4f}"

    print('[HEADFT] classification Δ_full:', stats(cls_d))
    print('[HEADFT] regression Δ_full:', stats(reg_d))
    if cls2_d:
        print('[HEADFT] classification Δ_cls_only:', stats(cls2_d))
    print(f'[HEADFT] wrote {out_csv}')


if __name__ == '__main__':
    main()
