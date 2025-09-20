#!/usr/bin/env python3
"""
One-click runner for all Tox21 tasks:
- Exports per-endpoint Top-K most dissimilar molecules as XYZ files
- Generates per-endpoint ECFP4 similarity-map visualizations

This wraps:
- tools/export_topk_xyz_per_task.py (tasks=all)
- tools/visualize_structural_outliers.py (per_task)

Usage:
  python ToxD4C/tools/run_all_tasks_outliers.py \
    --overlap_dir tox21_overlap_check \
    --labels_dir "tox21 challenge" \
    --preds_dir tox21_preds \
    --xyz_out tox21_overlap_check/topk_xyz \
    --viz_out tox21_overlap_check/outliers_viz \
    --top_k 5
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n>>>", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    p = argparse.ArgumentParser(description='Run per-task Top-K XYZ export and per-task outlier visualizations')
    p.add_argument('--overlap_dir', type=str, default='tox21_overlap_check')
    p.add_argument('--labels_dir', type=str, default='tox21 challenge')
    p.add_argument('--preds_dir', type=str, default='tox21_preds')
    p.add_argument('--xyz_out', type=str, default='tox21_overlap_check/topk_xyz')
    p.add_argument('--viz_out', type=str, default='tox21_overlap_check/outliers_viz')
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--use_chirality', action='store_true')
    p.add_argument('--mode', type=str, default='low', choices=['low','high','both'], help='Select lowest (dissimilar), highest (similar) or both')
    p.add_argument('--with_train_xyz', action='store_true', help='Also export train NN as XYZ alongside challenge molecules')
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    # 1) Export XYZ per endpoint (all tasks)
    modes = ['low','high'] if args.mode == 'both' else [args.mode]
    for m in modes:
        cmd = [
            sys.executable,
            str(repo_root / 'tools' / 'export_topk_xyz_per_task.py'),
            '--overlap_dir', args.overlap_dir,
            '--labels_dir', args.labels_dir,
            '--output_dir', args.xyz_out,
            '--top_k', str(args.top_k),
            '--tasks', 'all',
            '--mode', m
        ]
        if args.with_train_xyz:
            cmd.append('--with_train_xyz')
        run(cmd)

    # 2) Visualize per endpoint similarity maps (all tasks available in preds)
    # 2) Visualizations
    for m in modes:
        cmd = [
            sys.executable,
            str(repo_root / 'tools' / 'visualize_structural_outliers.py'),
            '--overlap_dir', args.overlap_dir,
            '--preds_dir', args.preds_dir,
            '--output_dir', args.viz_out,
            '--top_k', str(args.top_k),
            '--per_task',
            '--mode', m
        ]
        if args.use_chirality:
            cmd.append('--use_chirality')
        run(cmd)

    print("\nAll tasks processed. XYZ exported to:", args.xyz_out)
    print("Visualizations saved to:", args.viz_out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
