#!/usr/bin/env python3
"""
Head-only fine-tuning runner for ToxD4C single endpoints using a shared trunk.

This script automates running train.py with:
  - --resume_from <multi_full best.pth>
  - --freeze_trunk
  - --task_mode single + (--single_endpoint_cls i | --single_endpoint_reg j)

Features:
  - Covers all classification (26) and/or regression (5) endpoints, or a subset
  - Skips runs that already have a results JSON
  - Parallel execution with a configurable worker pool
  - Optional postprocess: ingest baseline results.json and export delta tables

Example (from repo root):
  python ToxD4C/tools/run_headft_single_endpoints.py \
    --resume_from_ckpt ToxD4C/experiments/r1c3_multi_full_seed_42_XXXX/checkpoints/r1c3_multi_full_seed_42_best.pth \
    --mode cls --seeds 42 --max_workers 2

Both cls+reg and postprocess delta:
  python ToxD4C/tools/run_headft_single_endpoints.py \
    --resume_from_ckpt ToxD4C/experiments/r1c3_multi_full_seed_42_XXXX/checkpoints/r1c3_multi_full_seed_42_best.pth \
    --mode both --seeds 42 --max_workers 3 --postprocess
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import subprocess


logger = logging.getLogger("run_headft")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    ap = argparse.ArgumentParser(description="Head-only fine-tuning runner for ToxD4C single endpoints")
    ap.add_argument('--resume_from_ckpt', type=str, required=True,
                    help='Path to multi_full best checkpoint (.pth) used as shared trunk')
    ap.add_argument('--mode', type=str, default='cls', choices=['cls', 'reg', 'both'],
                    help='Which endpoints to fine-tune')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42], help='Seeds to run')
    ap.add_argument('--cls_idx', type=str, default=None,
                    help='Comma-separated classification indices to run (override default set). Example: 0,1,3')
    ap.add_argument('--reg_idx', type=str, default=None,
                    help='Comma-separated regression indices to run (override default set). Example: 0,2')
    ap.add_argument('--epochs', type=int, default=3, help='Epochs for head-only fine-tuning')
    ap.add_argument('--lr', type=float, default=5e-4, help='Learning rate for head-only fine-tuning')
    ap.add_argument('--warmup', type=float, default=0.02, help='Warmup ratio for head-only fine-tuning')
    ap.add_argument('--batch_size', type=int, default=16, help='Batch size')
    ap.add_argument('--preprocessed_dir', type=str, default='data/data/processed',
                    help='Preprocessed LMDB dir (relative to ToxD4C)')
    ap.add_argument('--max_workers', type=int, default=2, help='Parallel workers')
    ap.add_argument('--dry_run', action='store_true', help='Print commands without executing')
    ap.add_argument('--postprocess', action='store_true',
                    help='After training, ingest baseline results and export delta CSVs')
    ap.add_argument('--baseline_results', type=str, default=None,
                    help='Path to multi_full results.json; if not set and --postprocess, infer from ckpt path')
    return ap.parse_args()


def toxd4c_root() -> Path:
    # tools/ -> parent is ToxD4C
    return Path(__file__).resolve().parents[1]


def find_existing_results(exp_name: str, base: Path) -> Optional[Path]:
    exps_dir = base / 'experiments'
    if not exps_dir.exists():
        return None
    # Search newest matching dir
    candidates = sorted(exps_dir.glob(f"{exp_name}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in candidates:
        rf = d / 'checkpoints' / f"{exp_name}_results.json"
        if rf.exists():
            return rf
    return None


def run_one(cmd: List[str], cwd: Path, dry: bool = False) -> int:
    pretty = ' '.join(cmd)
    logger.info(f"RUN: {pretty}")
    if dry:
        return 0
    res = subprocess.run(cmd, cwd=str(cwd))
    return res.returncode


def build_cmd(train_py: Path,
              exp_name: str,
              seed: int,
              resume_from_ckpt: Path,
              task_flag: List[str],
              batch_size: int,
              epochs: int,
              lr: float,
              warmup: float,
              preprocessed_dir: Path) -> List[str]:
    python = sys.executable or 'python'
    return [
        python, str(train_py),
        '--experiment_name', exp_name,
        '--seed', str(seed),
        '--batch_size', str(batch_size),
        '--num_epochs', str(epochs),
        '--learning_rate', str(lr),
        '--use_preprocessed',
        '--preprocessed_dir', str(preprocessed_dir),
        '--deterministic',
        '--warmup_ratio', str(warmup),
        '--task_mode', 'single',
    ] + task_flag + [
        '--resume_from', str(resume_from_ckpt),
        '--freeze_trunk'
    ]


def infer_results_from_ckpt(ckpt_path: Path) -> Optional[Path]:
    # Replace *_best.pth with *_results.json if possible
    try:
        if ckpt_path.name.endswith('_best.pth'):
            results_name = ckpt_path.name.replace('_best.pth', '_results.json')
            r = ckpt_path.parent / results_name
            return r if r.exists() else None
    except Exception:
        pass
    return None


def main():
    args = parse_args()

    base = toxd4c_root()
    train_py = base / 'train.py'
    if not train_py.exists():
        logger.error(f"train.py not found at {train_py}")
        sys.exit(1)

    resume_from_ckpt = Path(args.resume_from_ckpt).resolve()
    if not resume_from_ckpt.exists():
        logger.error(f"Checkpoint not found: {resume_from_ckpt}")
        sys.exit(1)

    # Determine indices
    if args.cls_idx:
        cls_indices = [int(x) for x in args.cls_idx.split(',') if x.strip() != '']
    else:
        cls_indices = list(range(26))
    if args.reg_idx:
        reg_indices = [int(x) for x in args.reg_idx.split(',') if x.strip() != '']
    else:
        reg_indices = list(range(5))

    # Build job list
    jobs = []
    if args.mode in ('cls', 'both'):
        for i in cls_indices:
            for seed in args.seeds:
                exp_name = f"r1c3_headft_cls_{i}_seed_{seed}"
                task_flag = ['--single_endpoint_cls', str(i)]
                jobs.append((exp_name, seed, task_flag))
    if args.mode in ('reg', 'both'):
        for j in reg_indices:
            for seed in args.seeds:
                exp_name = f"r1c3_headft_reg_{j}_seed_{seed}"
                task_flag = ['--single_endpoint_reg', str(j)]
                jobs.append((exp_name, seed, task_flag))

    preprocessed_dir = (base / args.preprocessed_dir).resolve()
    if not preprocessed_dir.exists():
        logger.warning(f"Preprocessed dir not found: {preprocessed_dir}. The training script may preprocess on the fly.")

    # Submit runs (skip existing)
    t0 = time.time()
    logger.info(f"Submitting {len(jobs)} head-only fine-tuning runs (mode={args.mode})…")

    def submit(job):
        exp_name, seed, task_flag = job
        # Skip if results exists
        if (rf := find_existing_results(exp_name, base)) is not None:
            logger.info(f"♻️ Skip {exp_name} (results exist): {rf}")
            return 0
        cmd = build_cmd(train_py, exp_name, seed, resume_from_ckpt, task_flag,
                        args.batch_size, args.epochs, args.lr, args.warmup, preprocessed_dir)
        return run_one(cmd, cwd=base, dry=args.dry_run)

    failed = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        fut2job = {ex.submit(submit, job): job for job in jobs}
        for fut in as_completed(fut2job):
            code = fut.result()
            exp_name = fut2job[fut][0]
            if code != 0:
                logger.warning(f"❌ {exp_name} failed with code {code}")
                failed.append(exp_name)
            else:
                logger.info(f"✅ {exp_name} done")

    logger.info(f"All submitted. Elapsed: {time.time()-t0:.1f}s. Failures: {len(failed)}")

    # Optional postprocess: ingest baseline and export deltas
    if args.postprocess:
        # Try to infer results.json if not set
        baseline_results = Path(args.baseline_results).resolve() if args.baseline_results else infer_results_from_ckpt(resume_from_ckpt)
        if not baseline_results or not baseline_results.exists():
            logger.warning("Postprocess requested but baseline results.json not found; skip.")
            return 0
        python = sys.executable or 'python'
        post_cmd = [
            python, str(base / 'sensitivity_analysis_r1c3.py'),
            '--seeds', str(args.seeds[0]),
            '--postprocess_only',
            '--cover_all_cls', '--cover_all_reg',
            '--ingest_baseline_results', str(baseline_results)
        ]
        logger.info(f"POSTPROCESS: {' '.join(post_cmd)}")
        if not args.dry_run:
            subprocess.run(post_cmd, cwd=str(base), check=False)

    return 0


if __name__ == '__main__':
    sys.exit(main())

