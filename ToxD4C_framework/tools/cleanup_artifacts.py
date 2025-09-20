#!/usr/bin/env python3
"""
Safe cleanup helper: list or remove generated artifacts by glob patterns.

Default is dry-run (only list). Use --apply to actually delete. Intended to help
tidy up large experiment outputs without accidental removals.

Examples:
  # List candidates under outliers_viz and topk_xyz
  python ToxD4C/tools/cleanup_artifacts.py --roots tox21_overlap_check/outliers_viz tox21_overlap_check/topk_xyz --patterns "*.png" "*.xyz" --dry_run

  # Actually delete only PNGs under a specific task/mode
  python ToxD4C/tools/cleanup_artifacts.py --roots tox21_overlap_check/outliers_viz/nr-ahr/low --patterns "*.png" --apply
"""

import argparse
from pathlib import Path
from typing import List


def collect(roots: List[Path], patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for r in roots:
        if not r.exists():
            continue
        for pat in patterns:
            files.extend(r.rglob(pat))
    # unique
    seen = set()
    uniq: List[Path] = []
    for p in files:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            uniq.append(p)
    return uniq


def main():
    ap = argparse.ArgumentParser(description='Safe cleanup for generated artifacts (dry-run by default)')
    ap.add_argument('--roots', nargs='+', type=str, required=True, help='Root directories to search')
    ap.add_argument('--patterns', nargs='+', type=str, default=['*.tmp'], help='Glob patterns to delete')
    ap.add_argument('--apply', action='store_true', help='Actually delete (otherwise dry-run)')
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots]
    files = collect(roots, args.patterns)
    if not files:
        print('No files matched')
        return 0

    total = 0
    for p in files:
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        total += size
        action = 'DELETE' if args.apply else 'FOUND '
        print(f"{action}: {p} ({size/1e6:.2f} MB)")
        if args.apply:
            try:
                p.unlink()
            except Exception as e:
                print(f"  Failed to delete {p}: {e}")

    print(f"Total files: {len(files)}  Total size: {total/1e6:.2f} MB")
    if not args.apply:
        print("Dry-run only. Re-run with --apply to actually delete.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

