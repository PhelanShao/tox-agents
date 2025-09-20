#!/usr/bin/env python3
"""
Batch inference over multiple SMILES files using ToxD4C.

- Loads the model once and iterates all *.smiles files under input_dir
- Uses ToxD4C/inference_toxd4c.py classes to ensure consistent preprocessing
- Writes one CSV per input file, including *_prob probability columns
"""

import argparse
import sys
from pathlib import Path
from typing import List
import pandas as pd
import torch

# Add project root to path so we can import project modules
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from configs.toxd4c_config import get_enhanced_toxd4c_config
from inference_toxd4c import ToxD4CPredictor, SmilesDataset
from data.lmdb_dataset import collate_lmdb_batch


def run_inference_on_file(predictor: ToxD4CPredictor, smiles_file: Path, batch_size: int) -> pd.DataFrame:
    with smiles_file.open('r') as f:
        smiles_list: List[str] = f.readlines()

    dataset = SmilesDataset(smiles_list)
    if len(dataset) == 0:
        return pd.DataFrame()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_lmdb_batch,
        num_workers=0,
        drop_last=False,
    )
    return predictor.predict_on_loader(loader)


def main():
    parser = argparse.ArgumentParser(description="Batch inference for SMILES files")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained checkpoint *.pth')
    parser.add_argument('--input_dir', type=str, default='tox21 challenge', help='Directory containing *.smiles files')
    parser.add_argument('--pattern', type=str, default='*.smiles', help='Glob pattern for SMILES files')
    parser.add_argument('--output_dir', type=str, default='tox21_preds', help='Directory to save predictions')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (use small values to avoid OOM)')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], help='Inference device')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Config and predictor
    config = get_enhanced_toxd4c_config()
    predictor = ToxD4CPredictor(model_path=args.model_path, config=config, device=device)

    # Enumerate files
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        print(f"No files matched: {input_dir}/{args.pattern}")
        return 1

    print(f"Found {len(files)} SMILES files under {input_dir}")

    for fp in files:
        try:
            print(f"\n-> Predicting: {fp}")
            df = run_inference_on_file(predictor, fp, args.batch_size)
            if df.empty:
                print(f"   Skipped (no valid molecules): {fp}")
                continue
            out_path = output_dir / f"{fp.stem}_preds.csv"
            df.to_csv(out_path, index=False)
            print(f"   Saved: {out_path}")
        except RuntimeError as e:
            print(f"   RuntimeError on {fp}: {e}")
            print("   Tip: reduce --batch_size or use --device cpu")
        except Exception as e:
            print(f"   Failed on {fp}: {e}")

    print("\nBatch inference completed.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

