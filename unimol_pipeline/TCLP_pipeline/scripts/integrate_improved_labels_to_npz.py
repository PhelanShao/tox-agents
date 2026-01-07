#!/usr/bin/env python3
"""
Integrate improved TCPL labels into NPZ file.

Rules:
1. Keep y=0 samples unchanged (original y=0 in processed_final8k213_original.csv)
2. For samples with chid mapping: use improved tcpl_binary_improved labels
3. For samples without chid mapping but y>0 in original: keep as y=1 (toxic)

Outputs:
- Updated NPZ with improved labels
- Summary statistics
"""

import os
import numpy as np
import pandas as pd
import argparse
import shutil
from datetime import datetime


def load_data(base_dir: str, npz_path: str):
    """Load all required data."""
    # Load original CSV to identify y=0 samples and y values
    original_csv = os.path.join(base_dir, "original", "processed_final8k213_original.csv")
    original_df = pd.read_csv(original_csv)
    print(f"Loaded original CSV: {len(original_df)} rows")

    # Load improved scores
    improved_csv = os.path.join(base_dir, "processed", "tcpl_scores_improved.csv")
    improved_df = pd.read_csv(improved_csv)
    print(f"Loaded improved scores: {len(improved_df)} rows")

    # Load NPZ
    npz_data = np.load(npz_path, allow_pickle=True)
    print(f"Loaded NPZ: {len(npz_data['SampleName'])} samples")

    return original_df, improved_df, npz_data


def create_pubchem_to_chid_mapping(base_dir: str):
    """Create PUBCHEM_CID to chid mapping."""
    bridge_path = os.path.join(base_dir, "processed", "chemical_bridge_table.csv")
    bridge = pd.read_csv(bridge_path)

    # Group by PUBCHEM_CID and take first chid
    mapping = bridge.groupby("PUBCHEM_CID")["chid"].first().to_dict()
    print(f"Created PUBCHEM->chid mapping: {len(mapping)} entries")
    return mapping


def integrate_labels(original_df, improved_df, npz_data, pubchem_to_chid):
    """Integrate improved labels while preserving y=0."""
    sample_names = npz_data["SampleName"]
    old_y = npz_data["y"]
    n_samples = len(sample_names)

    # Create lookups from original CSV
    original_y_by_pubchem = original_df.set_index("PUBCHEM_CID")["y"].to_dict()
    original_y0_pubchem = set(original_df[original_df["y"] == 0]["PUBCHEM_CID"].values)
    print(f"Original y=0 samples in CSV: {len(original_y0_pubchem)}")

    # Create improved label lookup by chid
    improved_by_chid = improved_df.set_index("chid")["tcpl_binary_improved"].to_dict()
    improved_ensemble_by_chid = improved_df.set_index("chid")["S_ensemble"].to_dict()

    # Process each sample
    new_y = np.zeros(n_samples, dtype=np.int64)
    new_ensemble = np.full(n_samples, np.nan, dtype=np.float64)

    stats = {
        "kept_y0_original": 0,
        "with_mapping_relabeled_0": 0,
        "with_mapping_relabeled_1": 0,
        "no_mapping_kept_y1": 0,
        "no_mapping_invalid": 0,
    }

    for i in range(n_samples):
        pubchem_id = float(sample_names[i])

        # Get original y value from CSV (not from NPZ which might be already binary)
        original_y_raw = original_y_by_pubchem.get(pubchem_id, None)

        # Rule 1: If original CSV has y=0 for this PUBCHEM_CID, keep it as 0
        if pubchem_id in original_y0_pubchem:
            new_y[i] = 0
            new_ensemble[i] = 0.0
            stats["kept_y0_original"] += 1
            continue

        # Rule 2: For samples with y>0, try to use improved label
        chid = pubchem_to_chid.get(pubchem_id)

        if chid is not None and chid in improved_by_chid:
            improved_label = improved_by_chid[chid]
            improved_score = improved_ensemble_by_chid.get(chid, np.nan)

            if improved_label == -1 or pd.isna(improved_label):
                # No valid improved label, but original y>0, so label as 1
                new_y[i] = 1
                stats["no_mapping_kept_y1"] += 1
            else:
                new_y[i] = int(improved_label)
                new_ensemble[i] = improved_score if not pd.isna(improved_score) else np.nan

                if new_y[i] == 0:
                    stats["with_mapping_relabeled_0"] += 1
                else:
                    stats["with_mapping_relabeled_1"] += 1
        else:
            # No chid mapping, but original y>0, so keep as y=1 (toxic)
            if original_y_raw is not None and original_y_raw > 0:
                new_y[i] = 1
                stats["no_mapping_kept_y1"] += 1
            else:
                # Edge case: no mapping and no original data
                new_y[i] = old_y[i]  # fallback to NPZ value
                stats["no_mapping_invalid"] += 1

    print("\n=== Label Integration Statistics ===")
    print(f"  Kept y=0 (original CSV y=0): {stats['kept_y0_original']}")
    print(f"  With mapping, relabeled to 0: {stats['with_mapping_relabeled_0']}")
    print(f"  With mapping, relabeled to 1: {stats['with_mapping_relabeled_1']}")
    print(f"  No mapping, kept as y=1 (original y>0): {stats['no_mapping_kept_y1']}")
    print(f"  No mapping, invalid (fallback): {stats['no_mapping_invalid']}")

    print(f"\nNew label distribution:")
    print(f"  y=0: {(new_y == 0).sum()}")
    print(f"  y=1: {(new_y == 1).sum()}")

    # Compare with old labels
    changed = (new_y != old_y).sum()
    print(f"\nLabel changes from original NPZ:")
    print(f"  Changed: {changed}")
    print(f"  Unchanged: {n_samples - changed}")

    return new_y, new_ensemble, stats


def save_updated_npz(npz_data, new_y, new_ensemble, output_path):
    """Save updated NPZ with new labels."""
    # Copy all arrays
    arrays = {}
    for key in npz_data.keys():
        if key == "y":
            arrays[key] = new_y
        else:
            arrays[key] = npz_data[key]
    
    # Add S_ensemble as new field
    arrays["S_ensemble"] = new_ensemble
    
    np.savez_compressed(output_path, **arrays)
    print(f"\nSaved updated NPZ to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Integrate improved labels to NPZ")
    parser.add_argument("--base_dir", type=str, default=".",
                       help="Base directory of TCLP pipeline")
    parser.add_argument("--npz", type=str, 
                       default="7330merged_structures_merged_teacherfull_L1_H6.npz",
                       help="Input NPZ file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output NPZ path (default: adds _improved suffix)")
    args = parser.parse_args()
    
    npz_path = os.path.join(args.base_dir, args.npz)
    output_path = args.output or npz_path.replace(".npz", "_improved.npz")
    
    # Backup original
    backup_path = npz_path.replace(".npz", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")
    shutil.copy2(npz_path, backup_path)
    print(f"Backed up original to: {backup_path}")
    
    # Load data
    original_df, improved_df, npz_data = load_data(args.base_dir, npz_path)
    pubchem_to_chid = create_pubchem_to_chid_mapping(args.base_dir)
    
    # Integrate labels
    new_y, new_ensemble, stats = integrate_labels(
        original_df, improved_df, npz_data, pubchem_to_chid)
    
    # Save
    save_updated_npz(npz_data, new_y, new_ensemble, output_path)
    
    print("\n" + "="*60)
    print("DONE: NPZ updated with improved labels")
    print("="*60)


if __name__ == "__main__":
    main()

