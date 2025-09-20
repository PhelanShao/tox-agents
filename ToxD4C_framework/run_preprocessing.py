#!/usr/bin/env python3
"""
Standalone preprocessing script for ToxD4C
Pre-generates 3D conformers and molecular features for faster training.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Add project path
sys.path.append(str(Path(__file__).parent))
from preprocess_data import preprocess_lmdb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='Preprocess ToxD4C data for faster training')
    parser.add_argument('--input_dir', type=str, default='data/data/dataset',
                       help='Directory containing raw LMDB files')
    parser.add_argument('--output_dir', type=str, default='data/data/processed',
                       help='Directory to save preprocessed LMDB files')
    parser.add_argument('--max_atoms', type=int, default=64,
                       help='Maximum number of atoms per molecule')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                       help='Data splits to preprocess')
    parser.add_argument('--force', action='store_true',
                       help='Force preprocessing even if output files exist')
    
    args = parser.parse_args()
    
    logger.info("🚀 ToxD4C Data Preprocessing")
    logger.info(f"   Input directory: {args.input_dir}")
    logger.info(f"   Output directory: {args.output_dir}")
    logger.info(f"   Max atoms: {args.max_atoms}")
    logger.info(f"   Splits: {args.splits}")
    logger.info(f"   Force overwrite: {args.force}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    total_start_time = time.time()
    processed_splits = []
    skipped_splits = []
    
    for split in args.splits:
        input_path = Path(args.input_dir) / f"{split}.lmdb"
        output_path = Path(args.output_dir) / f"{split}.lmdb"
        
        # Check if input exists
        if not input_path.exists():
            logger.warning(f"❌ Input file not found: {input_path}")
            skipped_splits.append(split)
            continue
        
        # Check if output already exists
        if output_path.exists() and not args.force:
            logger.info(f"⏭️  Output already exists, skipping: {output_path}")
            logger.info("   Use --force to overwrite existing files")
            skipped_splits.append(split)
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Processing {split} split")
        logger.info(f"{'='*60}")
        
        split_start_time = time.time()
        
        try:
            preprocess_lmdb(str(input_path), str(output_path), args.max_atoms)
            
            split_time = time.time() - split_start_time
            logger.info(f"✅ {split} split completed in {split_time:.1f}s")
            processed_splits.append(split)
            
        except Exception as e:
            logger.error(f"❌ Failed to process {split} split: {e}")
            skipped_splits.append(split)
    
    total_time = time.time() - total_start_time
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📋 PREPROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Processed splits: {len(processed_splits)}")
    logger.info(f"Skipped splits: {len(skipped_splits)}")
    
    if processed_splits:
        logger.info(f"✅ Successfully processed: {', '.join(processed_splits)}")
    
    if skipped_splits:
        logger.info(f"⏭️  Skipped: {', '.join(skipped_splits)}")
    
    if processed_splits:
        logger.info(f"\n🎉 Preprocessing completed!")
        logger.info(f"📁 Preprocessed data saved to: {args.output_dir}")
        logger.info(f"🚀 You can now train with: python train.py --use_preprocessed --preprocessed_dir {args.output_dir}")
        
        # Estimate speedup
        logger.info(f"\n⚡ Expected training speedup:")
        logger.info(f"   - Data loading: 5-20x faster")
        logger.info(f"   - Overall training: 2-5x faster")
        logger.info(f"   - No more 3D conformer generation during training!")
        
        return 0
    else:
        logger.error(f"❌ No splits were successfully processed")
        return 1

if __name__ == "__main__":
    exit(main())
