#!/usr/bin/env python3
"""
Benchmark the performance difference between raw and preprocessed data loading
"""

import time
import logging
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project path
import sys
sys.path.append(str(Path(__file__).parent))
from data.lmdb_dataset import create_lmdb_dataloaders

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def benchmark_dataloader(dataloader, name: str, num_batches: int = 20):
    """Benchmark a dataloader's performance."""
    logger.info(f"🔥 Benchmarking {name}...")
    
    times = []
    total_samples = 0
    
    try:
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            start_time = time.time()
            
            # Simulate some processing
            if batch is not None:
                batch_size = len(batch['smiles'])
                total_samples += batch_size
                
                # Move to GPU if available (simulate training)
                if torch.cuda.is_available():
                    for key in ['atom_features', 'coordinates', 'bond_features', 'edge_index']:
                        if key in batch:
                            batch[key] = batch[key].cuda()
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            if i % 5 == 0:
                logger.info(f"   Batch {i}: {elapsed:.3f}s")
        
        avg_time = sum(times) / len(times) if times else 0
        total_time = sum(times)
        throughput = total_samples / total_time if total_time > 0 else 0
        
        logger.info(f"✅ {name} Results:")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Average time per batch: {avg_time:.3f}s")
        logger.info(f"   Total samples: {total_samples}")
        logger.info(f"   Throughput: {throughput:.1f} samples/sec")
        
        return {
            'name': name,
            'total_time': total_time,
            'avg_time_per_batch': avg_time,
            'total_samples': total_samples,
            'throughput': throughput,
            'num_batches': len(times)
        }
        
    except Exception as e:
        logger.error(f"❌ {name} failed: {e}")
        return None

def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description='Benchmark ToxD4C data loading performance')
    parser.add_argument('--raw_data_dir', type=str, default='data/data/dataset',
                       help='Directory with raw LMDB files')
    parser.add_argument('--preprocessed_data_dir', type=str, default='data/data/processed',
                       help='Directory with preprocessed LMDB files')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for benchmarking')
    parser.add_argument('--num_batches', type=int, default=20,
                       help='Number of batches to benchmark')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    logger.info("🚀 ToxD4C Data Loading Performance Benchmark")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Number of batches: {args.num_batches}")
    logger.info(f"   Number of workers: {args.num_workers}")
    logger.info(f"   CUDA available: {torch.cuda.is_available()}")
    
    results = []
    
    # Benchmark raw data (slow)
    logger.info("\n" + "="*60)
    logger.info("RAW DATA (Real-time 3D conformer generation)")
    logger.info("="*60)
    
    try:
        if Path(args.raw_data_dir).exists():
            train_loader, _, _ = create_lmdb_dataloaders(
                args.raw_data_dir,
                batch_size=args.batch_size
            )
            
            result = benchmark_dataloader(
                train_loader, 
                "Raw Data (Real-time 3D)", 
                args.num_batches
            )
            if result:
                results.append(result)
        else:
            logger.warning(f"Raw data directory not found: {args.raw_data_dir}")
    
    except Exception as e:
        logger.error(f"Failed to benchmark raw data: {e}")
    
    # Benchmark preprocessed data (fast)
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSED DATA (Pre-computed 3D conformers)")
    logger.info("="*60)
    
    try:
        if Path(args.preprocessed_data_dir).exists():
            train_loader, _, _ = create_lmdb_dataloaders(
                args.preprocessed_data_dir,
                batch_size=args.batch_size
            )
            
            result = benchmark_dataloader(
                train_loader,
                "Preprocessed Data (Pre-computed 3D)",
                args.num_batches
            )
            if result:
                results.append(result)
        else:
            logger.warning(f"Preprocessed data directory not found: {args.preprocessed_data_dir}")
            logger.info("💡 Run: python run_preprocessing.py to create preprocessed data")
    
    except Exception as e:
        logger.error(f"Failed to benchmark preprocessed data: {e}")
    
    # Compare results
    if len(results) >= 2:
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE COMPARISON")
        logger.info("="*60)
        
        raw = results[0]
        preprocessed = results[1]
        
        speedup = raw['total_time'] / preprocessed['total_time'] if preprocessed['total_time'] > 0 else 0
        throughput_improvement = preprocessed['throughput'] / raw['throughput'] if raw['throughput'] > 0 else 0
        
        logger.info(f"📊 Performance Metrics:")
        logger.info(f"   Raw data total time: {raw['total_time']:.2f}s")
        logger.info(f"   Preprocessed total time: {preprocessed['total_time']:.2f}s")
        logger.info(f"   🚀 Speedup: {speedup:.1f}x faster")
        logger.info("")
        logger.info(f"   Raw data throughput: {raw['throughput']:.1f} samples/sec")
        logger.info(f"   Preprocessed throughput: {preprocessed['throughput']:.1f} samples/sec")
        logger.info(f"   📈 Throughput improvement: {throughput_improvement:.1f}x")
        logger.info("")
        logger.info(f"   Raw avg time/batch: {raw['avg_time_per_batch']:.3f}s")
        logger.info(f"   Preprocessed avg time/batch: {preprocessed['avg_time_per_batch']:.3f}s")
        
        # Estimate training time savings
        batches_per_epoch = 4970  # Approximate from your logs
        epochs = 20
        total_batches = batches_per_epoch * epochs
        
        raw_training_time = total_batches * raw['avg_time_per_batch'] / 3600  # hours
        preprocessed_training_time = total_batches * preprocessed['avg_time_per_batch'] / 3600  # hours
        time_saved = raw_training_time - preprocessed_training_time
        
        logger.info(f"\n⏰ Estimated Training Time (20 epochs, {batches_per_epoch} batches/epoch):")
        logger.info(f"   Raw data: {raw_training_time:.1f} hours")
        logger.info(f"   Preprocessed data: {preprocessed_training_time:.1f} hours")
        logger.info(f"   💰 Time saved: {time_saved:.1f} hours ({time_saved*60:.0f} minutes)")
        
        logger.info("\n🎯 Recommendation:")
        if speedup > 3:
            logger.info("   ✅ STRONGLY RECOMMENDED: Use preprocessed data for training")
            logger.info("   🚀 Significant performance improvement!")
        elif speedup > 1.5:
            logger.info("   ✅ RECOMMENDED: Use preprocessed data for training")
            logger.info("   📈 Notable performance improvement")
        else:
            logger.info("   ⚠️  Modest improvement - consider other optimizations")
    
    elif len(results) == 1:
        logger.info(f"\n📊 Single benchmark completed: {results[0]['name']}")
        logger.info(f"   Throughput: {results[0]['throughput']:.1f} samples/sec")
    
    else:
        logger.warning("\n❌ No successful benchmarks completed")
        logger.info("💡 Make sure data directories exist and contain valid LMDB files")

if __name__ == "__main__":
    main()
