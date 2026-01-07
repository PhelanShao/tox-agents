#!/usr/bin/env python3
"""
NPZ Chunk Generation Script for TCPL-labeled Unimol Dataset
Generates multiple smaller npz files with updated tcpl labels from CSV data.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TCPLNpzGenerator:
    def __init__(self, csv_path: str, npz_path: str, output_dir: str = "tcpl_chunks"):
        """
        Initialize the TCPL NPZ generator.
        
        Args:
            csv_path: Path to CSV file with tcpl labels
            npz_path: Path to original npz file
            output_dir: Directory to save generated chunks
        """
        self.csv_path = Path(csv_path)
        self.npz_path = Path(npz_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.csv_data = None
        self.npz_data = None
        self.mapping = None
        self.filtered_data = None
        
    def load_data(self) -> bool:
        """Load CSV and npz data."""
        try:
            logger.info(f"Loading CSV data from {self.csv_path}")
            self.csv_data = pd.read_csv(self.csv_path)
            logger.info(f"CSV loaded: {len(self.csv_data)} rows")
            
            logger.info(f"Loading npz data from {self.npz_path}")
            self.npz_data = np.load(self.npz_path, allow_pickle=True)
            logger.info(f"npz loaded with {len(self.npz_data['SampleName'])} records")
            
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def create_mapping(self) -> bool:
        """Create mapping between CSV and npz data."""
        try:
            logger.info("Creating data mapping...")
            
            # Create PUBCHEM_CID to index mappings
            csv_id_to_index = {float(row['PUBCHEM_CID']): idx for idx, row in self.csv_data.iterrows()}
            npz_id_to_index = {float(self.npz_data['SampleName'][i]): i for i in range(len(self.npz_data['SampleName']))}
            
            # Find matching records
            csv_pubchem_ids = set(csv_id_to_index.keys())
            npz_pubchem_ids = set(npz_id_to_index.keys())
            matched_ids = csv_pubchem_ids.intersection(npz_pubchem_ids)
            
            logger.info(f"Found {len(matched_ids)} matching records")
            
            # Create mapping
            self.mapping = {}
            for pubchem_id in matched_ids:
                csv_idx = csv_id_to_index[pubchem_id]
                npz_idx = npz_id_to_index[pubchem_id]
                self.mapping[npz_idx] = csv_idx
            
            return True
        except Exception as e:
            logger.error(f"Error creating mapping: {e}")
            return False
    
    def filter_labeled_data(self) -> bool:
        """Filter data to include only records with valid tcpl labels."""
        try:
            logger.info("Filtering data for valid tcpl labels...")
            
            valid_indices = []
            for npz_idx, csv_idx in self.mapping.items():
                tcpl_binary = self.csv_data.iloc[csv_idx]['tcpl_binary_compliant']
                if tcpl_binary != -1:  # Exclude unlabeled records
                    valid_indices.append((npz_idx, csv_idx))
            
            logger.info(f"Found {len(valid_indices)} records with valid tcpl labels")
            
            # Sort by npz index for consistent ordering
            valid_indices.sort(key=lambda x: x[0])
            self.filtered_data = valid_indices
            
            return True
        except Exception as e:
            logger.error(f"Error filtering data: {e}")
            return False
    
    def generate_chunks(self, chunk_size: int = 1200) -> bool:
        """Generate npz chunks with specified size."""
        try:
            logger.info(f"Generating chunks with size {chunk_size}...")
            
            total_records = len(self.filtered_data)
            num_chunks = (total_records + chunk_size - 1) // chunk_size
            
            chunk_manifest = {
                'total_records': total_records,
                'chunk_size': chunk_size,
                'num_chunks': num_chunks,
                'chunks': []
            }
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_records)
                chunk_data = self.filtered_data[start_idx:end_idx]
                actual_size = len(chunk_data)
                
                logger.info(f"Generating chunk {chunk_idx + 1}/{num_chunks} with {actual_size} records")
                
                # Create chunk filename
                chunk_filename = f"tcpl_unimol_chunk{chunk_idx + 1}_{actual_size}.npz"
                chunk_path = self.output_dir / chunk_filename
                
                # Generate chunk data
                chunk_arrays = self._create_chunk_arrays(chunk_data)
                
                # Save chunk
                np.savez_compressed(chunk_path, **chunk_arrays)
                
                # Update manifest
                chunk_info = {
                    'chunk_id': chunk_idx + 1,
                    'filename': chunk_filename,
                    'size': actual_size,
                    'start_record': start_idx,
                    'end_record': end_idx - 1
                }
                chunk_manifest['chunks'].append(chunk_info)
                
                logger.info(f"Saved chunk: {chunk_filename}")
            
            # Save manifest
            manifest_path = self.output_dir / "chunk_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(chunk_manifest, f, indent=2)
            
            logger.info(f"Generated {num_chunks} chunks successfully")
            logger.info(f"Manifest saved to: {manifest_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error generating chunks: {e}")
            return False
    
    def _create_chunk_arrays(self, chunk_data: List[Tuple[int, int]]) -> Dict:
        """Create arrays for a single chunk."""
        chunk_size = len(chunk_data)
        
        # Initialize arrays
        chunk_arrays = {}
        
        # Get all keys from original npz
        original_keys = list(self.npz_data.keys())
        
        # Process each key
        for key in original_keys:
            if key == '__label_order__':
                # Skip this metadata key
                continue
            elif key == 'y':
                # Use tcpl_binary_compliant as new y labels
                chunk_arrays[key] = np.array([
                    int(self.csv_data.iloc[csv_idx]['tcpl_binary_compliant'])
                    for npz_idx, csv_idx in chunk_data
                ], dtype=np.int32)
            else:
                # Copy original data
                original_array = self.npz_data[key]
                if len(original_array.shape) == 1:
                    # 1D array
                    chunk_arrays[key] = np.array([
                        original_array[npz_idx] for npz_idx, csv_idx in chunk_data
                    ], dtype=original_array.dtype)
                else:
                    # Multi-dimensional array
                    chunk_arrays[key] = np.array([
                        original_array[npz_idx] for npz_idx, csv_idx in chunk_data
                    ])
        
        # Add new tcpl fields
        tcpl_fields = [
            'tcpl_binary_compliant',
            'tcpl_ternary_compliant', 
            'S_c_tcpl_compliant',
            'tcpl_n_tested_compliant',
            'tcpl_n_positive_compliant'
        ]
        
        for field in tcpl_fields:
            if field in self.csv_data.columns:
                chunk_arrays[field] = np.array([
                    self.csv_data.iloc[csv_idx][field] for npz_idx, csv_idx in chunk_data
                ])
        
        # Add PUBCHEM_CID for reference
        chunk_arrays['PUBCHEM_CID'] = np.array([
            self.csv_data.iloc[csv_idx]['PUBCHEM_CID'] for npz_idx, csv_idx in chunk_data
        ])
        
        return chunk_arrays
    
    def validate_chunks(self) -> bool:
        """Validate generated chunks."""
        try:
            logger.info("Validating generated chunks...")
            
            manifest_path = self.output_dir / "chunk_manifest.json"
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            total_validated = 0
            for chunk_info in manifest['chunks']:
                chunk_path = self.output_dir / chunk_info['filename']
                
                if not chunk_path.exists():
                    logger.error(f"Chunk file not found: {chunk_path}")
                    return False
                
                # Load and validate chunk
                chunk_data = np.load(chunk_path, allow_pickle=True)
                
                # Check required fields
                required_fields = ['coord', 'symbol', 'y', 'SampleName']
                for field in required_fields:
                    if field not in chunk_data.keys():
                        logger.error(f"Missing required field '{field}' in {chunk_info['filename']}")
                        return False
                
                # Check sizes
                expected_size = chunk_info['size']
                actual_size = len(chunk_data['y'])
                if actual_size != expected_size:
                    logger.error(f"Size mismatch in {chunk_info['filename']}: expected {expected_size}, got {actual_size}")
                    return False
                
                total_validated += actual_size
                logger.info(f"Validated chunk {chunk_info['chunk_id']}: {actual_size} records")
            
            logger.info(f"All chunks validated successfully. Total records: {total_validated}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating chunks: {e}")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate TCPL-labeled NPZ chunks")
    parser.add_argument("--csv", default="processed_final8k213_tcpl_labeled_final.csv", 
                       help="Path to CSV file with tcpl labels")
    parser.add_argument("--npz", default="7330merged_structures_merged.npz",
                       help="Path to original npz file")
    parser.add_argument("--output", default="tcpl_chunks",
                       help="Output directory for chunks")
    parser.add_argument("--chunk-size", type=int, default=1200,
                       help="Records per chunk")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = TCPLNpzGenerator(args.csv, args.npz, args.output)
    
    # Execute pipeline
    if not generator.load_data():
        return 1
    
    if not generator.create_mapping():
        return 1
    
    if not generator.filter_labeled_data():
        return 1
    
    if not generator.generate_chunks(args.chunk_size):
        return 1
    
    if not generator.validate_chunks():
        return 1
    
    logger.info("NPZ chunk generation completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
