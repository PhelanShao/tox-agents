#file_converter.py
import os
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def parse_xyz_frame(lines: List[str], frame_id: int) -> Tuple[Dict, List[str], List[List[float]]]:
    """Parse a single XYZ structure frame and assign sequential ID."""
    try:
        n_atoms = int(lines[0])
        # Replace the second line with sequential ID
        prop_line = f"id:{frame_id}"
        
        symbols = []
        coords = []
        for line in lines[2:2+n_atoms]:
            parts = line.strip().split()
            if len(parts) == 4:
                symbol, x, y, z = parts
                symbols.append(symbol)
                coords.append([float(x), float(y), float(z)])
        
        if len(symbols) != n_atoms or len(coords) != n_atoms:
            return None
            
        properties = {'id': frame_id}
        return properties, symbols, coords
    except Exception as e:
        logger.error(f"Frame parsing error: {str(e)}")
        return None

def read_xyz_file(filename: str) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """Read XYZ file and convert to arrays with sequential IDs."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    all_properties = []
    all_symbols = []
    all_coords = []
    invalid_frames = []
    
    frame_id = 0  # Initialize sequential ID counter
    i = 0
    pbar = tqdm(total=len(lines), desc="Reading structures")
    
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
            
        try:
            n_atoms = int(lines[i].strip())
            frame_lines = lines[i:i + n_atoms + 2]
            
            if len(frame_lines) < n_atoms + 2:
                break
                
            result = parse_xyz_frame(frame_lines, frame_id)
            if result is None:
                invalid_frames.append(i)
            else:
                properties, symbols, coords = result
                all_properties.append(properties)
                all_symbols.append(symbols)
                all_coords.append(coords)
                frame_id += 1  # Increment ID counter
            
            i += n_atoms + 2
            pbar.update(n_atoms + 2)
        except:
            invalid_frames.append(i)
            i += 1
            pbar.update(1)
    
    pbar.close()
    
    # Convert to numpy arrays
    data_dict = {
        'id': np.array([p['id'] for p in all_properties], dtype=np.int64),
        'symbol': np.array(all_symbols, dtype=object),
        'coord': np.array(all_coords, dtype=object)
    }
    
    return data_dict, invalid_frames

def save_npz(data_dict: Dict[str, np.ndarray], output_file: str) -> str:
    """Save data to NPZ format with ordered labels."""
    try:
        # Define standard label order
        standard_labels = ['id', 'symbol', 'coord']
        
        # Save with label order
        save_dict = {label: data_dict[label] for label in standard_labels if label in data_dict}
        save_dict['__label_order__'] = np.array(standard_labels)
        
        np.savez(output_file, **save_dict)
        
        info = []
        info.append(f"Data saved to: {output_file}")
        info.append("Saved labels: " + ", ".join(standard_labels))
        info.append("\nData information:")
        for label in standard_labels:
            if label in save_dict:
                data = save_dict[label]
                shape = data.shape if hasattr(data, 'shape') else len(data)
                dtype = data.dtype if hasattr(data, 'dtype') else type(data)
                info.append(f"{label}: shape={shape}, dtype={dtype}")
                
        return "\n".join(info)
                
    except Exception as e:
        return f"Error saving data: {str(e)}"
