from typing import Dict, List, Union, Any
import numpy as np
from tqdm import tqdm

class NPZToXYZ:
    def __init__(self, npz_file: str):
        """Initialize with path to NPZ file"""
        self.npz_file = npz_file
        self.data = np.load(npz_file, allow_pickle=True)
        
    def _format_property_line(self, properties: Dict[str, Any], idx: int) -> str:
        """Format the property line for a single structure"""
        prop_strings = []
        for key in properties:
            if key not in ['symbol', 'coord', '__label_order__']:
                value = properties[key][idx]
                if isinstance(value, (int, np.integer)):
                    prop_strings.append(f"{key}:{int(value)}")
                elif isinstance(value, (float, np.floating)):
                    prop_strings.append(f"{key}:{value:.6f}")
                elif value is not None:
                    prop_strings.append(f"{key}:{str(value)}")
        return " ".join(prop_strings)

    def _format_coordinates(self, symbols: List[str], coords: List[List[float]]) -> str:
        """Format atomic coordinates for a single structure"""
        lines = []
        for symbol, (x, y, z) in zip(symbols, coords):
            lines.append(f"{symbol:<2} {x:>12.6f} {y:>12.6f} {z:>12.6f}")
        return "\n".join(lines)

    def convert(self, output_file: str):
        """Convert NPZ to XYZ format"""
        try:
            # Get array length from coord array
            n_structures = len(self.data['coord'])
            
            # Create properties dictionary excluding special keys
            properties = {k: self.data[k] for k in self.data.files 
                        if k not in ['symbol', 'coord', '__label_order__']}
            
            with open(output_file, 'w') as f:
                for i in tqdm(range(n_structures), desc="Converting structures"):
                    # Get current structure data
                    symbols = self.data['symbol'][i]
                    coords = self.data['coord'][i]
                    n_atoms = len(symbols)
                    
                    # Write number of atoms
                    f.write(f"{n_atoms}\n")
                    
                    # Write properties line
                    prop_line = self._format_property_line(properties, i)
                    f.write(f"{prop_line}\n")
                    
                    # Write coordinates
                    f.write(self._format_coordinates(symbols, coords) + "\n")
            
            return f"Successfully converted {n_structures} structures to: {output_file}"
            
        except Exception as e:
            raise Exception(f"Error during conversion: {str(e)}")
