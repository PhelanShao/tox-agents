"""
Molecular Descriptors Module

This module provides functions for calculating molecular descriptors
and replacing predicted values with calculated ones.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import RDKit for descriptor calculations
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available. Some descriptor calculations will be skipped.")


def calculate_basic_descriptors(mol) -> Dict[str, float]:
    """
    Calculate basic molecular descriptors using RDKit.
    
    Args:
        mol: RDKit molecule object.
        
    Returns:
        Dictionary of descriptor names to values.
    """
    if not RDKIT_AVAILABLE or mol is None:
        return {}
    
    descriptors = {}
    
    try:
        # Basic physical properties
        descriptors['MolWt'] = Descriptors.MolWt(mol)
        descriptors['LogP'] = Descriptors.MolLogP(mol)
        descriptors['TPSA'] = Descriptors.TPSA(mol)
        descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
        descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
        descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
        descriptors['NumAromaticRings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
        descriptors['NumHeavyAtoms'] = mol.GetNumHeavyAtoms()
        descriptors['FractionCSP3'] = rdMolDescriptors.CalcFractionCSP3(mol)
        
        # Ring information
        descriptors['NumRings'] = rdMolDescriptors.CalcNumRings(mol)
        descriptors['NumAliphaticRings'] = rdMolDescriptors.CalcNumAliphaticRings(mol)
        
        # Complexity
        descriptors['BertzCT'] = Descriptors.BertzCT(mol)
        
    except Exception as e:
        logger.warning(f"Error calculating descriptors: {e}")
    
    return descriptors


def calculate_descriptors_from_smiles(smiles: str) -> Dict[str, float]:
    """
    Calculate molecular descriptors from SMILES string.
    
    Args:
        smiles: SMILES string of the molecule.
        
    Returns:
        Dictionary of descriptor names to values.
    """
    if not RDKIT_AVAILABLE:
        return {}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Could not parse SMILES: {smiles}")
            return {}
        return calculate_basic_descriptors(mol)
    except Exception as e:
        logger.error(f"Error calculating descriptors for {smiles}: {e}")
        return {}


def calculate_descriptors_from_xyz(xyz_file: str, frame_idx: int = 0) -> Dict[str, float]:
    """
    Calculate molecular descriptors from XYZ file.
    
    Note: XYZ files don't contain bond information, so only geometry-based
    descriptors can be calculated reliably.
    
    Args:
        xyz_file: Path to XYZ file.
        frame_idx: Frame index for multi-frame XYZ files.
        
    Returns:
        Dictionary of descriptor names to values.
    """
    import numpy as np
    
    descriptors = {}
    
    try:
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
        
        # Parse XYZ file (may contain multiple frames)
        frames = []
        i = 0
        while i < len(lines):
            try:
                n_atoms = int(lines[i].strip())
                comment = lines[i + 1].strip() if i + 1 < len(lines) else ""
                atoms = []
                coords = []
                for j in range(i + 2, min(i + 2 + n_atoms, len(lines))):
                    parts = lines[j].split()
                    if len(parts) >= 4:
                        atoms.append(parts[0])
                        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                frames.append({
                    'n_atoms': n_atoms,
                    'comment': comment,
                    'atoms': atoms,
                    'coords': np.array(coords)
                })
                i += 2 + n_atoms
            except (ValueError, IndexError):
                i += 1
        
        if frame_idx < len(frames):
            frame = frames[frame_idx]
            coords = frame['coords']
            atoms = frame['atoms']
            
            # Geometry-based descriptors
            descriptors['NumAtoms'] = len(atoms)
            descriptors['NumHeavyAtoms'] = sum(1 for a in atoms if a != 'H')
            
            # Centroid
            centroid = np.mean(coords, axis=0)
            
            # Radius of gyration
            distances_to_centroid = np.linalg.norm(coords - centroid, axis=1)
            descriptors['RadiusOfGyration'] = np.sqrt(np.mean(distances_to_centroid ** 2))
            
            # Span (maximum distance between atoms)
            from scipy.spatial.distance import pdist
            if len(coords) > 1:
                pairwise_distances = pdist(coords)
                descriptors['MolecularSpan'] = np.max(pairwise_distances)
            else:
                descriptors['MolecularSpan'] = 0.0
            
            # Asphericity (from inertia tensor)
            if len(coords) > 2:
                centered = coords - centroid
                inertia = np.dot(centered.T, centered) / len(coords)
                eigenvalues = np.linalg.eigvalsh(inertia)
                eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
                if eigenvalues[0] > 0:
                    descriptors['Asphericity'] = eigenvalues[0] - 0.5 * (eigenvalues[1] + eigenvalues[2])
                else:
                    descriptors['Asphericity'] = 0.0
            
    except Exception as e:
        logger.error(f"Error reading XYZ file {xyz_file}: {e}")
    
    return descriptors


def replace_predicted_with_calculated(
    properties: List[Dict[str, Any]], 
    xyz_file: str
) -> List[Dict[str, Any]]:
    """
    Replace predicted property values with calculated ones where available.
    
    This function takes a list of property dictionaries (from model predictions)
    and replaces values for basic descriptors with actually calculated values
    from the molecular structure.
    
    Args:
        properties: List of property dictionaries with 'name' and 'value' keys.
        xyz_file: Path to XYZ file for calculating descriptors.
        
    Returns:
        Updated list of property dictionaries.
    """
    # Calculate descriptors from XYZ
    calculated = calculate_descriptors_from_xyz(xyz_file, frame_idx=0)
    
    if not calculated:
        return properties
    
    # Mapping from property names to calculated descriptor names
    name_mapping = {
        'MolWt': 'MolWt',
        'Molecular Weight': 'MolWt',
        'LogP': 'LogP',
        'TPSA': 'TPSA',
        'NumHDonors': 'NumHDonors',
        'H-Bond Donors': 'NumHDonors',
        'NumHAcceptors': 'NumHAcceptors',
        'H-Bond Acceptors': 'NumHAcceptors',
        'NumRotatableBonds': 'NumRotatableBonds',
        'Rotatable Bonds': 'NumRotatableBonds',
        'NumAtoms': 'NumAtoms',
        'Atom Count': 'NumAtoms',
        'NumHeavyAtoms': 'NumHeavyAtoms',
        'Heavy Atom Count': 'NumHeavyAtoms',
        'RadiusOfGyration': 'RadiusOfGyration',
        'Radius of Gyration': 'RadiusOfGyration',
        'MolecularSpan': 'MolecularSpan',
        'Molecular Span': 'MolecularSpan',
    }
    
    for prop in properties:
        prop_name = prop.get('name', '')
        calc_key = name_mapping.get(prop_name)
        if calc_key and calc_key in calculated:
            prop['value'] = calculated[calc_key]
            prop['source'] = 'calculated'
    
    return properties


def replace_predicted_with_calculated_for_frame(
    properties: List[Dict[str, Any]], 
    xyz_file: str,
    frame_idx: int
) -> List[Dict[str, Any]]:
    """
    Replace predicted property values with calculated ones for a specific frame.
    
    Args:
        properties: List of property dictionaries.
        xyz_file: Path to XYZ file.
        frame_idx: Frame index in multi-frame XYZ file.
        
    Returns:
        Updated list of property dictionaries.
    """
    calculated = calculate_descriptors_from_xyz(xyz_file, frame_idx=frame_idx)
    
    if not calculated:
        return properties
    
    name_mapping = {
        'NumAtoms': 'NumAtoms',
        'Atom Count': 'NumAtoms',
        'NumHeavyAtoms': 'NumHeavyAtoms',
        'Heavy Atom Count': 'NumHeavyAtoms',
        'RadiusOfGyration': 'RadiusOfGyration',
        'Radius of Gyration': 'RadiusOfGyration',
        'MolecularSpan': 'MolecularSpan',
        'Molecular Span': 'MolecularSpan',
        'Asphericity': 'Asphericity',
    }
    
    for prop in properties:
        prop_name = prop.get('name', '')
        calc_key = name_mapping.get(prop_name)
        if calc_key and calc_key in calculated:
            prop['value'] = calculated[calc_key]
            prop['source'] = 'calculated'
    
    return properties


if __name__ == '__main__':
    # Test descriptor calculation
    if RDKIT_AVAILABLE:
        test_smiles = "CCO"  # Ethanol
        descriptors = calculate_descriptors_from_smiles(test_smiles)
        print(f"Descriptors for {test_smiles}:")
        for name, value in descriptors.items():
            print(f"  {name}: {value:.4f}")
    else:
        print("RDKit not available for testing")
