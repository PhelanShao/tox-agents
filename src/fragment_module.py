#fragment_module.py
import os
import re
import plotly.graph_objs as go
import zipfile
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
import numpy as np
from typing import Dict, List

logger = logging.getLogger(__name__)

def extract_data(log_file_content, keyword_pairs_columns):
    """Extract data from CP2K log file"""
    data = []
    first_frame_data = []
    lines = log_file_content.split('\n')
    
    md_init_pattern = re.compile(r'MD_INI\| MD initialization')
    step_number_pattern = re.compile(r'MD\| Step number\s+(\d+)')

    step_number = None
    first_frame_found = False

    def extract_values(line, columns):
        values = line.split()
        return [values[int(col)-1] if int(col)-1 < len(values) else 'N/A' for col in columns]

    def search_within_range(start, end, k1, k2, columns):
        keyword1_index = -1
        for j in range(start, end):
            if re.search(k1, lines[j]):
                keyword1_index = j
                break
        if keyword1_index == -1:
            return ['N/A'] * len(columns)
        for j in range(keyword1_index, end):
            if re.search(k2, lines[j]):
                return extract_values(lines[j], columns)
        return ['N/A'] * len(columns)

    for i, line in enumerate(lines):
        if md_init_pattern.search(line) and not first_frame_found:
            first_frame_data = ['1']
            for k1, k2, columns in keyword_pairs_columns:
                first_frame_data += search_within_range(0, len(lines), k1, k2, columns)
            data.append(first_frame_data)
            first_frame_found = True

        step_match = step_number_pattern.search(line)
        if step_match:
            step_number = int(step_match.group(1))
            next_step_index = None

            for j in range(i + 1, len(lines)):
                if step_number_pattern.search(lines[j]):
                    next_step_index = j
                    break
            if next_step_index is None:
                next_step_index = len(lines)

            current_row = [str(step_number)]
            for k1, k2, columns in keyword_pairs_columns:
                current_row += search_within_range(i, next_step_index, k1, k2, columns)
            data.append(current_row)

    result = '\n'.join([' '.join(row) for row in data])
    return result, data

def plot_extracted_data(data):
    """Create plot from extracted data"""
    step_numbers = [int(row[0]) for row in data]
    traces = []
    for col_idx in range(1, len(data[0])):
        y_values = [float(row[col_idx]) if row[col_idx] != 'N/A' else None for row in data]
        trace_name = f'Column {col_idx}'
        traces.append(go.Scatter(x=step_numbers, y=y_values, mode='lines', name=trace_name))

    fig = go.Figure(data=traces)
    fig.update_layout(title='Extracted Data Plot', xaxis_title='Step Number', yaxis_title='Values')
    return fig

def parse_species_file(species_file):
    """Parse species file to extract molecular fragments"""
    species_data = {}
    current_timestep = None
    
    with open(species_file, 'r') as f:
        for line in f:
            if line.startswith('Timestep'):
                try:
                    parts = line.split(':')
                    current_timestep = int(parts[0].split()[1])
                    species_data[current_timestep] = []
                    
                    if len(parts) > 1:
                        fragments_part = parts[1].strip()
                        if fragments_part:
                            fragments = []
                            current_fragment = []
                            in_brackets = False
                            
                            for char in fragments_part:
                                if char == '[':
                                    in_brackets = True
                                    current_fragment.append(char)
                                elif char == ']':
                                    in_brackets = False
                                    current_fragment.append(char)
                                elif char.isspace() and not in_brackets:
                                    if current_fragment:
                                        fragments.append(''.join(current_fragment))
                                        current_fragment = []
                                else:
                                    current_fragment.append(char)
                            
                            if current_fragment:
                                fragments.append(''.join(current_fragment))
                            
                            species_data[current_timestep].extend(
                                frag for frag in fragments 
                                if any(c in frag for c in '[]()CHON=')
                            )
                except Exception as e:
                    logger.error(f"Error parsing line: {line}")
                    logger.error(f"Error details: {str(e)}")
                    continue
    
    return species_data

def create_species_timeline(species_data):
    """Create timeline visualization of molecular fragments"""
    all_species = sorted(set(fragment for fragments in species_data.values() for fragment in fragments))
    timesteps = sorted(species_data.keys())
    species_presence = {species: [] for species in all_species}
    
    offset = 0.1
    offsets = {species: i * offset for i, species in enumerate(all_species)}
    
    for t in timesteps:
        for species in all_species:
            value = 1 + offsets[species] if species in species_data[t] else offsets[species]
            species_presence[species].append(value)
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    line_styles = ['solid', 'dot', 'dash', 'dashdot', 'longdash']
    
    for i, species in enumerate(all_species):
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=species_presence[species],
            name=species,
            mode='lines',
            line=dict(
                color=color,
                width=2,
                dash=line_style
            ),
            hovertemplate=f"Fragment: {species}<br>Time: %{{x}}<br>Present: %{{y}}<extra></extra>"
        ))
    
    fig.update_layout(
        title='Molecular Fragment Distribution Over Time',
        xaxis_title='Timestep',
        yaxis_title='Fragment Presence',
        showlegend=True,
        height=800,  # 增加高度以适应底部图例
        hovermode='x unified',
        yaxis=dict(
            tickmode='array',
            ticktext=['Absent', 'Present'],
            tickvals=[0, 1],
            range=[-0.1, 1 + offset * (len(all_species) + 1)]
        ),
        legend=dict(
            orientation="h",     # 水平排列图例
            yanchor="top",      # 图例顶部对齐
            y=-0.5,             # 移动到图表下方
            xanchor="center",   # 水平居中对齐
            x=0.5,              # 水平居中位置
            font=dict(size=10)  # 调整字体大小
        ),
        margin=dict(b=200)      # 增加底部边距以容纳图例
    )
    
    return fig

def get_fragment_type(smiles):
    """判断片段类型(自由基/离子/分子)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unknown"
        
    total_valence = 0
    total_bonds = 0
    has_radical = False
    charge = 0
    
    for atom in mol.GetAtoms():
        charge += atom.GetFormalCharge()
        if atom.GetNumRadicalElectrons() > 0:
            has_radical = True
            
        valence = 0
        if atom.GetSymbol() == 'H':
            valence = 1
        elif atom.GetSymbol() == 'C':
            valence = 4
        elif atom.GetSymbol() == 'O':
            valence = 2
        elif atom.GetSymbol() == 'N':
            valence = 3
            
        total_valence += valence
        total_bonds += len(atom.GetBonds())
    
    if has_radical:
        return "radical"
    elif charge != 0:
        return "ion"
    else:
        return "molecule"

def extract_coordinates(job_id, timestep, selected_fragments):
    """Extract coordinates for selected fragments"""
    try:
        from openbabel import openbabel
    except ImportError:
        raise ImportError("Open Babel 3.1.0 is required.")

    xyz_file = os.path.join(job_id, "B-pos-1.xyz")
    
    coords = []
    atoms = []
    current_step = -1
    n_atoms = None
    
    with open(xyz_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
                
            if n_atoms is None:
                n_atoms = int(line.strip())
                continue
                
            if "time" in line:
                current_step = int(line.split()[2].strip(','))
                if current_step == timestep:
                    for _ in range(n_atoms):
                        atom_line = f.readline().strip().split()
                        atoms.append(atom_line[0])
                        coords.append([float(x) for x in atom_line[1:4]])
                    break
                else:
                    for _ in range(n_atoms):
                        f.readline()
    
    if not coords:
        return None
        
    coords = np.array(coords)
    
    mol = openbabel.OBMol()
    mol.BeginModify()
    
    for idx, (atom_symbol, position) in enumerate(zip(atoms, coords)):
        atom = mol.NewAtom(idx)
        atom.SetAtomicNum(openbabel.GetAtomicNum(atom_symbol))
        atom.SetVector(*position)
    
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()
    mol.EndModify()
    
    bonds = []
    for bond in openbabel.OBMolBondIter(mol):
        begin_idx = bond.GetBeginAtom().GetId()
        end_idx = bond.GetEndAtom().GetId()
        bond_order = bond.GetBondOrder()
        if bond_order == 5:  # 芳香键
            bond_order = 12
        bonds.append((begin_idx, end_idx, bond_order))
    
    def dfs(v, visited, component):
        visited[v] = True
        component.append(v)
        for bond in bonds:
            if bond[0] == v and not visited[bond[1]]:
                dfs(bond[1], visited, component)
            elif bond[1] == v and not visited[bond[0]]:
                dfs(bond[0], visited, component)
    
    visited = [False] * len(atoms)
    molecular_fragments = []
    
    for i in range(len(atoms)):
        if not visited[i]:
            component = []
            dfs(i, visited, component)
            if component:
                molecular_fragments.append(component)
    
    output_files = []
    for i, fragment in enumerate(molecular_fragments):
        fragment_mol = openbabel.OBMol()
        fragment_mol.BeginModify()
        atom_map = {}
        
        for idx, atom_idx in enumerate(fragment):
            atom = fragment_mol.NewAtom(idx)
            atom.SetAtomicNum(openbabel.GetAtomicNum(atoms[atom_idx]))
            atom.SetVector(*coords[atom_idx])
            atom_map[atom_idx] = idx
        
        for bond in bonds:
            if bond[0] in fragment and bond[1] in fragment:
                fragment_mol.AddBond(
                    atom_map[bond[0]]+1, 
                    atom_map[bond[1]]+1, 
                    bond[2]
                )
        
        fragment_mol.EndModify()
        
        conv = openbabel.OBConversion()
        conv.SetOutFormat("smi")
        smiles = conv.WriteString(fragment_mol).strip()
        
        fragment_type = get_fragment_type(smiles)
        
        output_file = os.path.join(job_id, f"fragment_{i+1}_{fragment_type}_step_{timestep}.xyz")
        with open(output_file, 'w') as f:
            f.write(f"{len(fragment)}\n")
            f.write(f"Fragment {i+1} ({fragment_type}) from timestep {timestep}\n")
            for atom_idx in fragment:
                f.write(f"{atoms[atom_idx]} {coords[atom_idx][0]:.6f} {coords[atom_idx][1]:.6f} {coords[atom_idx][2]:.6f}\n")
        output_files.append(output_file)
    
    return output_files

def merge_xyz_files(output_files):
    """Merge multiple XYZ files into one"""
    merged_content = []
    radical_content = []
    molecule_content = []
    
    for xyz_file in output_files:
        with open(xyz_file, 'r') as f:
            content = f.readlines()
            merged_content.extend(content)
            
            if '_radical_' in xyz_file:
                radical_content.extend(content)
            elif '_molecule_' in xyz_file:
                molecule_content.extend(content)
            
    return merged_content, radical_content, molecule_content

def find_first_occurrence(species_data):
    """Find first occurrence of each fragment"""
    first_occurrence = {}
    all_species = set()
    
    for timestep in sorted(species_data.keys()):
        for fragment in species_data[timestep]:
            if fragment not in first_occurrence:
                first_occurrence[fragment] = timestep
            all_species.add(fragment)
    
    return first_occurrence
