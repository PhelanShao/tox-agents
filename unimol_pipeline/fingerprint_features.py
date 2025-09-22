from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog('rdApp.warning')


@dataclass
class FingerprintConfig:
    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = True


class FingerprintDatasetBuilder:
    """Utility for constructing fingerprint feature matrices and metadata."""

    def __init__(self, config: Optional[FingerprintConfig] = None) -> None:
        self.config = config or FingerprintConfig()

    def _compute_fingerprint(
        self, smiles: str
    ) -> Tuple[Optional[Chem.Mol], Optional[np.ndarray], Dict[int, List[Tuple[int, int]]]]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, {}

        bit_info: Dict[int, List[Tuple[int, int]]] = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            self.config.radius,
            nBits=self.config.n_bits,
            useChirality=self.config.use_chirality,
            bitInfo=bit_info,
        )

        arr = np.zeros((self.config.n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return mol, arr, bit_info

    def build_dataset(
        self,
        smiles_csv: Path,
        labels_csv: Path,
        include_physchem: bool = True,
        include_label_numeric: bool = False,
        include_smiles_numeric_extra: bool = False,
        label_numeric_include: Optional[List[str]] = None,
        label_numeric_exclude: Optional[List[str]] = None,
        drop_na_smiles: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
        smiles_df = pd.read_csv(smiles_csv)
        labels_df = pd.read_csv(labels_csv)

        smiles_df['SampleName_x'] = smiles_df['SampleName_x'].astype(str)
        labels_df['SampleName_x'] = labels_df['SampleName_x'].astype(str)

        merged = labels_df.merge(smiles_df, on='SampleName_x', how='left', suffixes=('', '_smiles'))
        merged = merged.drop_duplicates(subset=['SampleName_x'])
        if drop_na_smiles:
            merged = merged.dropna(subset=['CanonicalSMILES'])

        sample_ids = merged['SampleName_x'].astype(str)
        smiles_values = merged['CanonicalSMILES'].astype(str)

        n_samples = len(merged)
        fp_matrix = np.zeros((n_samples, self.config.n_bits), dtype=np.uint8)

        failed_samples: List[str] = []
        sample_smiles: Dict[str, str] = {}
        bit_support = np.zeros(self.config.n_bits, dtype=np.int64)

        for idx, (sample_id, smiles) in enumerate(zip(sample_ids, smiles_values)):
            mol, fp_arr, bit_info = self._compute_fingerprint(smiles)
            if mol is None or fp_arr is None:
                failed_samples.append(sample_id)
                continue

            fp_matrix[idx] = fp_arr
            sample_smiles[sample_id] = smiles

            active_bits = np.nonzero(fp_arr)[0]
            bit_support[active_bits] += 1

        valid_mask = np.array([sid not in failed_samples for sid in sample_ids], dtype=bool)
        fp_matrix = fp_matrix[valid_mask]
        merged = merged.loc[valid_mask].copy()
        merged.index = merged['SampleName_x'].astype(str)

        column_names = [f'FP_{bit:04d}' for bit in range(self.config.n_bits)]
        fp_df = pd.DataFrame(fp_matrix, columns=column_names, index=merged.index)

        feature_frames = [fp_df]

        added_physchem_cols: List[str] = []
        if include_physchem:
            # Base physchem subset from SMILES source
            physchem_columns = [
                'MolecularWeight',
                'XLogP',
                'ExactMass',
                'MonoisotopicMass',
                'TPSA',
                'Complexity',
                'HBondDonorCount',
                'HBondAcceptorCount',
                'RotatableBondCount',
            ]

            available = [col for col in physchem_columns if col in merged.columns]
            if available:
                physchem_df = merged[available].copy()
                physchem_df.index = merged.index
                feature_frames.append(physchem_df)
                added_physchem_cols.extend(available)

        # Optionally include ALL other numeric columns from SMILES table
        added_smiles_numeric_extra: List[str] = []
        if include_smiles_numeric_extra:
            smiles_numeric = smiles_df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_smiles = set([
                'y', 'SampleName_x', 'SampleName_y',
            ])
            # remove already added physchem
            exclude_smiles.update(added_physchem_cols)
            smiles_to_add = [c for c in smiles_numeric if c not in exclude_smiles]
            if smiles_to_add:
                extra_smiles_df = merged[smiles_to_add].copy()
                extra_smiles_df.index = merged.index
                feature_frames.append(extra_smiles_df)
                added_smiles_numeric_extra.extend(smiles_to_add)

        # Optionally include numeric columns from labels dataset (quantum etc.)
        added_label_numeric_cols: List[str] = []
        if include_label_numeric:
            label_numeric = labels_df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_labels = set([
                'y', 'SampleName_x', 'SampleName_y'
            ])
            # avoid duplicating physchem columns
            exclude_labels.update(added_physchem_cols)
            # apply include/exclude lists
            if label_numeric_include:
                label_numeric = [c for c in label_numeric if c in set(label_numeric_include)]
            if label_numeric_exclude:
                exclude_labels.update(label_numeric_exclude)
            label_to_add = [c for c in label_numeric if c not in exclude_labels]
            if label_to_add:
                extra_label_df = merged[label_to_add].copy()
                extra_label_df.index = merged.index
                feature_frames.append(extra_label_df)
                added_label_numeric_cols.extend(label_to_add)

        feature_df = pd.concat(feature_frames, axis=1)

        if 'y' not in merged.columns:
            raise ValueError("Label column 'y' not found in labels dataset after merging")

        feature_df['y'] = merged['y']

        metadata: Dict[str, Dict] = {
            'fingerprint': {
                'radius': self.config.radius,
                'n_bits': self.config.n_bits,
                'use_chirality': self.config.use_chirality,
                'column_to_bit': {col: idx for idx, col in enumerate(column_names)},
                'bit_to_column': {idx: col for idx, col in enumerate(column_names)},
                'bit_support': {int(idx): int(count) for idx, count in enumerate(bit_support) if count > 0},
                'sample_smiles': sample_smiles,
                'failed_samples': failed_samples,
            },
            'extra_features': {
                'physchem': added_physchem_cols,
                'smiles_numeric_extra': added_smiles_numeric_extra,
                'label_numeric': added_label_numeric_cols,
            }
        }

        return feature_df, metadata


def load_fingerprint_dataset(
    smiles_csv: Path,
    labels_csv: Path,
    config: Optional[FingerprintConfig] = None,
    include_physchem: bool = True,
    include_label_numeric: bool = False,
    include_smiles_numeric_extra: bool = False,
    label_numeric_include: Optional[List[str]] = None,
    label_numeric_exclude: Optional[List[str]] = None,
    drop_na_smiles: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    builder = FingerprintDatasetBuilder(config)
    return builder.build_dataset(
        smiles_csv,
        labels_csv,
        include_physchem=include_physchem,
        include_label_numeric=include_label_numeric,
        include_smiles_numeric_extra=include_smiles_numeric_extra,
        label_numeric_include=label_numeric_include,
        label_numeric_exclude=label_numeric_exclude,
        drop_na_smiles=drop_na_smiles,
    )
