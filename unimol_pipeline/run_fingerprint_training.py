from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from fingerprint_features import FingerprintConfig, load_fingerprint_dataset


def import_pipeline(module_path: Path):
    spec = importlib.util.spec_from_file_location('label_pipeline', str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    smiles_csv = base_dir / '8k21stoutput.csv'
    labels_csv = base_dir / '7100enhanced_optimized_labels_mapped_from_7330.csv'

    if not smiles_csv.exists():
        raise FileNotFoundError(f"SMILES source file missing: {smiles_csv}")
    if not labels_csv.exists():
        raise FileNotFoundError(f"Label dataset missing: {labels_csv}")

    config = FingerprintConfig(radius=2, n_bits=2048, use_chirality=True)
    dataset_df, metadata = load_fingerprint_dataset(
        smiles_csv=smiles_csv,
        labels_csv=labels_csv,
        config=config,
        include_physchem=True,
        include_label_numeric=True,
        include_smiles_numeric_extra=True,
    )

    dataset_output = base_dir / 'fingerprint_training_dataset.csv'
    dataset_df.to_csv(dataset_output, index_label='SampleName')

    metadata_output = base_dir / 'fingerprint_metadata.json'
    with open(metadata_output, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    pipeline_module = import_pipeline(base_dir / '01label2.py')
    pipeline_cls = pipeline_module.MLPipeline

    pipeline = pipeline_cls(
        output_dir=base_dir / 'fingerprint_results',
        target_column='y',
        drop_columns=None,
        feature_metadata=metadata,
        feature_selection={
            'drop_constant': True,
            'fp_min_support': 1,   # keep rare bits during review
            'corr_threshold': 1.0, # disable correlation dropping for transparency
            'l1_select': False,
        },
    )

    results = pipeline.run_pipeline(dataset_df)

    print('Training finished. Output directory:', pipeline.output_dir)
    print('Results head:\n', results.head())


if __name__ == '__main__':
    main()
