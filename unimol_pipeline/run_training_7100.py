import os
from pathlib import Path
import pandas as pd
import importlib.util


def import_pipeline(module_path: Path):
    spec = importlib.util.spec_from_file_location("label_pipeline", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main():
    base_dir = Path(__file__).parent
    data_csv = base_dir / "7100enhanced_optimized_labels_mapped_from_7330.csv"
    assert data_csv.exists(), f"Data file not found: {data_csv}"

    pipeline_mod = import_pipeline(base_dir / "01label2.py")

    output_prefix = base_dir / "7100ml_results_mapped"

    pipeline = pipeline_mod.MLPipeline(output_dir=output_prefix)
    results = pipeline.run_pipeline(data_csv)
    print("Training finished. Output dir prefix:", output_prefix)
    print("Results head:\n", results.head())


if __name__ == "__main__":
    main()



