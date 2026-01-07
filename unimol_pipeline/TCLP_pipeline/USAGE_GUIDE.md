# EPA ToxCast/tcpl Compliance Toxicity Labeling System - Usage Guide

##  File Structure

### `/code/` - Core Code
- `tcpl_system_final.py` - Complete tcpl compliance system (main code)
- `data_verification.py` - Data verification system
- `mapping_system.py` - Chemical mapping system

### `/data/` - Data Files
- `original/` - Original data
  - `processed_final8k213_original.csv` - Original 21st dataset (7,330 chemicals)
- `processed/` - Processed intermediate data
  - `tcpl_chemical_scores_final.csv` - Chemical tcpl scores detailed table
  - `cas_pubchem_mapping.json` - CAS to PUBCHEM_CID mapping
  - `chemical_bridge_table.csv` - chid to CAS mapping bridge table
- `final/` - Final results
  - `processed_final8k213_tcpl_labeled_final.csv` - **Main output**: final dataset with tcpl labels
  - `dataset_summary.json` - Dataset statistics summary

### `/validation/` - Validation Documents
- `external_validation_results.csv` - External validation results (ToxRefDB)
- `data_verification.json` - Data source verification report
- `data_integrity_sha256.json` - File integrity verification
- `reproducibility_report.json` - Reproducibility report (seeds, parameters, etc.)

### `/reports/` - Analysis Reports
- `compliance_assessment.md` - tcpl compliance assessment report
- `system_analysis.md` - System analysis report

## Main Results

### Data Coverage
- **Total Records**: 7,330 chemicals
- **tcpl Label Coverage**: 6,216 (84.8%)
- **External Validation**: 1,369 chemicals aligned with ToxRefDB

### Label Quality
- **Data Source**: EPA invitrodb v4.2 Summary Files
- **Hit Determination**: SC2 hitc∈{-1,0,1}, excluding hitc=-1
- **Quality Control**: mc6_flags artifact filtering + cytotox burst filtering (21,258 records)
- **Statistical Method**: Jeffreys Beta-Binomial shrinkage + mechanism equal-weight aggregation


## Usage Methods

### 1. Direct Use of Final Data
```python
import pandas as pd

# Load the final dataset with tcpl labels
data = pd.read_csv('data/final/processed_final8k213_tcpl_labeled_final.csv')

# View tcpl label columns
tcpl_columns = [col for col in data.columns if 'tcpl' in col.lower()]
print("Available tcpl label columns:", tcpl_columns)

# Filter chemicals with tcpl labels
labeled_data = data[data['tcpl_binary_compliant'] != -1]
print(f"Chemicals with tcpl labels: {len(labeled_data)}")
```

### 2. Re-run Complete System
```bash
cd code/
python tcpl_system_final.py
```

### 3. Verify Data Integrity
```bash
cd code/
python data_verification.py
```

## Label Description

### Binary Classification Label (`tcpl_binary_compliant`)
- `0`: Low toxicity
- `1`: High toxicity
- `-1`: No label

### Ternary Classification Label (`tcpl_ternary_compliant`)
- `0`: Low toxicity
- `1`: Medium toxicity
- `2`: High toxicity
- `-1`: No label

### Score Columns
- `S_c_tcpl_compliant`: tcpl score (range 0-1)
- `S_c_ci_lower_compliant`, `S_c_ci_upper_compliant`: 95% confidence interval
- `tcpl_n_tested_compliant`: Number of tested endpoints
- `tcpl_n_positive_compliant`: Number of positive endpoints

##  Quality Assurance

1. **Full tcpl Compliance**: Strictly implemented according to EPA ToxCast/tcpl standards
2. **Data Integrity**: SHA256 verification for all source files
3. **Reproducibility**: Fixed random seeds, complete parameter records
4. **External Validation**: Independent ToxRefDB validation, statistically significant
5. **Method Transparency**: Complete code and documentation

