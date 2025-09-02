# EPA ToxCast/tcpl Data Sources

This directory contains the original EPA invitrodb v4.2 Summary Files used for tcpl-compliant toxicity labeling.

## Data Source Files

### Primary EPA invitrodb v4.2 Summary Files

1. **sc1_sc2_invitrodb_v4_2_SEPT2024.xlsx**
   - Source: EPA ToxCast/Tox21 Summary Files
   - Purpose: Official hit call determination (hitc values)
   - Key fields: chid, aeid, hitc, casn
   - Usage: Primary source for tested/positive determination
   - Size: ~510,971 records after filtering

2. **cytotox_invitrodb_v4_2_SEPT2024.xlsx**
   - Source: EPA ToxCast/Tox21 Summary Files
   - Purpose: Chemical-specific cytotoxicity thresholds
   - Key fields: chid, cytotox_lower_bound_log
   - Usage: Cytotoxicity burst filtering (Δ=3 log10 units)
   - Size: ~9,614 chemical records

3. **assay_annotations_invitrodb_v4_2_SEPT2024.xlsx**
   - Source: EPA ToxCast/Tox21 Summary Files
   - Purpose: Assay endpoint metadata and mechanism mapping
   - Key fields: aeid, intended_target_family, biological_process_target
   - Usage: Endpoint to mechanism classification (NR, SR, DDR, CYTO, etc.)
   - Size: ~1,570 assay endpoints

4. **MC5-6 File (Large File - Not Included)**
   - File: mc5-6_winning_model_fits-flags_invitrodbv4_2_SEPT2024.csv
   - Source: EPA ToxCast/Tox21 Summary Files
   - Purpose: Artifact flags (mc6_flags) and potency values (ac50)
   - Key fields: chid, aeid, mc6_flags, ac50, conc_unit
   - Usage: Platform/multi-channel artifact filtering and concentration standardization
   - Size: ~3.3 million records (too large for repository)
   - Note: Available from EPA ToxCast Data Downloads

### External Validation Data

5. **tox21_toxrefdb_matched_via_cas.csv**
   - Source: ToxRefDB v2 matched to ToxCast chemicals via CAS numbers
   - Purpose: External validation using in vivo POD values
   - Key fields: CAS_NORM, POD_MGKGDAY, PUBCHEM_CID
   - Usage: External anchoring for threshold selection and validation
   - Size: ~1,414 chemical-endpoint combinations

## Data Provenance and Integrity

### File Integrity Verification
All data files include SHA256 hash verification for integrity checking:
- SC2: 29a2b8298d0ec748c63cca5463e2259dcbaffbacf96f2d0191aafc785a8c22d8
- Cytotox: c85bf754274f491b71f24e3898d196a1f0280cc0b85a0a5eddcd1e2e290e10cd
- Assay annotations: 50280fb0da2a31cfa80456b22d2f56c641669171cabec0715ff2210888000c24
- ToxRefDB: f2aa6f4a1133ea95b9217ed215ecbca88b690a443507a73757d7d15fc31169cc

### Data Source URLs
- EPA ToxCast Data: https://www.epa.gov/chemical-research/toxicity-forecaster-toxcasttm-data
- ToxRefDB: https://www.epa.gov/chemical-research/toxicity-reference-database-toxrefdb
- invitrodb v4.2: https://doi.org/10.23645/epacomptox.6062623.v4

## tcpl Compliance Standards

### Hit Call Determination
- **Primary source**: SC2 hitc values only
- **Valid values**: {-1, 0, 1}
- **Exclusions**: hitc = -1 (undefined/not applicable) excluded from denominator
- **Tested definition**: Record exists and hitc ≠ -1
- **Positive definition**: hitc = 1

### Quality Control Filters
1. **Artifact Control**: mc6_flags from MC5-6 data
2. **Cytotoxicity Filtering**: cytotox_lower_bound_log with Δ=3 log10 units
3. **Concentration Standardization**: All ac50 values converted to μM

### Statistical Methods
- **Shrinkage**: Jeffreys Beta-Binomial (α=β=0.5)
- **Aggregation**: Mechanism-weighted equal averaging
- **Confidence Intervals**: Mechanism-layer bootstrap for consistency
- **Threshold Selection**: 5×3 nested cross-validation with Youden's J

### External Validation
- **Reference**: ToxRefDB POD values
- **Thresholds**: τ ∈ {3, 10, 30} mg/kg/day
- **Metrics**: Spearman correlation, ROC-AUC with bootstrap 95% CI
- **Direction Check**: Use -S_c if AUC < 0.5

## Usage Notes

1. **Large File Handling**: The MC5-6 file (~3.3M records) requires chunked processing
2. **Memory Requirements**: Full pipeline requires ~8GB RAM for complete processing
3. **Processing Time**: Complete pipeline takes ~15-30 minutes depending on hardware
4. **Reproducibility**: All random seeds fixed (seed=42) for reproducible results

## Regulatory Compliance

This implementation follows EPA ToxCast/tcpl standards as documented in:
- Filer et al. (2017). tcpl: the ToxCast pipeline for high-throughput screening data. Bioinformatics.
- EPA ToxCast & Tox21 Summary Files Documentation
- EPA Chemical Safety for Sustainability Research Program guidelines

For questions regarding data sources or compliance, refer to EPA ToxCast documentation or contact the EPA Chemical Safety for Sustainability Research Program.
