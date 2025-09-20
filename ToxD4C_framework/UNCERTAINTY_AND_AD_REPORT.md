# Uncertainty, Calibration, and Applicability Domain (AD)

This repository now includes first-class support for uncertainty quantification, calibration assessment, and applicability domain (AD) analysis, addressing reviewer requests for risk-aware predictions.

What’s included

- Classification calibration: reliability curves, Expected Calibration Error (ECE), and Brier score.
- Prediction uncertainty: Test-Time Augmentation (TTA) with coordinate perturbations and optional MC-Dropout; reports per-sample prediction variance for classification and regression.
- AD in embedding space: Mahalanobis distance in the model’s graph representation; threshold defined via a training-embedding percentile (default p95) and OOD flags per sample.
- Optional similarity-based AD: use `tools/check_nonoverlap_and_similarity.py` for ECFP4 nearest-neighbor Tanimoto against the training set (for external sets and case studies).

How to run

1) Evaluate and export detailed per-endpoint metrics with uncertainty and AD:

```
python ToxD4C/evaluate_detailed_endpoints.py \
  --experiment_dir ToxD4C/experiments/your_experiment_dir \
  --data_dir data/data/processed \
  --batch_size 16 \
  --split test \
  --with_uncertainty \
  --tta_runs 8 \
  --tta_noise_std 0.02 \
  --with_ad \
  --ad_threshold_percentile 95 \
  --save_plots
```

Outputs (under `<experiment_dir>/checkpoints`):

- `detailed_endpoint_results.json` (or user-defined name): now includes
  - per-endpoint ECE and Brier (classification)
  - calibration curves (first up to 6 endpoints) under `calibration_curves`
  - per-sample average prediction variance under `uncertainty`
  - embedding AD report under `applicability_domain` with Mahalanobis thresholds and OOD flags
- `calibration_curves_<split>.png`: grid of reliability diagrams
- `ad_mahalanobis_<split>.png`: histogram of Mahalanobis distances with the threshold

Notes and assumptions

- TTA: coordinates are perturbed by Gaussian noise (default σ=0.02 Å). Increase `--tta_runs` for more stable variance and mean estimates. Add `--mc_dropout` to enable MC-Dropout during TTA.
- Calibration: ECE uses 15 equal-width bins in [0,1]. Brier is standard binary Brier score.
- AD: Mahalanobis distance is computed in the model’s latent graph representation; the threshold defaults to the 95th percentile of training distances (configurable). Samples above the threshold are flagged OOD. This complements descriptor/similarity-based AD (see below).
- Similarity-based AD: for external sets (e.g., transformation products), use

```
python ToxD4C/tools/check_nonoverlap_and_similarity.py \
  --lmdb_dir data/data/processed \
  --challenge_dir path/to/external_smiles \
  --output_dir tox21_overlap_check \
  --compute_similarity
```

This produces nearest-neighbor ECFP4 Tanimoto similarities and summary stats, which can be used as an orthogonal AD criterion. Lower similarity or high Mahalanobis distance indicates out-of-domain chemistry.

Recommended reporting

- Report calibration curves, ECE, and Brier at least for representative endpoints.
- Report TTA prediction variances as uncertainty bands; prioritize high-variance predictions for expert review.
- For AOP case studies, mark transformation products OOD if they fall below a similarity threshold and/or exceed the Mahalanobis threshold; summarize their predictions with variance.

