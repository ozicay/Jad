# Verified experiment results

[2026-10-10 results](2026-10-10/) cover six experiment groups:
gain depression/anxiety, separately retuned SHAP depression/anxiety, and
fixed-gain-parameter SHAP depression/anxiety.

- `six_groups_results.csv`: 72 reporting rows, including six explicitly marked
  reused gain All baselines. There are 66 actual runs, not 72 independent runs.
- `gain_shap_comparison.csv`: 24 rows comparing gain, retuned SHAP and fixed
  SHAP for each target/feature/K setting.
- Six per-group CSVs: 12 reporting rows each.
- `all_run_metrics_with_full_feature_controls.csv`: 120 actual final metric
  rows, preserving each Top-K run's ALL_FEATURES control as a separate row.
- `verification.json`: audit results, timestamp and numerical tolerance.

Each result records target, method, hyperparameter strategy, job ID, commit,
config and metric/prediction paths with SHA-256 hashes. Top-K summary values
are final retrain patient-level results, not Optuna objective values. All fixed
SHAP rows reuse the independently tuned gain All run and are marked accordingly.
Retuned SHAP and gain have different tuned params; only fixed SHAP versus its
matching gain Top-K isolates ranking method while holding params fixed.

All 66 jobs completed with exit 0:0, no detected training/trial errors, 120
patients per metric row and the expected model file counts. Patient identities
and labels agree within each target across runs. Fixed params match gain logs,
and fixed full-feature metrics match gain full-feature controls exactly at
saved precision. Metrics were independently recomputed from stored patient
prediction CSVs; these store predictions/labels at three decimals, so agreement
uses the explicit rounding bound. Official metrics are retained at six decimals.
Models were not reloaded or evaluated again. No individual patient data is
included in the exported reports.

Reproduce read-only collection on MN5 with `export_verified_results.py`, then
run `build_result_tables.py --input EXPORT_JSON --output DIRECTORY` on staging.
