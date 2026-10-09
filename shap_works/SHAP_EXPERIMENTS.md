# SHAP configurable depression experiments

The new `optuna_xgb_configurable_shap.py` copies the gain configurable experiment.
Its only methodological change is the fold importance source: training-row
mean absolute TreeSHAP replaces XGBoost built-in importance. Every trial trains
its own full-feature LOPO models and recomputes SHAP on each model's `X_train`.
The held-out patient's segments are excluded from that fold's SHAP input.
The existing across-fold ranking and evaluation design is retained without
introducing nested selection or changing any baseline splits.

CSV loading/parsing, patient extraction/grouping, target `depresyon_skoru`,
LOPO masks, all XGBoost parameters, CUDA, 5 workers, random state 42, Optuna
search ranges and 20 trials, fold-mean objective, median prediction, and final
RMSE/MAE calculations are unchanged. Both selection stages use the same config
`top_k`. All mode skips trial ranking and reduced-feature retraining; its final
training-only SHAP analysis does not change its inputs or evaluation.

Each final full-feature fold saves a SHAP vector. The existing six-decimal
importance serialization/reload is retained for final selection. Ranking
artifacts are `shap_importance_all_models.csv`, `shap_importance_summary.csv`,
`shap_global_ranking.csv` (rank/feature/feature_index/mean_abs_shap), and
`selected_top5_features.csv`, `selected_top10_features.csv`, or
`selected_top15_features.csv`. Results, comparisons, all patient models,
`SHAP_OPTUNA_TMP_IMPORTANCE.csv` for Top-K, and
`final_patient_level_rmse_mae.csv` stay in the individual experiment directory.

Production locations under `/gpfs/projects/etur92/ozu150751/jad/shap_works`:

- `optuna_xgb_configurable_shap.py`: hash-verified copy of committed source.
- `shap_configs/<experiment>.json`: 12 independent configurations.
- `shap_exp_models/<experiment>/`: separate output directories.
- `shap_logs/shap_<experiment>_<job_id>.out` and `.err`: separate logs.
- `shap_scripts/run_experiment.slurm`, `submit_all.sh`, `audit_configs.py`.
- `source-shap-configurable/`: separate Git checkout recording source revision.

Slurm uses the same module Python environment, account etur92, QoS acc_ehpc,
partition acc, 1 GPU, 20 requested CPUs, 24 hours, and default memory as gain
experiments. No shared dependencies are changed.

Tests cover all 12 configurations with synthetic models and signed SHAP
values, prohibit gain usage, verify exact training-row SHAP input, per-trial
recomputation, ranking/counts, All mode, and a real CPU TreeExplainer smoke.
AST normalization confirms the complete training function matches gain after
only importance substitutions and additional artifact writes.
