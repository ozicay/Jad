# EDAIC SHAP with separately retuned Optuna parameters

Original EDAIC population: 233 patients, 4829 segments, depresyon_skoru.
Uses existing ../metadata files; never the random-five subset.
MFCC608/eGeMAPS88/IS09384, each Top5/10/15/ALL: twelve configurations.

Single node, four H100 GPUs. Four isolated Python processes run independent
configs; each process sees exactly one GPU as cuda:0. Each config has sequential
LOPO folds and 20 sequential COMPLETE Optuna trials. When one config completes,
the free GPU receives the next config in the canonical list. Completion order
can differ. Trials are retuned separately; this is method 2, not fixed gain params.

Ranking replaces gain with mean absolute TreeSHAP, calculated only on each
full-feature LOPO model's X_train, then averaged across folds. The helper is
copied unchanged from shap_works/optuna_xgb_configurable_shap.py. Trial Top-K
ranking uses the original CSV roundtrip; final ranking retains six-decimal CSV
roundtrip. Top-K models do not compute SHAP. ALL trials do not select features;
final ALL SHAP is diagnostic only. Gain vectors retained for Top-K diagnostics
are never used for feature selection. No nested CV or new evaluation split.

Original EDAIC parsing/IDs/masks, objective, Optuna search ranges, final params,
segment predictions, patient median and RMSE/MAE are preserved. Only one final
best-fold model is saved per Top-K config, none for ALL. Every fold prediction
and segment prediction is retained. Existing experiments are not overwritten.

Validation: python run_shap_configs.py --validate-only
Submit: sbatch run_shap.slurm
Logs: logs/<config>.log and logs/slurm_<job>.out/.err.
Summary: experiment_summary.csv; population: patient_fold_index.csv.
Sources/config/data hashes: SHA256SUMS, run_manifest.json and per-config records.
Reuse existing MN5 runtime; no package installation. This group changes both
ranking and tuned hyperparameters; fixed-parameter SHAP would be another group.
