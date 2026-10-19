# EDAIC controlled SHAP: fixed gain best parameters

Method 3: nine Top-K configurations; no Optuna import, search or trials.
233 patients, 4829 original EDAIC segments, target depresyon_skoru.
Each matching gain config supplies its own best learning_rate/max_depth/n_estimators.
Registry entries were checked against all 20 completed original gain trials and
final summary. Baseline file and dataset hashes are verified before execution.

Full-feature LOPO with fixed params -> mean absolute training-only TreeSHAP
per fold -> average across folds -> six-decimal CSV roundtrip -> Top5/10/15
-> final Top-K LOPO with the same fixed params. All patients' predictions and
segment predictions are retained; only one minimum-error final Top-K fold model
is saved per config. No full-feature models are persisted. Parsing, masks,
XGBoost defaults, patient medians and RMSE/MAE match the existing EDAIC protocol.

Single Slurm job, one node and four H100 GPUs; four independent config processes,
one GPU/config. A free GPU takes the next config. The concurrent method-2 node
as04r5b01 is explicitly excluded. Reuse the existing MN5 environment.

No ALL retraining: the three previous gain ALL rows are appended to the summary,
explicitly marked result_origin=reused_gain_all. Final summary has 12 reporting
rows but only nine new experiments. Method-3 fixed SHAP versus matching gain is
the controlled ranking comparison; method-2 retunes its hyperparameters.

Outputs: edaic_xgb/shap_fixed_params on MN5; logs/<config>.log,
experiment_summary.csv, patient_fold_index.csv, run_manifest.json and config
artifacts. Existing gain and method-2 experiments are not modified.
Validation: python run_shap_configs.py --validate-only
Submit: sbatch run_shap.slurm
