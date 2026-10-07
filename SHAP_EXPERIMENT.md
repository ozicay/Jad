# SHAP Top-15 experiment

`optuna_xgb_is09_top15_shap.py` is a copy of the unchanged
`optuna_xgb_is09_top15.py` gain baseline. Its two importance call sites use
`shap.TreeExplainer(model)` on that fold's `X_train`, then
`np.mean(np.abs(values), axis=0)`. Explanation objects, array results, and older
non-callable explainers are supported. The regression SHAP array must have the
same shape as `X_train`.

LOPO masks, training/prediction calls, estimator parameters, Optuna ranges,
20 trials, five workers, seed 42, segment prediction medians, Top-15 selection,
and RMSE/MAE formulas are preserved. Features remain indexed by original columns;
selected columns are ordered by ascending original index as in the baseline.
The existing CSV serialization/reload precision and equal-fold averaging remain.

Each individual fold excludes its held-out patient from SHAP. The baseline's
cross-fold selection is retained: a patient excluded in one fold is present in
other folds' training sets. Consequently, the global ranking and subsequent
Top-15 LOPO evaluation are not a nested, independently selected evaluation.
This inherited limitation is not corrected in this controlled comparison.

The requested production metadata path overrides the old path in the copy only:
`/gpfs/projects/etur92/ozu150751/jad/geriatri_new/geriatri_metadata/metadata_with_is09.csv`.
The metadata is not copied or modified.

All experiment outputs, including Optuna temporary importance, go under
`/gpfs/projects/etur92/ozu150751/jad/xgb_codes/depr_is09_top15_shap`.
The final global ranking and selected Top-15 CSVs record ranks, original feature
names/indices, and mean absolute SHAP. The all-feature and Top-15 patient-level
metrics remain available. No clinical experiment is run on staging.

## Validation

The staging base environment uses NumPy 1 binaries, whereas SHAP 0.50 requires
NumPy 2. Validation therefore uses the ignored project-local `.venv-shap`.
SHAP 0.49.1 failed a synthetic TreeExplainer test against XGBoost 3.1.2 because
of its vector-form base_score. SHAP 0.50.0 passes against XGBoost 3.2.0, the
existing MN5 XGBoost version; no production XGBoost upgrade is required.

```bash
source .venv-shap/bin/activate
python -m py_compile optuna_xgb_is09_top15_shap.py
python -m compileall -q .
python -m pytest -v
bash -n run_xgb_is09_top15_shap.slurm
```

Tests execute isolated AST-extracted functions/source sections without importing
the clinical script's top-level training. They cover modern/older SHAP return
formats, absolute magnitudes, a tiny CPU XGBoost smoke, fold averaging/ranking,
Top-15 column mapping and dimensions, training-only explanations in final folds
and each Optuna trial, and structural equality of split assignments, estimator
parameters, search ranges and metric calls with the baseline.

## MN5 deployment

The existing Jad directory has no Git checkout. Deploy to a separate
`/gpfs/projects/etur92/ozu150751/jad/source-shap` checkout, preserving the existing
project files. The job runs from the main Jad directory and executes the new
script in that checkout.

`run_xgb_is09_top15_shap.slurm` follows the project script
`geriatri_job/geriatri_gdö_xgb.slurm` (completed job 45887169): account `etur92`,
QoS `acc_ehpc`, partition `acc`, one task, 20 requested CPUs, one GPU, 24 hours,
and default site memory allocation. Python is the existing module
`Python/3.12.3-GCCcore-13.3.0` with `$HOME/.local/lib/python3.12/site-packages`.
OMP/MKL/OpenBLAS thread limits are four, following that working five-worker job.
The scheduler's CUDA visibility is preserved. Logs use `%x-%j` inside the SHAP
output directory, so job logs are unique. Create the log directory before sbatch.
Install only missing SHAP dependencies into this existing user-site environment;
MN5 already has NumPy 2 and the other scientific dependencies.
