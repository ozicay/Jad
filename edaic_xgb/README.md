# EDAIC depression — Optuna XGBoost on MN5

All new code, metadata, run logs and experiment outputs live under:

`/gpfs/projects/etur92/ozu150751/jad/edaic_xgb`

Submit once on MN5:

```bash
sbatch /gpfs/projects/etur92/ozu150751/jad/edaic_xgb/run_edaic_xgb.slurm
```

Or submit from this computer with one command:

```bash
ssh alogin2.bsc.es 'sbatch /gpfs/projects/etur92/ozu150751/jad/edaic_xgb/run_edaic_xgb.slurm'
```

The job requests four H100 GPUs, 20 CPUs, and the verified 72-hour `acc_ehpc`
limit. One Python process executes these twelve experiments sequentially:

```
mfcc_top5 → mfcc_top10 → mfcc_top15 → mfcc_all
egemaps_top5 → egemaps_top10 → egemaps_top15 → egemaps_all
is09_top5 → is09_top10 → is09_top15 → is09_all
```

Trial and final LOPO use four workers, one pinned to each of cuda:0–cuda:3.
Optuna trials and experiments remain sequential. Only independent patient
folds within the same LOPO stage run concurrently. The first finished fold
on each GPU prints its verified XGBoost device and the cumulative GPU count;
each completed stage prints the total GPUs actually used. Slurm startup
checks and prints four H100 names. No Accelerate or new dependencies are used.
There is no job array, cross-experiment concurrency or random/validation split.

## Data and target

The three files in `metadata/` are byte-for-byte copies of the metadata
used to prepare the earlier local EDAIC experiment. Audio files are not
needed: all inputs are numeric feature vectors already stored in these CSVs.

| CSV | Feature column | Dimension |
|---|---|---|
| metadata_with_mfcc.csv | features | 608 |
| metadata_with_egemapsv0b1.csv | egemaps_features | 88 |
| metadata_with_is09.csv | is09_features | 384 |

Target: `depresyon_skoru`. All three CSVs have 4,829 segments and the same
233 patients. Patient ID is extracted from the `file_name` CSV index using
`basename(filename).split('-')[0].lower()`, the reference extraction rule.
No geriatric/comorbid dataset prefixes or age features are added to EDAIC.
Patient-ID sets, segment identities, score consistency and feature dimensions
are checked before fitting. Existing train/dev/test annotations are not used;
the requested evaluation remains patient-level LOPO over this population.

## Reference protocol and deliberate adaptations

Reference: `reference/optuna_xgb_is09_top15.py`, copied unchanged from the main
Jad script. Its SHA256 is
`e78e97f99278321b3d12035b638e874f2088f8cb5b828a08af7313b188e4fb51`.
Original files and the earlier fixed-parameter EDAIC runner are unchanged.

Each experiment has a separate study with **20 complete trials**:

| Parameter | Search range |
|---|---|
| learning_rate | 0.001–0.1, log=True |
| max_depth | integer 4–14 |
| n_estimators | integer 100–500 |

Trial parameters also preserve `objective='reg:squarederror'`,
`tree_method='hist'`, `device='cuda'`, `random_state=42`.
The default Optuna sampler is used with no newly imposed seed or pruning.

For Top-K, the trial objective was extracted from the reference. Its only
adaptations are explicit data arguments, experiment-local temporary paths,
progress printing, configurable K (5/10/15), and a full-feature-only ALL
branch. The four-GPU objective preserves the numerical operations and trial
parameter expressions, distributing independent folds while keeping their
aggregation order fixed; synthetic tests compare it with the serial reference. AST validation compares it with that documented transformation.

Every Top-K trial performs full-feature LOPO, averages
`model.feature_importances_` across all held-out patients, ranks descending,
selects K features, and repeats LOPO with those feature indices. The
temporary importance CSV roundtrip uses the reference default precision.
Optuna minimizes **mean per-patient fold RMSE**, preserving the reference
objective; this is distinct from the final pooled patient-level RMSE.
ALL trials perform one full-feature LOPO, using that same mean-fold-error
objective, with no feature reduction or second LOPO stage.

After tuning, the best parameters drive final full-feature LOPO. Final
models omit `tree_method`, as in the reference's final model constructor.
The final importance CSV is written with six decimals and read back, as
in the reference. Top-K selection then uses the configured K and final
LOPO is rerun; ALL uses the full-feature predictions directly.

LOPO masks remain `target_mask = groups_all == target_patient` and
`train_mask = ~target_mask`. All held-out segments are predicted; patient
prediction is their median. Each fold's RMSE is
`sqrt(mean_squared_error([true_score], [predicted_median]))`.
Final metrics use **every** patient's held-out prediction:
`sqrt(mean_squared_error(y_true, y_pred))` and `mean(abs(y_true-y_pred))`.
Failures stop the experiment; partial fold populations are never reported
as a complete result.

Only the minimum-fold-RMSE already-trained Top-K model is saved, once per
experiment. Ties choose the first patient in sorted ID order. Only the
current best candidate is retained across folds; active fold workers hold
their own working models. Full-feature/trial models are not saved. The
saved artifact does not select patients or change final evaluation.

## Outputs and continuation

Each experiment writes its requested patient-result, importance, selected
feature, comparison and final-metric CSVs. Top-K experiments also write
`best_topk_model.pkl` and `best_topk_model_info.csv`; ALL experiments save
no model artifact. Feature names `feat_N` identify original vector indices;
no semantic feature names are supplied by the metadata. Segment predictions
are preserved in NPZ files by patient and original within-patient row order.

Per-experiment additions: `optuna_trials.csv`, `best_params.json`,
`optuna_study.sqlite3`, per-trial temporary importance files,
`experiment_config.json`, and a completion record with artifact hashes.
At the root: `experiment_summary.csv` (12 rows when all complete),
`patient_fold_index.csv`, `sequential_run.log`, `run_manifest.json`,
`logs/slurm_JOBID.{out,err}`, `logs/environment_JOBID.json`, and journal.

Each new script invocation is logged. If a 72-hour allocation ends early,
the same submission command continues existing studies to 20 complete
trials, verifies completed artifacts and skips completed experiments.
An unfinished final stage is recomputed using its existing best parameters;
no extra Optuna search is performed. Interrupted trials are recorded as
failed; a replacement trial uses the remaining complete-trial budget.
The one-GPU source was archived and its study backed up before migration.
The batch script permits an explicit scheduling-only migration from the old
source SHA256 a8ac8030f8029f7dc8e2be353aded81c49e922f2a34b33a5984917dc22c191b8;
metadata/reference identities must still match. Completed trials are reused.
Other changed source/config/data identities are rejected, and an exclusive lock
prevents simultaneous runners in this directory. Do not edit production
source in the middle of a run.

## Validation and environment

`python run_edaic_xgb.py --validate-only` checks source and real metadata
without training. Run this under the Slurm script's module/PYTHONPATH setup
on MN5. Dependency versions and their actual module paths are recorded.
`env_site` links to the existing compatible core packages in
`../anxiety_age_pool/env_site`; this reuses package code, not that experiment's
data/configuration. Remaining dependencies are read from the established MN5
`py312_job_site` and user site. Bytecode writes are disabled by the job script;
no shared environment is modified. An abandoned, incomplete package copy is
preserved separately as `env_site_incomplete` and is never on PYTHONPATH.

`python -B edaic_xgb/test_runner.py` locally exercises real Optuna studies
using synthetic models: reference-objective equivalence, all twelve configs,
240 trials, masks, median, all-patient metrics, one model per Top-K, ALL
bypass and restart handling. It never trains on EDAIC or uses a GPU.

Exact deployed files are identified by SHA256SUMS and the deployment
manifest. When published, SOURCE_COMMIT records the deployed source revision;
SOURCE_PARENT_COMMIT records the original base revision. No real results
are claimed by preparation or validation alone.
