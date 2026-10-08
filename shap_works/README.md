# Configurable XGBoost experiments

`optuna_xgb_configurable.py` is a configurable copy of the unchanged
`optuna_xgb_is09_top15.py`. The production configurations use only the exact
`metadata_with_{mfcc,egemaps,is09}_nong` files in MN5's
`jad/nongeriatri_metadata_negative` directory, targeting `depresyon_skoru`.

The baseline parser, filename patient-ID extraction, patient LOPO masks,
XGBoost constructors, trial search space, 20 Optuna trials, fold-mean objective,
built-in feature importance and patient median prediction are preserved.
For Top-K, both the trial and final stages use the same `config["top_k"]`;
the final full-feature run is retained before feature reduction and retraining.
The final ranking still reloads the baseline six-decimal importance CSV.
All mode skips trial ranking/reduction and the second final LOPO stage.
Final RMSE/MAE are calculated from patient predictions, as in the baseline.

This intentionally retains the baseline feature-ranking procedure across all
full-feature LOPO folds. It does not introduce nested selection or change the
scientific evaluation design. The `shap_works` name does not imply SHAP usage.

Outputs, models, temporary trial importance, comparisons and final metrics go
to each config's independent `exp_models/<experiment_name>` directory.
Submission refuses nonempty output directories and records job IDs immediately
in `logs/submissions.tsv`, preventing accidental duplicate launches.

The runner reuses completed Jad job 45887169's module-based Python environment,
`etur92` / `acc_ehpc`, 1 GPU, 20 requested CPUs, and 24 hours. Memory remains the
partition default (no explicit memory directive), as in that successful script.
Slurm may allocate more CPU resources than requested due to GPU/node policy.

Validation uses the existing `.venv-shap` interpreter. Synthetic model doubles
exercise all 12 configurations without running production training on staging.
The tests also compare the parser, patient extractor and final training function
against the baseline AST. Run:

```bash
PATH="$PWD/.venv-shap/bin:$PATH" python -m py_compile shap_works/optuna_xgb_configurable.py
PATH="$PWD/.venv-shap/bin:$PATH" python -m compileall .
PATH="$PWD/.venv-shap/bin:$PATH" python -m pytest -v
```

On MN5, audit deployed configs with `scripts/audit_configs.py CONFIG_DIRECTORY`
before calling `scripts/submit_all.sh`. It checks headers, every row's parsed
feature dimension and the complete 12-row configuration table without training.

Source layout in GitHub:

```text
shap_works/
├── optuna_xgb_configurable.py
├── configs/              # 12 production configurations
├── scripts/              # generic runner, submitter and config audit
├── tests/                # configuration/LOPO regression tests
└── README.md
```

The original baseline and the earlier SHAP experiment remain at their existing
locations. This layout change does not update the source checkout used by the
12 jobs already submitted at revision `0d53a434116f0670b8415cd08eb617f53736988e`.
Future deployments must use the runner and Python path from the same revision.
MN5 logs and experiment outputs are not versioned.
