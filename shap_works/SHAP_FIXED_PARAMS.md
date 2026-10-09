# Controlled gain versus SHAP: fixed gain hyperparameters

Only nine depression Top-K experiments are submitted. The 24 completed gain
parameter sets (both targets, including All) were extracted from their actual
stdout `Best params` lines and verified against the minimum objective among
all 20 completed trial records in stderr. Exact values and source logs are in
`gain_best_params_verified.json` and `.csv`.

`optuna_xgb_configurable_shap_fixed_params.py` removes the tuning stage from
the configurable final workflow. It contains no Optuna import, study, objective
or search. Both full-feature and Top-K models load the same three fixed values
from the matching depression gain config. The explicitly stated `hist` matches
the final gain XGBoost 3.2 default; objective, CUDA and random_state=42 remain.

Loading/parsing, patient IDs, groups, LOPO masks, all segments, five workers,
median prediction and final RMSE/MAE are retained. Only full-feature folds
compute mean absolute TreeSHAP on their training rows. Fold averaging and the
original six-decimal importance serialization/reload are retained. All configs
are rejected: the existing gain All baseline is reused without retraining.

Outputs under MN5 `jad/shap_works/shap_fixed_param_exp_models/<experiment>/`
include full-feature and selected patient models/predictions, SHAP fold vectors,
global ranking, selected_topK_features.csv, final_patient_level_rmse_mae.csv,
fixed_params.json and run_config.json. The patient comparison reads the existing
gain prediction CSV, stored by the baseline at three decimals, and does not
alter either result. Official gain metrics are read from its final metric CSV.
The rerun full-feature SHAP metrics are also reported for consistency checks;
gain's corresponding ALL_FEATURES row remains the existing control baseline.

Configs are `shap_fixed_param_configs/`; scripts `shap_fixed_param_scripts/`;
logs `shap_fixed_param_logs/shap_fixed_<experiment>_<job>.out` / `.err`.
Source is an independent `source-shap-fixed-params/` checkout under MN5
shap_works, with a hash-verified Python copy at the requested workspace root.
Same Slurm environment/resources: etur92 / acc_ehpc / acc, one GPU, 20 requested
CPUs, default memory, 24 hours. No gain or earlier SHAP output is overwritten.

Tests cover all nine settings, forbid new Optuna, verify both stages receive
identical fixed parameters, verify training-only SHAP and no selected-model SHAP,
median/LOPO, exact parameter registry matching, and output/comparison artifacts.

## Anksiyete: same controlled design

`optuna_xgb_configurable_shap_fixed_params_anksiyete.py` changes only target
validation to `anksiyete_skoru`; the complete training function and helper are
identical to the fixed-parameter depression implementation. Nine configs in
`shap_fixed_param_configs/anksiyete/` use each matching anxiety gain log's own
best_params (registry entries prefixed `anx_`). No Optuna or All retraining.

Output folders explicitly use `shap_fixed_param_exp_models/anksiyete_<setting>`;
logs `shap_fixed_param_logs/anksiyete/shap_fixed_anksiyete_<setting>_<job>.*`.
Separate runner, audit, and submit scripts have `anksiyete` in their names.
An independent source-shap-fixed-params-anksiyete checkout preserves previous
runs. SHAP receives only full-feature fold training rows; both training stages
use the same fixed params. Clinical labels/LOPO/median/metrics remain unchanged.
