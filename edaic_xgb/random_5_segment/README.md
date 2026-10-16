# EDAIC: random five segments per patient

One node, four H100 GPUs; four independent LOPO folds at a time.
Twelve configurations run sequentially, with 20 completed Optuna trials each.

The existing four-GPU runner is copied byte for byte as `runner_core.py`.
Only its input population changes: retain patients with at least five segments,
then select five distinct segments per patient using Python `random.Random(42)`.
Patient IDs and candidate filenames are sorted before sampling. The same fixed
segment filenames are used for all three feature sets, all trials and all configs.
This run uses one sampling seed, not three repeats.

229 patients, 1145 segments; every LOPO fold trains on 1140 segments and tests
on the held-out patient's five segments. Target remains `depresyon_skoru`.
Optuna search ranges/objective, XGBoost parameters, importance averaging/ranking,
Top-K reselection, median aggregation and RMSE/MAE remain unchanged.
Each final Top-K experiment saves exactly one best-fold model; ALL saves none.

Preparation: `python prepare_random5.py --source-metadata ../metadata`
Validation: `python run_random5.py --validate-only`
Submission on MN5: `sbatch run_random5.slurm`

Reuse the existing MN5 runtime; do not install/copy packages. Selection metadata,
patient identities, models and outputs are excluded from Git.
`selection_manifest.json`, `selected_segments.csv`, `SOURCE_COMMIT` and
`SHA256SUMS` identify the deployed selection and source.

Compare against an all-segment condition on the same 229 patients for a controlled
assessment. The prior 233-patient experiment is descriptive context, not a matched
comparison. Improvement and resolution of mean collapse are not guaranteed.
