# Pooled anxiety + age_group

Exact copies of local depression Table-4 metadata (9 CSVs), unchanged selected-102 CSV.
Target anksiyete_skoru. 12 sequential MFCC/eGeMAPS/IS09 Top-5/10/15/All configurations.
40 Optuna trials/config; unchanged gain/LOPO/median/mean-fold-RMSE objective and final
patient RMSE/MAE. Age is binary cohort feature and not forced into Top-K.
One node, one Slurm job, 1 GPU, 20 CPU, five final LOPO workers; 72h time limit.
Default unseeded Optuna sampler and model seed42 retained; All repeated LOPO retained.
One minimum-held-out-AE final model saved per config; all 204 patients contribute to metrics.
All data/source/logs/results under jad/anxiety_age_pool; outputs/ and logs/ are writable.
Git branch agent/anxiety-age-pool records source only. SOURCE_COMMIT records deployed revision; SHA256SUMS records exact source/input copies. Clinical CSVs and results are not committed.
No sound files are needed: feature matrices are already in CSVs.

Validation: python run_anxiety_age_pool.py --validate-only
Submission (after verified remote deployment/environment): sbatch run_anxiety_age_pool.slurm
Pinned direct versions match depression staging environment; supporting dependency versions
are recorded at runtime. env_site isolates pinned core packages; shared dependencies read only.
Job completion must be verified with sacct and all expected artifacts; not merely empty squeue.
