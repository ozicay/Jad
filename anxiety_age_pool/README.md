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


## Four-GPU migration (2026-10-15)

User authorized only anxiety job47287479 migration; EDAIC job must not be touched.
New single-node/single-job wrapper run_anxiety_four_gpu.slurm requests four GPUs and
four tasks at20CPUs/task. Controller launches at most four exclusive one-GPU srun steps.
Each config still uses original scientific LOPO/objective/feature selection code and40trials.
Completed config outputs are preserved and skipped after artifact/coverage checks.
Original outputs must be snapshotted after predecessor job is cancelled before launch.
Completed Optuna CSV histories are imported into per-config SQLite studies; incomplete
trial restarts. Original in-memory unseeded sampler state is unavailable, so subsequent
sampling is not guaranteed to match uninterrupted execution. Imported parameters and
objective values are exact; original timestamps remain in migration snapshot.
Controller alone updates shared summary; workers write distinct config directories.
Existing MN5 env_site/PYTHONPATH reused unchanged; no new environment or installations.
Source and metadata integrity are recorded in SHA256SUMS_4GPU/SOURCE_COMMIT_4GPU.
