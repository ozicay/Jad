import os
import re
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pickle
import joblib
from datetime import datetime
import concurrent.futures
from threading import Lock
import gc
import optuna
import shap
from sklearn.model_selection import train_test_split

# —————————————————————————————
# Ayarlar
# —————————————————————————————home/

depr_only_csv = '/gpfs/projects/etur92/ozu150751/jad/geriatri_new/geriatri_metadata/metadata_with_is09.csv'
#extra = '/home/utku/emo_depression/xgboost/metadata_w2v2_scores_mfcc_with_3_feat_comorbid.csv'
base_model_dir = '/gpfs/projects/etur92/ozu150751/jad/xgb_codes/depr_is09_top15_shap' #14 15 16 17,               18full
os.makedirs(base_model_dir, exist_ok=True)
# GPU ve Thread ayarları
MAX_WORKERS = 5
CUDA_DEVICE = "cuda:0"

print(f"🚀 ENHANCED COMBINED METADATA + STANDARD LOO (NO DATASET FLAG) TRAINING")
print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"👤 User: tgrhnkrkdgl")
print(f"🔥 Using {CUDA_DEVICE} with {MAX_WORKERS} parallel workers")
print(f"📦 Strategy: metadata → Standard LOO (119 train, 1 test) → NO dataset flag")

model_lock = Lock()
# —————————————————————————————
# Feature parsing
# —————————————————————————————
num_re = re.compile(r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+(?:\.\d+)?)?')

def parse_token(tok: str) -> float:
    if 'e' in tok or 'E' in tok:
        mant, exp = re.split('[eE]', tok)
        return float(mant) * (10 ** float(exp))
    return float(tok)

def parse_features(s: str) -> np.ndarray:
    toks = re.split(r'[,\s]+', s)
    return np.array([parse_token(t) for t in toks if t and num_re.fullmatch(t)], dtype=float)

def mean_abs_training_shap(model, X_train):
    """One mean absolute SHAP score per feature, using fold training rows only."""
    explainer = shap.TreeExplainer(model)
    if callable(explainer):
        shap_values = explainer(X_train)
    else:
        shap_values = explainer.shap_values(X_train)
    shap_array = np.asarray(getattr(shap_values, 'values', shap_values))
    if shap_array.shape != X_train.shape:
        raise ValueError(f"Expected SHAP shape {X_train.shape}, got {shap_array.shape}")
    return np.mean(np.abs(shap_array), axis=0)

def train_loo_model_with_importance(loo_args):
    """Standard LOO + Mean Absolute Training SHAP Importance"""
    (target_patient, X_train, y_train, X_target_test, y_target_test,
     true_score, target_dataset, patient_model_dir) = loo_args


    try:
        print(f"    🎯 LOO Training {target_patient} ({target_dataset})")
        print(f"      📊 Training: {len(X_train)} samples with {X_train.shape[1]} features")
        print(f"      📊 Testing: {len(X_target_test)} samples")

        model = XGBRegressor(
            **best_params,
            objective='reg:squarederror',
            random_state=42,
            device='cuda' if 'cuda' in CUDA_DEVICE else 'cpu'
        )


        model.fit(X_train, y_train)

        target_predictions = model.predict(X_target_test)
        predicted_median = np.median(target_predictions)

        rmse = np.sqrt(mean_squared_error([true_score], [predicted_median]))

        shap_importance = mean_abs_training_shap(model, X_train)

        with model_lock:
            model_path = os.path.join(patient_model_dir, f'loo_model_{target_patient}.pkl')
            joblib.dump(model, model_path)

        result = {
            'patient_id': target_patient,
            'true_score': true_score,
            'predicted_score': predicted_median,
            'rmse': rmse,
            'target_dataset': target_dataset,
            'training_samples': len(X_train),
            'test_samples': len(X_target_test),
            'features_count': X_train.shape[1],
            'model_path': model_path,
            'shap_importance': shap_importance,
            'status': 'success'
        }

        cluster_importance = shap_importance[-1]
        cluster_rank = len(shap_importance) - np.argsort(shap_importance)[::-1].tolist().index(len(shap_importance)-1)

        print(f"      🎯 Last feature SHAP importance: {cluster_importance:.4f} (rank: {cluster_rank})")
        print(f"      ✅ {target_patient}: True={true_score:.3f}, Pred={predicted_median:.3f}, RMSE={rmse:.3f}")

        del model
        gc.collect()

        return result

    except Exception as e:
        print(f"      ❌ Error training {target_patient}: {e}")
        return {
            'patient_id': target_patient,
            'target_dataset': target_dataset,
            'status': 'failed',
            'error': str(e)
        }

print(f"\n📊 Loading and combining metadata...")

def extract_patient_id_from_filename(filename):
    """Filename'den hasta ID'sini çıkar"""
    # Örnek: ad1-clip1.wav → ad1
    base_name = os.path.basename(str(filename))
    patient_id = base_name.split('-')[0].lower()
    return patient_id

print(f"  📁 Loading depr_only: {depr_only_csv}")
df_depr = pd.read_csv(depr_only_csv, index_col=0, encoding='utf-8')
df_depr['dataset'] = 'depr_only'
df_depr['original_path'] = df_depr.index
df_depr['patient_id_raw'] = df_depr['original_path'].map(extract_patient_id_from_filename)
df_depr['patient_id'] = 'depr_' + df_depr['patient_id_raw'].astype(str)
print(f"    📦 Depr only: {len(df_depr)} samples")

# İki CSV'yi alt alta birleştiriyoruz
df = pd.concat([df_depr], axis=0, ignore_index=True)

print(f"\n📦 POOLED DATASET CREATED:")
print(f"  Total samples: {len(df)}")
print(f"  Total patients: {df['patient_id'].nunique()}")
print(df['dataset'].value_counts())

#df2= pd.read_csv(extra, index_col=0, encoding='utf-8')
# === 🧩 Ek metadata özellikleri (age, medeni_hal, egitim, meslek)
meta_cols = ["age", "medeni_hal", "egitim", "meslek"]
#X_meta = df2[meta_cols].astype(float).values

print(f"\n🔧 PROCESSING FEATURES (NO DATASET FLAG):")
df['is09_features'] = df['is09_features'].apply(parse_features)
df = df[df['is09_features'].notna()].reset_index(drop=True)

X_all = np.vstack(df['is09_features'].values)
#X_all = np.hstack([X_features, X_meta])



y_all = df['depresyon_skoru'].values
groups_all = df['patient_id'].values
unique_pids = np.unique(groups_all)

print(f"\n📊 FINAL PATIENT CHECK:")
print(f"  👥 Unique patients: {len(unique_pids)}")
print(f"  📋 Sample patient IDs: {list(unique_pids)[:10]}")
if any(pid.startswith('patient_') for pid in unique_pids):
    print(f"  ❌ ERROR: Still getting 'patient_X' IDs!")
    print(f"  🔍 Check original file paths in CSV index column")
else:
    print(f"  ✅ SUCCESS: Getting proper patient IDs like 'ad1', 'ad2', etc.")

patient_info = df.groupby('patient_id').agg({
    'depresyon_skoru': 'first',
    'dataset': 'first'
}).to_dict('index')

print(f"\n📊 DATASET INFO:")
print(f"  📦 Total samples: {len(df)}")
print(f"  👥 Total patients: {len(unique_pids)}")
print(f"  🎯 Features: {X_all.shape[1]}")
print(f"  📊 LOO Strategy: {len(unique_pids)} models (119 train → 1 test each)")


print("\n🚀 OPTUNA hyperparameter tuning starting...")

# def objective(trial):

#     params = {
#         "objective": "reg:squarederror",

#         # learning rate 1e-4 ile 0.3 arası (log-scale önerilir)
#         "learning_rate": trial.suggest_float(
#             "learning_rate", 1e-3, 0.1, log=True
#         ),

#         # max_depth 4 ile 14 arası integer
#         "max_depth": trial.suggest_int(
#             "max_depth", 4, 14
#         ),

#         # n_estimators 100 ile 500 arası integer
#         "n_estimators": trial.suggest_int(
#             "n_estimators", 100, 500
#         ),

#         "tree_method": "hist",
#         "device": "cuda",
#         "random_state": 42,
#     }


#     rmses = []

#     for pid in unique_pids:
#         mask = (groups_all == pid)

#         X_train = X_all[~mask]
#         y_train = y_all[~mask]
#         X_test  = X_all[mask]

#         model = XGBRegressor(**params)
#         model.fit(X_train, y_train)

#         preds = model.predict(X_test)
#         med = np.median(preds)

#         true = patient_info[pid]['depresyon_skoru']
#         rmse = np.sqrt(mean_squared_error([true], [med]))

#         rmses.append(rmse)

#         del model
#         gc.collect()

#     mean_rmse = np.mean(rmses)

#     print(f"OPTUNA trial FULL FEATURES → mean RMSE = {mean_rmse:.4f}")

#     return mean_rmse




def objective(trial):

    params = {
        "objective": "reg:squarederror",

        # learning rate 1e-4 ile 0.3 arası (log-scale önerilir)
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-3, 0.1, log=True
        ),

        # max_depth 4 ile 14 arası integer
        "max_depth": trial.suggest_int(
            "max_depth", 4, 14
        ),

        # n_estimators 100 ile 500 arası integer
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 500
        ),

        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
    }

    # ------------------------------------
    # 1) FULL-FEATURE LOO (importance + RMSE)
    # ------------------------------------
    LOO_results = []


    for pid in unique_pids:

        mask = (groups_all == pid)

        X_train = X_all[~mask]
        y_train = y_all[~mask]
        X_test  = X_all[mask]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        pred_med = np.median(preds)
        true = patient_info[pid]['depresyon_skoru']

        rmse = np.sqrt(mean_squared_error([true],[pred_med]))


        LOO_results.append({
            "pid":pid,
            "rmse":rmse,
            "importance":mean_abs_training_shap(model, X_train)
        })

        del model
        gc.collect()


    # ------------------------------
    # 2) SAVE TO TEMP CSV (exact same as main!)
    # ------------------------------
    imp_rows=[]
    for r in LOO_results:
        row={"pid":r["pid"],"rmse":r["rmse"]}
        for i,v in enumerate(r["importance"]):
            row[f"feat_{i}"]=v
        imp_rows.append(row)

    tmp_imp_df=pd.DataFrame(imp_rows)
    tmp_imp_df.to_csv(os.path.join(base_model_dir, "OPTUNA_TMP_SHAP_IMPORTANCE.csv"), index=False)


    # ------------------------------
    # 3) RELOAD + compute avg imp (exact same as main!)
    # ------------------------------
    imp_df = pd.read_csv(os.path.join(base_model_dir, "OPTUNA_TMP_SHAP_IMPORTANCE.csv"))

    feat_cols = [c for c in imp_df.columns if c.startswith("feat_")]

    avg_imp = imp_df[feat_cols].mean().sort_values(ascending=False)

    top_features = avg_imp.head(15).index.tolist()  # top feature değiştirme yeri
    top15_idx = sorted([int(f.split("_")[1]) for f in top_features])


    # ------------------------------6.6027
    # 4) TOP-15 FEATURE LOO
    # ------------------------------
    rmses=[]
    for pid in unique_pids:

        mask = (groups_all == pid)

        X_train = X_all[~mask][:,top15_idx]
        y_train = y_all[~mask]
        X_test  = X_all[mask][:,top15_idx]

        model = XGBRegressor(**params)
        model.fit(X_train,y_train)

        preds = model.predict(X_test)
        med = np.median(preds)

        true = patient_info[pid]['depresyon_skoru']
        rmse = np.sqrt(mean_squared_error([true],[med]))

        rmses.append(rmse)

        del model
        gc.collect()

    mean_rmse = np.mean(rmses)

    print(f"OPTUNA trial → mean = {mean_rmse:.4f}")

    return mean_rmse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

best_params = study.best_params
print("Best params:", best_params)


# —————————————————————————————
# Main Standard LOO training loop ()
# —————————————————————————————
print(f"\n🚀 STARTING STANDARD LOO TRAINING...")

# Prepare all LOO jobs
print(f"\n🚀 PREPARING LOO JOBS WITH TRAINING SHAP IMPORTANCE...")

training_jobs = []
start_time = datetime.now()

for target_patient in unique_pids:
    target_mask = groups_all == target_patient
    train_mask = ~target_mask

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]

    X_target_test = X_all[target_mask]
    y_target_test = y_all[target_mask]

    true_score = patient_info[target_patient]['depresyon_skoru']
    target_dataset = patient_info[target_patient]['dataset']

    patient_model_dir = os.path.join(base_model_dir, f'patient_{target_patient}')
    os.makedirs(patient_model_dir, exist_ok=True)

    training_jobs.append((
        target_patient, X_train, y_train, X_target_test, y_target_test,
        true_score, target_dataset, patient_model_dir
    ))

print(f"  📦 Prepared {len(training_jobs)} Standard LOO training jobs")
print(f"  📊 Each model: {len(unique_pids)-1} patients for training, 1 patient for testing")
print(f"  📊 Features per model: {X_all.shape[1]} (original only")

# —————————————————————————————
# Parallel Standard LOO training
# —————————————————————————————
all_results = []

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    print(f"  🚀 Starting parallel LOO training with feature importance...")

    future_to_patient = {executor.submit(train_loo_model_with_importance, job): job[0] for job in training_jobs}

    for future in concurrent.futures.as_completed(future_to_patient):
        target_patient = future_to_patient[future]
        try:
            result = future.result()
            if result:
                all_results.append(result)
                if len(all_results) % 10 == 0:
                    print(f"    ✅ Completed {len(all_results)}/{len(training_jobs)} LOO models...")
        except Exception as e:
            print(f"    ❌ Job failed for {target_patient}: {e}")

total_duration = (datetime.now() - start_time).total_seconds()

# —————————————————————————————
# Save and analyze results
# —————————————————————————————
successful_results = [r for r in all_results if r.get('status') == 'success']

if successful_results:
    results_df = pd.DataFrame(successful_results)

    overall_mean_rmse = results_df['rmse'].mean()
    overall_std_rmse = results_df['rmse'].std()
    overall_median_rmse = results_df['rmse'].median()
    overall_min_rmse = results_df['rmse'].min()
    overall_max_rmse = results_df['rmse'].max()

    # Add overall row
    overall_row = pd.DataFrame({
        'patient_id': ['OVERALL_AVERAGE'],
        'true_score': [results_df['true_score'].mean()],
        'predicted_score': [results_df['predicted_score'].mean()],
        'rmse': [overall_mean_rmse],
        'target_dataset': ['COMBINED'],
        'training_samples': [results_df['training_samples'].mean()],
        'test_samples': [results_df['test_samples'].mean()],
        'features_count': [results_df['features_count'].mean()],
        'model_path': [''],
        'status': ['summary']
    })

    final_df = pd.concat([results_df, overall_row], ignore_index=True)

    # Save results
    output_csv = os.path.join(base_model_dir, 'enhanced_combined_standard_loo_no_flag_results.csv')
    final_df.to_csv(output_csv, index=False, float_format='%.3f')

    print(f"\n✅ FINAL RESULTS SAVED TO: {output_csv}")
    print(f"\n📊 ENHANCED COMBINED STANDARD LOO (NO DATASET FLAG) SUMMARY:")
    print(f"  👥 Patients evaluated: {len(successful_results)}")
    print(f"  🎯 Overall Mean RMSE: {overall_mean_rmse:.3f}")
    print(f"  📊 Overall Std RMSE: {overall_std_rmse:.3f}")
    print(f"  📊 Overall Median RMSE: {overall_median_rmse:.3f}")
    print(f"  📈 RMSE Range: {overall_min_rmse:.3f} - {overall_max_rmse:.3f}")
    print(f"  🚫 NO dataset flag feature used")
    print(f"  ⏱️ Total training time: {total_duration/60:.1f} minutes")

    # Enhanced dataset breakdown
    print(f"\n📊 DETAILED DATASET BREAKDOWN:")

    for dataset in ['depr_only', 'depr_anx']:
        dataset_results = results_df[results_df['target_dataset'] == dataset]

        if len(dataset_results) > 0:
            dataset_count = len(dataset_results)
            dataset_mean_rmse = dataset_results['rmse'].mean()
            dataset_std_rmse = dataset_results['rmse'].std()
            dataset_min_rmse = dataset_results['rmse'].min()
            dataset_max_rmse = dataset_results['rmse'].max()
            dataset_median_rmse = dataset_results['rmse'].median()

            print(f"\n  🧠 {dataset.upper()}:")
            print(f"    👥 Count: {dataset_count} patients")
            print(f"    🎯 RMSE Mean: {dataset_mean_rmse:.3f}")
            print(f"    📊 RMSE Std: {dataset_std_rmse:.3f}")
            print(f"    📊 RMSE Median: {dataset_median_rmse:.3f}")
            print(f"    📈 RMSE Range: {dataset_min_rmse:.3f} - {dataset_max_rmse:.3f}")

            # Best and worst performers for this dataset
            best_patient = dataset_results.loc[dataset_results['rmse'].idxmin()]
            worst_patient = dataset_results.loc[dataset_results['rmse'].idxmax()]

            print(f"    🏆 Best: {best_patient['patient_id']} (RMSE: {best_patient['rmse']:.3f})")
            print(f"    📉 Worst: {worst_patient['patient_id']} (RMSE: {worst_patient['rmse']:.3f})")
        else:
            print(f"\n  ⚠️ {dataset.upper()}: No patients found")

    # Cross-dataset comparison
    if len(results_df[results_df['target_dataset'] == 'depr_only']) > 0 and len(results_df[results_df['target_dataset'] == 'depr_anx']) > 0:
        depr_only_mean = results_df[results_df['target_dataset'] == 'depr_only']['rmse'].mean()
        depr_anx_mean = results_df[results_df['target_dataset'] == 'depr_anx']['rmse'].mean()

        print(f"\n🔍 CROSS-DATASET COMPARISON:")
        print(f"  📊 depr_only average RMSE: {depr_only_mean:.3f}")
        print(f"  📊 depr_anx average RMSE: {depr_anx_mean:.3f}")
        print(f"  📈 Difference: {abs(depr_only_mean - depr_anx_mean):.3f}")

        if depr_only_mean < depr_anx_mean:
            print(f"  ✅ depr_only performs better by {depr_anx_mean - depr_only_mean:.3f} RMSE")
        elif depr_anx_mean < depr_only_mean:
            print(f"  ✅ depr_anx performs better by {depr_only_mean - depr_anx_mean:.3f} RMSE")
        else:
            print(f"  🤝 Both datasets perform equally")

    # Statistical summary
    print(f"\n📊 STATISTICAL SUMMARY:")
    print(f"  🎯 RMSE Statistics:")
    print(f"    Mean: {overall_mean_rmse:.3f}")
    print(f"    Median: {overall_median_rmse:.3f}")
    print(f"    Std: {overall_std_rmse:.3f}")
    print(f"    Min: {overall_min_rmse:.3f}")
    print(f"    Max: {overall_max_rmse:.3f}")

successful_results = [r for r in all_results if r.get('status') == 'success']

if successful_results:
    print(f"\n📊 CREATING SHAP IMPORTANCE CSV...")

    # Create training-fold SHAP importance dataframe
    importance_data = []

    for result in successful_results:
        if 'shap_importance' in result:
            importance_array = result['shap_importance']

            # Create row for this patient
            row = {
                'patient_id': result['patient_id'],
                'rmse': result['rmse'],
                'true_score': result['true_score'],
                'predicted_score': result['predicted_score'],
                'target_dataset': result['target_dataset']
            }

            # Add each mean absolute SHAP importance as feat_0, feat_1, feat_2...
            for i, importance in enumerate(importance_array):
                row[f'feat_{i}'] = importance

            importance_data.append(row)

    # Create DataFrame
    importance_df = pd.DataFrame(importance_data)

    # Save training-fold SHAP importance CSV
    importance_csv = os.path.join(base_model_dir, 'shap_importance_all_models.csv')
    importance_df.to_csv(importance_csv, index=False, float_format='%.6f')

    print(f"✅ SHAP importance CSV saved: {importance_csv}")
    print(f"📊 Shape: {importance_df.shape}")
    print(f"📊 Columns: patient_id, rmse, true_score, predicted_score, target_dataset, feat_0 to feat_{X_all.shape[1]-1}")

    # —————————————————————————————
    # Quick Analysis
    # —————————————————————————————
    print(f"\n🔍 QUICK SHAP IMPORTANCE ANALYSIS:")

    # Get feature columns only
    feature_cols = [col for col in importance_df.columns if col.startswith('feat_')]
    shap_importance_only = importance_df[feature_cols]

    # Calculate average importance per feature
    avg_importance = shap_importance_only.mean().sort_values(ascending=False)

    print(f"📊 TOP 15 FEATURES (MEAN ABSOLUTE SHAP):")
    for i, (feature, importance) in enumerate(avg_importance.head(15).items()):
        print(f"  {i+1:2d}. {feature}: {importance:.6f}")

    # Cluster_id analysis (last feature)
    cluster_feature = f'feat_{X_all.shape[1]-1}'  # Original last feature index
    cluster_avg_importance = avg_importance[cluster_feature]
    cluster_rank = avg_importance.index.get_loc(cluster_feature) + 1

    print(f"\n🎯 LAST FEATURE (feat_{X_all.shape[1]-1}) ANALYSIS:")
    print(f"  📊 Average importance: {cluster_avg_importance:.6f}")
    print(f"  📊 Rank: {cluster_rank} out of {len(avg_importance)}")
    print(f"  📊 Relative to top feature: {(cluster_avg_importance / avg_importance.iloc[0] * 100):.1f}%")

    if cluster_rank <= 10:
        print(f"  ✅ Last feature is in TOP 10 most important features!")
    elif cluster_rank <= 20:
        print(f"  ✅ Last feature is in TOP 20 most important features!")
    else:
        print(f"  ⚠️ Last feature SHAP importance is relatively low (rank {cluster_rank})")

    # Save summary
    summary_data = {
        'total_features': len(feature_cols),
        'cluster_feature_name': cluster_feature,
        'cluster_avg_importance': cluster_avg_importance,
        'cluster_rank': cluster_rank,
        'top_feature': avg_importance.index[0],
        'top_shap_importance': avg_importance.iloc[0]
    }

    summary_csv = os.path.join(base_model_dir, 'shap_importance_summary.csv')
    pd.DataFrame([summary_data]).to_csv(summary_csv, index=False, float_format='%.6f')

    print(f"\n💾 FILES SAVED:")
    print(f"  📊 SHAP importance (all models): {importance_csv}")
    print(f"  📋 Summary: {summary_csv}")
else:
    print(f"\n❌ No successful results!")

print(f"\n🚀 ENHANCED COMBINED STANDARD LOO (NO DATASET FLAG) COMPLETED!")
print(f"  📊 Strategy: Combined metadata → Standard LOO (NO dataset flag)")
print(f"  📁 Total models trained: {len(unique_pids)}")
print(f"  🎯 Features: {X_all.shape[1]} (original only, NO dataset flag)")
print(f"  ⏱️ Training completed in {total_duration/60:.1f} minutes")
print(f"  💾 Results saved to: {output_csv}")

# —————————————————————————————
# TOP-15 FEATURE İLE TEKRAR EĞİTİM ve KARŞILAŞTIRMA
# —————————————————————————————
print("\n🚀 RETRAIN: En yüksek önem skoruna sahip TOP-15 feature ile LOO başlıyor...")

# 1) Importance CSV'den ortalama önemleri oku (mevcut değişkenler varsa onları kullanır)
importance_csv = os.path.join(base_model_dir, 'shap_importance_all_models.csv')
if not os.path.exists(importance_csv):
    raise FileNotFoundError(f"Importance CSV bulunamadı: {importance_csv}")

importance_df = pd.read_csv(importance_csv)
feature_cols = [c for c in importance_df.columns if c.startswith('feat_')]

avg_importance = importance_df[feature_cols].mean().sort_values(ascending=False)
top_k =15

top_features = avg_importance.head(top_k).index.tolist()
global_ranking_df = pd.DataFrame({
    'rank': np.arange(1, len(avg_importance) + 1),
    'feature': avg_importance.index,
    'feature_index': [int(f.split('_')[1]) for f in avg_importance.index],
    'mean_abs_shap': avg_importance.values,
})
global_ranking_df.to_csv(os.path.join(base_model_dir, 'shap_global_ranking.csv'), index=False)
selected_top15_df = global_ranking_df.head(top_k)
selected_top15_df.to_csv(os.path.join(base_model_dir, 'selected_top15_features.csv'), index=False)
print("Final Top-15 SHAP ranking (original feature indices):")
print(selected_top15_df.to_string(index=False))

# feat_123 -> 123 indekslerine çevir
top_feature_indices = sorted([int(f.split('_')[1]) for f in top_features])

print(f"  🔝 Seçilen feature sayısı: {len(top_feature_indices)}")
print(f"  🧩 İlk 10: {top_feature_indices[:10]}{' ...' if len(top_feature_indices) > 10 else ''}")

# 2) Orijinal X_all'dan sadece TOP-15 kolonları al
X_top15 = X_all[:, top_feature_indices]
print(f"  📊 X_top15 shape: {X_top15.shape} (samples={X_top15.shape[0]}, features={X_top15.shape[1]})")

# 3) LOO işler listesini yeniden hazırla
print("\n🚀 TOP-15 için LOO işleri hazırlanıyor...")
training_jobs_top15 = []
start_time_top15 = datetime.now()

for target_patient in unique_pids:
    target_mask = groups_all == target_patient
    train_mask = ~target_mask

    X_train = X_top15[train_mask]
    y_train = y_all[train_mask]

    X_target_test = X_top15[target_mask]
    y_target_test = y_all[target_mask]

    true_score = patient_info[target_patient]['depresyon_skoru']
    target_dataset = patient_info[target_patient]['dataset']

    patient_model_dir = os.path.join(base_model_dir, f'patient_{target_patient}_top15')
    os.makedirs(patient_model_dir, exist_ok=True)

    training_jobs_top15.append((
        target_patient, X_train, y_train, X_target_test, y_target_test,
        true_score, target_dataset, patient_model_dir
    ))

print(f"  📦 Hazır LOO iş sayısı: {len(training_jobs_top15)}")
print(f"  📊 Her model feature sayısı: {X_top15.shape[1]}")

# 4) Paralel eğitim
all_results_top15 = []
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    print(f"  🚀 Paralel LOO (TOP-15) eğitim başlıyor...")
    future_to_patient = {executor.submit(train_loo_model_with_importance, job): job[0] for job in training_jobs_top15}

    for future in concurrent.futures.as_completed(future_to_patient):
        target_patient = future_to_patient[future]
        try:
            result = future.result()
            if result:
                all_results_top15.append(result)
                if len(all_results_top15) % 10 == 0:
                    print(f"    ✅ {len(all_results_top15)}/{len(training_jobs_top15)} tamamlandı...")
        except Exception as e:
            print(f"    ❌ Job failed for {target_patient}: {e}")

total_duration_top15 = (datetime.now() - start_time_top15).total_seconds()

# 5) Sonuçların kaydı
successful_top15 = [r for r in all_results_top15 if r.get('status') == 'success']
if not successful_top15:
    raise RuntimeError("TOP-15 LOO çalışmasından başarılı sonuç gelmedi.")

results_top15_df = pd.DataFrame(successful_top15)

# Özet satırı
overall_mean_rmse_top15 = results_top15_df['rmse'].mean()
overall_std_rmse_top15 = results_top15_df['rmse'].std()
overall_median_rmse_top15 = results_top15_df['rmse'].median()
overall_min_rmse_top15 = results_top15_df['rmse'].min()
overall_max_rmse_top15 = results_top15_df['rmse'].max()

overall_row_top15 = pd.DataFrame({
    'patient_id': ['OVERALL_AVERAGE_TOP15'],
    'true_score': [results_top15_df['true_score'].mean()],
    'predicted_score': [results_top15_df['predicted_score'].mean()],
    'rmse': [overall_mean_rmse_top15],
    'target_dataset': ['COMBINED'],
    'training_samples': [results_top15_df['training_samples'].mean()],
    'test_samples': [results_top15_df['test_samples'].mean()],
    'features_count': [results_top15_df['features_count'].mean()],
    'model_path': [''],
    'status': ['summary']
})

final_top15_df = pd.concat([results_top15_df, overall_row_top15], ignore_index=True)

output_csv_top15 = os.path.join(base_model_dir, 'enhanced_combined_standard_loo_TOP15_results.csv')
final_top15_df.to_csv(output_csv_top15, index=False, float_format='%.3f')

print(f"\n✅ TOP-15 sonuçları kaydedildi: {output_csv_top15}")
print(f"  ⏱️ TOP-15 eğitim süresi: {total_duration_top15/60:.1f} dk")
print(f"  🎯 TOP-15 Mean RMSE: {overall_mean_rmse_top15:.3f}")
print(f"  📊 TOP-15 Median RMSE: {overall_median_rmse_top15:.3f}")
print(f"  📈 TOP-15 RMSE Range: {overall_min_rmse_top15:.3f} - {overall_max_rmse_top15:.3f}")

# 6) Eski ve yeni sonuçları karşılaştırma
#    (Önceki tüm-feature sonuçlarını 'results_df' olarak oluşturmuştuk. Eğer scope dışında kaldıysa yeniden oku.)
if 'results_df' not in globals():
    prev_csv = os.path.join(base_model_dir, 'enhanced_combined_standard_loo_no_flag_results.csv')
    if not os.path.exists(prev_csv):
        raise FileNotFoundError(f"Önceki tüm-feature sonuç CSV bulunamadı: {prev_csv}")
    results_df = pd.read_csv(prev_csv)

# Yalnızca gerçek hasta satırları (OVERALL satırlarını hariç tut)
prev_patients = results_df[~results_df['patient_id'].str.startswith('OVERALL')].copy()
new_patients  = results_top15_df[~results_top15_df['patient_id'].str.startswith('OVERALL')].copy()

cmp = prev_patients[['patient_id','rmse','target_dataset']].rename(
    columns={'rmse':'rmse_all'}
).merge(
    new_patients[['patient_id','rmse']].rename(columns={'rmse':'rmse_top15'}),
    on='patient_id', how='inner'
)

cmp['delta_rmse'] = cmp['rmse_top15'] - cmp['rmse_all']  # negatifse iyileşme
cmp['improved'] = cmp['delta_rmse'] < 0

# Özet metrikler
mean_all = prev_patients['rmse'].mean()
mean_top15 = new_patients['rmse'].mean()
median_all = prev_patients['rmse'].median()
median_top15 = new_patients['rmse'].median()

improve_rate = cmp['improved'].mean() * 100.0
avg_delta = cmp['delta_rmse'].mean()
best_gain_row = cmp.loc[cmp['delta_rmse'].idxmin()]
worst_loss_row = cmp.loc[cmp['delta_rmse'].idxmax()]

print("\n📊 KARŞILAŞTIRMA (Tüm feature'lar vs TOP-15):")
print(f"  🎯 Mean RMSE:  ALL={mean_all:.3f}  →  TOP-15={mean_top15:.3f}  (Δ {mean_top15-mean_all:+.3f})")
print(f"  🎯 Median RMSE: ALL={median_all:.3f} →  TOP-15={median_top15:.3f} (Δ {median_top15-median_all:+.3f})")
print(f"  ✅ İyileşen hasta oranı: {improve_rate:.1f}%")
print(f"  🔎 Ortalama RMSE farkı (TOP-15 - ALL): {avg_delta:+.3f}")
print(f"  🏆 En iyi iyileşme: {best_gain_row['patient_id']}  (Δ {best_gain_row['delta_rmse']:+.3f})")
print(f"  ⚠️ En kötü bozulma: {worst_loss_row['patient_id']} (Δ {worst_loss_row['delta_rmse']:+.3f})")

# Dataset bazında ortalama kıyas
ds_cmp = cmp.groupby('target_dataset').agg(
    mean_rmse_all=('rmse_all','mean'),
    mean_rmse_top15=('rmse_top15','mean'),
    mean_delta=('delta_rmse','mean'),
    improve_rate=('improved','mean')
).reset_index()
ds_cmp['improve_rate'] = (ds_cmp['improve_rate']*100).round(1)

print("\n🧠 DATASET BAZINDA ORTALAMA KARŞILAŞTIRMA:")
for _, r in ds_cmp.iterrows():
    print(f"  • {r['target_dataset']}:  ALL={r['mean_rmse_all']:.3f} → TOP-15={r['mean_rmse_top15']:.3f}  (Δ {r['mean_delta']:+.3f}, iyileşme oranı {r['improve_rate']:.1f}%)")

cmp_csv = os.path.join(base_model_dir, 'rmse_comparison_all_vs_top15_by_patient.csv')
cmp.to_csv(cmp_csv, index=False, float_format='%.3f')

ds_cmp_csv = os.path.join(base_model_dir, 'rmse_comparison_all_vs_top15_by_dataset.csv')
ds_cmp.to_csv(ds_cmp_csv, index=False, float_format='%.3f')

print("\n💾 ÇIKTILAR:")
print(f"  📄 TOP-15 sonuçları: {output_csv_top15}")
print(f"  📄 Hasta bazlı kıyas: {cmp_csv}")
print(f"  📄 Dataset bazlı kıyas: {ds_cmp_csv}")
print("\n✅ TOP-15 ile yeniden eğitim ve karşılaştırma tamamlandı.")

print("Best params for training:", best_params)

print(f"🏆 BEST OPTUNA RMSE: {study.best_value:.4f}")



# —————————————————————————————
# FINAL PATIENT-LEVEL RMSE / MAE
# —————————————————————————————
print("\n" + "="*70)
print("📊 FINAL PATIENT-LEVEL METRICS")
print("="*70)

# 1) ALL FEATURES için gerçek patient-level RMSE ve MAE
prev_patients = results_df[~results_df['patient_id'].astype(str).str.startswith('OVERALL')].copy()

y_true_all = prev_patients["true_score"].values.astype(float)
y_pred_all = prev_patients["predicted_score"].values.astype(float)

patient_level_rmse_all = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
patient_level_mae_all = np.mean(np.abs(y_true_all - y_pred_all))

print("\n📌 ALL FEATURES:")
print(f"  Patients          : {len(prev_patients)}")
print(f"  Patient-level RMSE: {patient_level_rmse_all:.4f}")
print(f"  Patient-level MAE : {patient_level_mae_all:.4f}")

# 2) TOP-K FEATURES için gerçek patient-level RMSE ve MAE
top_patients = results_top15_df[~results_top15_df['patient_id'].astype(str).str.startswith('OVERALL')].copy()

y_true_top = top_patients["true_score"].values.astype(float)
y_pred_top = top_patients["predicted_score"].values.astype(float)

patient_level_rmse_top = np.sqrt(mean_squared_error(y_true_top, y_pred_top))
patient_level_mae_top = np.mean(np.abs(y_true_top - y_pred_top))

print(f"\n📌 TOP-{top_k} FEATURES:")
print(f"  Patients          : {len(top_patients)}")
print(f"  Patient-level RMSE: {patient_level_rmse_top:.4f}")
print(f"  Patient-level MAE : {patient_level_mae_top:.4f}")

# 3) Karşılaştırma
print("\n📊 FINAL COMPARISON:")
print(f"  RMSE: ALL={patient_level_rmse_all:.4f} → TOP-{top_k}={patient_level_rmse_top:.4f} "
      f"(Δ {patient_level_rmse_top - patient_level_rmse_all:+.4f})")

print(f"  MAE : ALL={patient_level_mae_all:.4f} → TOP-{top_k}={patient_level_mae_top:.4f} "
      f"(Δ {patient_level_mae_top - patient_level_mae_all:+.4f})")

# 4) Bu metrikleri CSV olarak kaydet
final_metrics_df = pd.DataFrame([
    {
        "setting": "ALL_FEATURES",
        "num_patients": len(prev_patients),
        "patient_level_rmse": patient_level_rmse_all,
        "patient_level_mae": patient_level_mae_all
    },
    {
        "setting": f"TOP_{top_k}_FEATURES",
        "num_patients": len(top_patients),
        "patient_level_rmse": patient_level_rmse_top,
        "patient_level_mae": patient_level_mae_top
    }
])

final_metrics_csv = os.path.join(base_model_dir, "final_patient_level_rmse_mae.csv")
final_metrics_df.to_csv(final_metrics_csv, index=False, float_format="%.6f")

print(f"\n💾 Final patient-level metrics saved to: {final_metrics_csv}")
print("="*70)
