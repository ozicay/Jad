"""Training-identical configurable experiment with training-only SHAP ranking."""
import argparse
import json
from pathlib import Path

FEATURE_SCHEMAS = {
    "mfcc": ("features", 608),
    "egemaps": ("egemaps_features", 88),
    "is09": ("is09_features", 384),
}


def load_config(path):
    config = json.loads(Path(path).read_text())
    required = {"experiment_name", "csv_path", "feature_type", "feature_column",
                "expected_feature_count", "target_column", "top_k", "output_dir"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")
    schema = FEATURE_SCHEMAS.get(config["feature_type"])
    if schema != (config["feature_column"], config["expected_feature_count"]):
        raise ValueError("Feature column/dimension does not match feature type")
    if config["target_column"] != "anksiyete_skoru":
        raise ValueError("Target must be anksiyete_skoru")
    if config["top_k"] is not None and (type(config["top_k"]) is not int or config["top_k"] not in (5, 10, 15)):
        raise ValueError("top_k must be 5, 10, 15, or null")
    if config["top_k"] is None:
        raise ValueError("Fixed-parameter SHAP runs only Top-5/10/15")
    params = config.get("fixed_params", {})
    if set(params) != {"learning_rate", "max_depth", "n_estimators"}:
        raise ValueError("Exactly the three verified gain parameters are required")
    if not (1e-3 <= params["learning_rate"] <= .1 and type(params["max_depth"]) is int
            and 4 <= params["max_depth"] <= 14 and type(params["n_estimators"]) is int
            and 100 <= params["n_estimators"] <= 500):
        raise ValueError("Fixed gain parameters are outside the baseline search ranges")
    if not config.get("gain_parameter_source") or not config.get("gain_output_dir"):
        raise ValueError("Verified gain parameter source and output directory required")
    return config


def mean_abs_training_shap(model, X_train):
    """One mean absolute TreeSHAP vector, using this fold's training rows only."""
    import numpy as np
    import shap
    explainer = shap.TreeExplainer(model)
    if callable(explainer):
        shap_values = explainer(X_train)
    else:
        shap_values = explainer.shap_values(X_train)
    shap_array = np.asarray(getattr(shap_values, "values", shap_values))
    if shap_array.shape != X_train.shape:
        raise ValueError(f"Expected SHAP shape {X_train.shape}, got {shap_array.shape}")
    shap_importance = np.mean(np.abs(shap_array), axis=0)
    return shap_importance


def run_experiment(config):
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

    # —————————————————————————————
    # Ayarlar
    # —————————————————————————————home/

    depr_only_csv = config["csv_path"]
    feature_column = config["feature_column"]
    expected_feature_count = config["expected_feature_count"]
    target_column = config["target_column"]
    top_k = config["top_k"]
    #extra = '/home/utku/emo_depression/xgboost/metadata_w2v2_scores_mfcc_with_3_feat_comorbid.csv'
    base_model_dir = config["output_dir"]
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

    def train_loo_model_with_importance(loo_args):
        """Standard LOO + Simple Feature Importance"""
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
                tree_method='hist',
                device='cuda' if 'cuda' in CUDA_DEVICE else 'cpu'
            )


            model.fit(X_train, y_train)

            target_predictions = model.predict(X_target_test)
            predicted_median = np.median(target_predictions)

            rmse = np.sqrt(mean_squared_error([true_score], [predicted_median]))

            feature_importance = mean_abs_training_shap(model, X_train) if X_train.shape[1] == expected_feature_count else None

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
                'feature_importance': feature_importance,
                'status': 'success'
            }

            if feature_importance is not None:
                cluster_importance = feature_importance[-1]
                cluster_rank = len(feature_importance) - np.argsort(feature_importance)[::-1].tolist().index(len(feature_importance)-1)

                print(f"      🎯 Cluster_id importance: {cluster_importance:.4f} (rank: {cluster_rank})")
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
    df[feature_column] = df[feature_column].apply(parse_features)
    df = df[df[feature_column].notna()].reset_index(drop=True)

    X_all = np.vstack(df[feature_column].values)
    if X_all.shape[1] != expected_feature_count:
        raise ValueError(f"{config['feature_type']}: expected {expected_feature_count} features, got {X_all.shape[1]}")
    #X_all = np.hstack([X_features, X_meta])



    y_all = df[target_column].values
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
        target_column: 'first',
        'dataset': 'first'
    }).to_dict('index')

    print(f"\n📊 DATASET INFO:")
    print(f"  📦 Total samples: {len(df)}")
    print(f"  👥 Total patients: {len(unique_pids)}")
    print(f"  🎯 Features: {X_all.shape[1]}")
    print(f"  📊 LOO Strategy: {len(unique_pids)} models (119 train → 1 test each)")


    best_params = config["fixed_params"].copy()
    print("Fixed gain parameters:", best_params)
    print("Gain parameter source:", config["gain_parameter_source"])
    Path(base_model_dir, "fixed_params.json").write_text(json.dumps(best_params, indent=2) + "\n")
    Path(base_model_dir, "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    # —————————————————————————————
    # Main Standard LOO training loop ()
    # —————————————————————————————
    print(f"\n🚀 STARTING STANDARD LOO TRAINING...")

    # Prepare all LOO jobs
    print(f"\n🚀 PREPARING LOO JOBS WITH SIMPLE FEATURE IMPORTANCE...")

    training_jobs = []
    start_time = datetime.now()

    for target_patient in unique_pids:
        target_mask = groups_all == target_patient
        train_mask = ~target_mask

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]

        X_target_test = X_all[target_mask]
        y_target_test = y_all[target_mask]

        true_score = patient_info[target_patient][target_column]
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
        print(f"\n📊 CREATING FEATURE IMPORTANCE CSV...")

        # Create feature importance dataframe
        importance_data = []

        for result in successful_results:
            if 'feature_importance' in result:
                importance_array = result['feature_importance']

                # Create row for this patient
                row = {
                    'patient_id': result['patient_id'],
                    'rmse': result['rmse'],
                    'true_score': result['true_score'],
                    'predicted_score': result['predicted_score'],
                    'target_dataset': result['target_dataset']
                }

                # Add each feature importance as feat_0, feat_1, feat_2...
                for i, importance in enumerate(importance_array):
                    row[f'feat_{i}'] = importance

                importance_data.append(row)

        # Create DataFrame
        importance_df = pd.DataFrame(importance_data)

        # Save feature importance CSV
        importance_csv = os.path.join(base_model_dir, 'shap_importance_all_models.csv')
        importance_df.to_csv(importance_csv, index=False, float_format='%.6f')

        print(f"✅ Feature importance CSV saved: {importance_csv}")
        print(f"📊 Shape: {importance_df.shape}")
        print(f"📊 Columns: patient_id, rmse, true_score, predicted_score, target_dataset, feat_0 to feat_{X_all.shape[1]-1}")

        # —————————————————————————————
        # Quick Analysis
        # —————————————————————————————
        print(f"\n🔍 QUICK FEATURE IMPORTANCE ANALYSIS:")

        # Get feature columns only
        feature_cols = [col for col in importance_df.columns if col.startswith('feat_')]
        feature_importance_only = importance_df[feature_cols]

        # Calculate average importance per feature
        avg_importance = feature_importance_only.mean().sort_values(ascending=False)

        print(f"📊 TOP 10 MOST IMPORTANT FEATURES (AVERAGE):")
        for i, (feature, importance) in enumerate(avg_importance.head(90).items()):
            print(f"  {i+1:2d}. {feature}: {importance:.6f}")

        # Cluster_id analysis (last feature)
        cluster_feature = f'feat_{X_all.shape[1]-1}'  # Last feature is cluster_id
        cluster_avg_importance = avg_importance[cluster_feature]
        cluster_rank = avg_importance.index.get_loc(cluster_feature) + 1

        print(f"\n🎯 CLUSTER_ID (feat_{X_all.shape[1]-1}) ANALYSIS:")
        print(f"  📊 Average importance: {cluster_avg_importance:.6f}")
        print(f"  📊 Rank: {cluster_rank} out of {len(avg_importance)}")
        print(f"  📊 Relative to top feature: {(cluster_avg_importance / avg_importance.iloc[0] * 100):.1f}%")

        if cluster_rank <= 10:
            print(f"  ✅ Cluster_id is in TOP 10 most important features!")
        elif cluster_rank <= 20:
            print(f"  ✅ Cluster_id is in TOP 20 most important features!")
        else:
            print(f"  ⚠️ Cluster_id importance is relatively low (rank {cluster_rank})")

        # Save summary
        summary_data = {
            'total_features': len(feature_cols),
            'cluster_feature_name': cluster_feature,
            'cluster_avg_importance': cluster_avg_importance,
            'cluster_rank': cluster_rank,
            'top_feature': avg_importance.index[0],
            'top_feature_importance': avg_importance.iloc[0]
        }

        summary_csv = os.path.join(base_model_dir, 'shap_importance_summary.csv')
        pd.DataFrame([summary_data]).to_csv(summary_csv, index=False, float_format='%.6f')

        print(f"\n💾 FILES SAVED:")
        print(f"  📊 Feature importance (all models): {importance_csv}")
        print(f"  📋 Summary: {summary_csv}")
    else:
        print(f"\n❌ No successful results!")

    print(f"\n🚀 ENHANCED COMBINED STANDARD LOO (NO DATASET FLAG) COMPLETED!")
    print(f"  📊 Strategy: Combined metadata → Standard LOO (NO dataset flag)")
    print(f"  📁 Total models trained: {len(unique_pids)}")
    print(f"  🎯 Features: {X_all.shape[1]} (original only, NO dataset flag)")
    print(f"  ⏱️ Training completed in {total_duration/60:.1f} minutes")
    print(f"  💾 Results saved to: {output_csv}")

    if top_k is not None:
        # —————————————————————————————
        # TOP-{top_k} FEATURE İLE TEKRAR EĞİTİM ve KARŞILAŞTIRMA
        # —————————————————————————————
        print(f"\n🚀 RETRAIN: En yüksek önem skoruna sahip TOP-{top_k} feature ile LOO başlıyor...")

        # 1) Importance CSV'den ortalama önemleri oku (mevcut değişkenler varsa onları kullanır)
        importance_csv = os.path.join(base_model_dir, 'shap_importance_all_models.csv')
        if not os.path.exists(importance_csv):
            raise FileNotFoundError(f"Importance CSV bulunamadı: {importance_csv}")

        importance_df = pd.read_csv(importance_csv)
        feature_cols = [c for c in importance_df.columns if c.startswith('feat_')]

        avg_importance = importance_df[feature_cols].mean().sort_values(ascending=False)
        shap_ranking = pd.DataFrame({
            "rank": np.arange(1, len(avg_importance) + 1),
            "feature": avg_importance.index,
            "feature_index": [int(f.split("_")[1]) for f in avg_importance.index],
            "mean_abs_shap": avg_importance.values,
        })
        shap_ranking.to_csv(os.path.join(base_model_dir, "shap_global_ranking.csv"), index=False)

        top_features = avg_importance.head(top_k).index.tolist()
        shap_ranking.head(top_k).to_csv(os.path.join(base_model_dir, f"selected_top{top_k}_features.csv"), index=False)
        print(top_features)

        # feat_123 -> 123 indekslerine çevir
        top_feature_indices = sorted([int(f.split('_')[1]) for f in top_features])
        assert len(top_feature_indices) == top_k

        print(f"  🔝 Seçilen feature sayısı: {len(top_feature_indices)}")
        print(f"  🧩 İlk 10: {top_feature_indices[:10]}{' ...' if len(top_feature_indices) > 10 else ''}")

        # 2) Orijinal X_all'dan sadece TOP-{top_k} kolonları al
        X_selected = X_all[:, top_feature_indices]
        print(f"  📊 X_selected shape: {X_selected.shape} (samples={X_selected.shape[0]}, features={X_selected.shape[1]})")

        # 3) LOO işler listesini yeniden hazırla
        print(f"\n🚀 TOP-{top_k} için LOO işleri hazırlanıyor...")
        training_jobs_selected = []
        start_time_selected = datetime.now()

        for target_patient in unique_pids:
            target_mask = groups_all == target_patient
            train_mask = ~target_mask

            X_train = X_selected[train_mask]
            y_train = y_all[train_mask]

            X_target_test = X_selected[target_mask]
            y_target_test = y_all[target_mask]

            true_score = patient_info[target_patient][target_column]
            target_dataset = patient_info[target_patient]['dataset']

            patient_model_dir = os.path.join(base_model_dir, f'patient_{target_patient}_top{top_k}')
            os.makedirs(patient_model_dir, exist_ok=True)

            training_jobs_selected.append((
                target_patient, X_train, y_train, X_target_test, y_target_test,
                true_score, target_dataset, patient_model_dir
            ))

        print(f"  📦 Hazır LOO iş sayısı: {len(training_jobs_selected)}")
        print(f"  📊 Her model feature sayısı: {X_selected.shape[1]}")

        # 4) Paralel eğitim
        all_results_selected = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            print(f"  🚀 Paralel LOO (TOP-{top_k}) eğitim başlıyor...")
            future_to_patient = {executor.submit(train_loo_model_with_importance, job): job[0] for job in training_jobs_selected}

            for future in concurrent.futures.as_completed(future_to_patient):
                target_patient = future_to_patient[future]
                try:
                    result = future.result()
                    if result:
                        all_results_selected.append(result)
                        if len(all_results_selected) % 10 == 0:
                            print(f"    ✅ {len(all_results_selected)}/{len(training_jobs_selected)} tamamlandı...")
                except Exception as e:
                    print(f"    ❌ Job failed for {target_patient}: {e}")

        total_duration_selected = (datetime.now() - start_time_selected).total_seconds()

        # 5) Sonuçların kaydı
        successful_selected = [r for r in all_results_selected if r.get('status') == 'success']
        if not successful_selected:
            raise RuntimeError(f"TOP-{top_k} LOO çalışmasından başarılı sonuç gelmedi.")

        results_selected_df = pd.DataFrame(successful_selected)

        # Özet satırı
        overall_mean_rmse_selected = results_selected_df['rmse'].mean()
        overall_std_rmse_selected = results_selected_df['rmse'].std()
        overall_median_rmse_selected = results_selected_df['rmse'].median()
        overall_min_rmse_selected = results_selected_df['rmse'].min()
        overall_max_rmse_selected = results_selected_df['rmse'].max()

        overall_row_selected = pd.DataFrame({
            'patient_id': [f'OVERALL_AVERAGE_TOP{top_k}'],
            'true_score': [results_selected_df['true_score'].mean()],
            'predicted_score': [results_selected_df['predicted_score'].mean()],
            'rmse': [overall_mean_rmse_selected],
            'target_dataset': ['COMBINED'],
            'training_samples': [results_selected_df['training_samples'].mean()],
            'test_samples': [results_selected_df['test_samples'].mean()],
            'features_count': [results_selected_df['features_count'].mean()],
            'model_path': [''],
            'status': ['summary']
        })

        final_selected_df = pd.concat([results_selected_df, overall_row_selected], ignore_index=True)

        output_csv_selected = os.path.join(base_model_dir, f'enhanced_combined_standard_loo_TOP{top_k}_results.csv')
        final_selected_df.to_csv(output_csv_selected, index=False, float_format='%.3f')

        print(f"\n✅ TOP-{top_k} sonuçları kaydedildi: {output_csv_selected}")
        print(f"  ⏱️ TOP-{top_k} eğitim süresi: {total_duration_selected/60:.1f} dk")
        print(f"  🎯 TOP-{top_k} Mean RMSE: {overall_mean_rmse_selected:.3f}")
        print(f"  📊 TOP-{top_k} Median RMSE: {overall_median_rmse_selected:.3f}")
        print(f"  📈 TOP-{top_k} RMSE Range: {overall_min_rmse_selected:.3f} - {overall_max_rmse_selected:.3f}")

        # 6) Eski ve yeni sonuçları karşılaştırma
        #    (Önceki tüm-feature sonuçlarını 'results_df' olarak oluşturmuştuk. Eğer scope dışında kaldıysa yeniden oku.)
        if 'results_df' not in locals():
            prev_csv = os.path.join(base_model_dir, 'enhanced_combined_standard_loo_no_flag_results.csv')
            if not os.path.exists(prev_csv):
                raise FileNotFoundError(f"Önceki tüm-feature sonuç CSV bulunamadı: {prev_csv}")
            results_df = pd.read_csv(prev_csv)

        # Yalnızca gerçek hasta satırları (OVERALL satırlarını hariç tut)
        prev_patients = results_df[~results_df['patient_id'].str.startswith('OVERALL')].copy()
        new_patients  = results_selected_df[~results_selected_df['patient_id'].str.startswith('OVERALL')].copy()

        cmp = prev_patients[['patient_id','rmse','target_dataset']].rename(
            columns={'rmse':'rmse_all'}
        ).merge(
            new_patients[['patient_id','rmse']].rename(columns={'rmse':'rmse_selected'}),
            on='patient_id', how='inner'
        )

        cmp['delta_rmse'] = cmp['rmse_selected'] - cmp['rmse_all']  # negatifse iyileşme
        cmp['improved'] = cmp['delta_rmse'] < 0

        # Özet metrikler
        mean_all = prev_patients['rmse'].mean()
        mean_selected = new_patients['rmse'].mean()
        median_all = prev_patients['rmse'].median()
        median_selected = new_patients['rmse'].median()

        improve_rate = cmp['improved'].mean() * 100.0
        avg_delta = cmp['delta_rmse'].mean()
        best_gain_row = cmp.loc[cmp['delta_rmse'].idxmin()]
        worst_loss_row = cmp.loc[cmp['delta_rmse'].idxmax()]

        print(f"\n📊 KARŞILAŞTIRMA (Tüm feature'lar vs TOP-{top_k}):")
        print(f"  🎯 Mean RMSE:  ALL={mean_all:.3f}  →  TOP-{top_k}={mean_selected:.3f}  (Δ {mean_selected-mean_all:+.3f})")
        print(f"  🎯 Median RMSE: ALL={median_all:.3f} →  TOP-{top_k}={median_selected:.3f} (Δ {median_selected-median_all:+.3f})")
        print(f"  ✅ İyileşen hasta oranı: {improve_rate:.1f}%")
        print(f"  🔎 Ortalama RMSE farkı (TOP-{top_k} - ALL): {avg_delta:+.3f}")
        print(f"  🏆 En iyi iyileşme: {best_gain_row['patient_id']}  (Δ {best_gain_row['delta_rmse']:+.3f})")
        print(f"  ⚠️ En kötü bozulma: {worst_loss_row['patient_id']} (Δ {worst_loss_row['delta_rmse']:+.3f})")

        # Dataset bazında ortalama kıyas
        ds_cmp = cmp.groupby('target_dataset').agg(
            mean_rmse_all=('rmse_all','mean'),
            mean_rmse_selected=('rmse_selected','mean'),
            mean_delta=('delta_rmse','mean'),
            improve_rate=('improved','mean')
        ).reset_index()
        ds_cmp['improve_rate'] = (ds_cmp['improve_rate']*100).round(1)

        print("\n🧠 DATASET BAZINDA ORTALAMA KARŞILAŞTIRMA:")
        for _, r in ds_cmp.iterrows():
            print(f"  • {r['target_dataset']}:  ALL={r['mean_rmse_all']:.3f} → TOP-{top_k}={r['mean_rmse_selected']:.3f}  (Δ {r['mean_delta']:+.3f}, iyileşme oranı {r['improve_rate']:.1f}%)")

        cmp_csv = os.path.join(base_model_dir, f'rmse_comparison_all_vs_top{top_k}_by_patient.csv')
        cmp.to_csv(cmp_csv, index=False, float_format='%.3f')

        ds_cmp_csv = os.path.join(base_model_dir, f'rmse_comparison_all_vs_top{top_k}_by_dataset.csv')
        ds_cmp.to_csv(ds_cmp_csv, index=False, float_format='%.3f')

        print("\n💾 ÇIKTILAR:")
        print(f"  📄 TOP-{top_k} sonuçları: {output_csv_selected}")
        print(f"  📄 Hasta bazlı kıyas: {cmp_csv}")
        print(f"  📄 Dataset bazlı kıyas: {ds_cmp_csv}")
        print(f"\n✅ TOP-{top_k} ile yeniden eğitim ve karşılaştırma tamamlandı.")

        print("Best params for training:", best_params)




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

    if top_k is not None:
        # 2) TOP-K FEATURES için gerçek patient-level RMSE ve MAE
        top_patients = results_selected_df[~results_selected_df['patient_id'].astype(str).str.startswith('OVERALL')].copy()

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
    metrics_rows = [{
        "setting": "ALL_FEATURES",
        "num_patients": len(prev_patients),
        "patient_level_rmse": patient_level_rmse_all,
        "patient_level_mae": patient_level_mae_all
    }]
    if top_k is not None:
        metrics_rows.append({
            "setting": f"TOP_{top_k}_FEATURES",
            "num_patients": len(top_patients),
            "patient_level_rmse": patient_level_rmse_top,
            "patient_level_mae": patient_level_mae_top
        })
    else:
        X_selected = X_all
        importance_df = pd.read_csv(os.path.join(base_model_dir, "shap_importance_all_models.csv"))
        feature_cols = [c for c in importance_df.columns if c.startswith("feat_")]
        avg_importance = importance_df[feature_cols].mean().sort_values(ascending=False)
        shap_ranking = pd.DataFrame({
            "rank": np.arange(1, len(avg_importance) + 1),
            "feature": avg_importance.index,
            "feature_index": [int(f.split("_")[1]) for f in avg_importance.index],
            "mean_abs_shap": avg_importance.values,
        })
        shap_ranking.to_csv(os.path.join(base_model_dir, "shap_global_ranking.csv"), index=False)
    final_metrics_df = pd.DataFrame(metrics_rows)

    final_metrics_csv = os.path.join(base_model_dir, "final_patient_level_rmse_mae.csv")
    final_metrics_df.to_csv(final_metrics_csv, index=False, float_format="%.6f")

    print(f"\n💾 Final patient-level metrics saved to: {final_metrics_csv}")
    print("="*70)





    gain_metrics = pd.read_csv(Path(config["gain_output_dir"], "final_patient_level_rmse_mae.csv"))
    print("GAIN patient-level metrics:", gain_metrics.to_dict("records"))
    print("SHAP patient-level metrics:", final_metrics_df.to_dict("records"))
    gain_results = Path(config["gain_output_dir"], f"enhanced_combined_standard_loo_TOP{top_k}_results.csv")
    if gain_results.exists():
        gain = pd.read_csv(gain_results)
        gain = gain[~gain.patient_id.astype(str).str.startswith("OVERALL")]
        comparison = gain[["patient_id", "true_score", "predicted_score"]].rename(columns={"predicted_score": "gain_prediction"}).merge(
            top_patients[["patient_id", "true_score", "predicted_score"]].rename(columns={"true_score": "shap_true_score", "predicted_score": "shap_prediction"}),
            on="patient_id", validate="one_to_one")
        if len(comparison) != len(top_patients) or not np.allclose(comparison.true_score, comparison.shap_true_score, atol=0.000501, rtol=0):
            raise ValueError("Gain/SHAP patient identity or target mismatch")
        comparison = comparison.drop(columns="shap_true_score")
        comparison["gain_abs_error"] = np.abs(comparison.true_score - comparison.gain_prediction)
        comparison["shap_abs_error"] = np.abs(comparison.true_score - comparison.shap_prediction)
        comparison.to_csv(Path(base_model_dir, "gain_vs_shap_by_patient.csv"), index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_experiment(load_config(args.config))
