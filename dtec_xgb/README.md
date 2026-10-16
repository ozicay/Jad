# DTEC / D3TEC SM-27: ortak random-11, 12 XGBoost konfigürasyonu

Kaynak: proje kökündeki `optuna_xgb_is09_top15.py`.
Değiştirilmeden dondurulmuş kopya `reference/optuna_xgb_is09_top15.py` içindedir.
SHA256: `e78e97f99278321b3d12035b638e874f2088f8cb5b828a08af7313b188e4fb51`.
Pooled anxiety eğitim kodu bu deneye kopyalanmadı/import edilmedi.
Yalnız konfigürasyon düzeyindeki `srun` scheduler yaklaşımı ve final minimum-AE
modelini saklama davranışı alındı.

## Veri ve ortak seçim

`data/` altındaki gerçek girdiler:

| Dosya | Vektör kolonu | Boyut |
|---|---|---:|
| metadata_with_mfcc_dtec | features | 608 |
| metadata_with_egemaps_dtec | egemaps_features | 88 |
| metadata_with_is09_dtec | is09_features | 384 |

`metadata.csv` ortak chunk metadata'sıdır. Dört dosya CSV biçimindedir;
feature dosyalarının `.csv` uzantısı yoktur. Hasta kimliği `Participant_ID`,
chunk kimliği `(Participant_ID, chunk_index)`, hedef `depresyon_skoru`dur;
hedef ayrıca `PHQ-9 Score` ile doğrulanır. Yaş veya başka metadata feature olarak
eklenmez. Kaynak yalnız SM-27'dir; `age_group` yoktur.

62 hasta, 1643 tam 20 saniyelik chunk, hasta başına 14–45 chunk doğrulandı.
Kayıtlar hasta içinde görev sırasıyla birleştirilmiş olduğundan bazı chunk'lar
yanıt sınırlarını geçer. Kaynak zaman aralıkları metadata'da korunur.

Seçim yalnız `controller.py --prepare-only` sırasında bir kez yapılır:
`RandomState(42)`, sıralı hasta ve sayısal chunk kimlikleri, tekrarsız 11 seçim.
Seçim etikete bakmaz. `selection/selected_11_chunks.csv` toplam 682 satır içerir;
12 konfigürasyonun tamamı aynı kaydedilmiş listeyi okur. Worker'larda RNG veya
yeniden örnekleme yoktur. Hash, hasta/chunk kapsamı ve tüm metadata eşleşmeleri
her worker başlamadan doğrulanır. Metadata satır sırası değişebilir; içerik
değişirse doğrulama durur. 11'den az chunk, eksik/duplicate satır, farklı skor,
boş/bozuk feature veya aileler arası uyuşmazlık açık hata üretir; hasta atılmaz.

`file_name` kaynak WAV'ın staging yoludur; model eğitimi kaydedilmiş feature
vektörlerini okur. MN5'te WAV'ların bu staging yolunda bulunması gerekmez.
MN5'e yalnız bu dört metadata ve ortak seçim aktarılır; ham ses gerekmez.
Klinik metadata, seçim ve deney çıktıları Git dışında tutulur.

## Korunan bilimsel akış ve açık farklar

Her aile için Top-5/10/15/All: toplam 12 bağımsız Optuna study.

- Search space aynıdır: learning_rate 0.001–0.1 log, max_depth 4–14 integer,
  n_estimators 100–500 integer. Objective reg:squarederror, trial tree_method hist,
  XGBoost random_state 42. Optuna'nın default seedsiz sampler'ı aynen korunur.
- **Kullanıcının son talimatıyla 40 trial**; orijinalde 20'dir.
- Top-K trial: full-feature LOPO → importance'ı CSV'ye yaz/oku → fold importance
  ortalamasını azalan sırala → head(K) → seçili indeksleri sayısal sırala →
  Top-K LOPO. Objective tek hasta RMSE'lerinin ortalamasıdır; toplu final RMSE
  ile aynı objective değildir. Tek hastalık RMSE sayısal olarak o hastanın AE'sidir.
- Top-K final: best_params ile full-feature LOPO → importance CSV'sini
  **6 ondalık** yaz/oku → aynı Top-K seçimi → final Top-K LOPO.
  Final modelde tree_method referanstaki gibi ayrıca verilmez.
- **All:** kullanıcı talimatıyla her trial yalnız bir full-feature LOPO;
  Optuna sonrası yalnız bir full-feature final LOPO. İkinci LOPO/retraining yoktur.
- Hasta train/test ayrımı aynı LOPO maskesidir. Her fold 61 hasta/671 segment
  eğitim, 1 hasta/11 segment test. Tahmin 11 segment olasılığı değil, skor
  regresyon tahminlerinin medyanıdır. Final RMSE ve MAE tüm 62 hasta tahmininden
  referanstaki formüllerle, yuvarlanmamış değerlerden hesaplanır.
- Ek scaling, imputation, weighting, clipping, erken durdurma, split veya feature
  ekleme yapılmaz. Referanstaki numeric parser korunur; bozuk veri sessizce atılmaz.
- Importance tüm full-feature LOPO fold'larından global ortalanır; bu kaynak
  protokol nested feature selection değildir. Bu uygulama nested değerlendirme
  veya leakage içermeyen yeni bir yöntem olarak sunulmamalıdır.
- Paralellik yalnız konfigürasyonlar arasındadır. Referans final fold thread'leri
  seri hale getirilmiştir; model başına n_jobs=20 operasyonel CPU sınırıdır.
- Top-K/All final LOPO tamamlanınca yalnız en düşük held-out AE fold modeli
  saklanır. Eşit AE'de sıralı ilk hasta seçilir. Bu, bütün hastalara en iyi
  genelleyen model olduğu anlamına gelmez. Trial modeli kaydedilmez.
- Feature isimleri referanstaki `feat_i` konumsal isimleridir; vektör CSV'lerinde
  orijinal akustik feature başlıkları bulunmadığından bunlar uydurulmaz.

## GPU scheduler ve çıktılar

Tek node, 4 GPU, 4 task × 20 CPU, toplam 160 GiB RAM. Her konfigürasyon
`srun --exclusive --exact --ntasks=1 --cpus-per-task=20 --mem=40G
--gpus-per-task=1 --gpu-bind=single:1` ile çalışır. En fazla dört worker aktiftir;
bitiren worker'ın yerine sıradaki konfigürasyon başlar. Slurm her worker'a
tek CUDA_VISIBLE_DEVICES verir, XGBoost o process içinde `cuda:0` kullanır.
Booster'ın CUDA yerine CPU'ya düşmesi açık hatadır. NumPy input nedeniyle
prediction DMatrix uyarısı GPU eğitimin CPU'ya düştüğü anlamına gelmez.

Mevcut, salt okunur MN5 runtime'ı yeniden kullanılır:
Python/3.12.3-GCCcore-13.3.0; anxiety_age_pool/env_site ve py312_job_site.
Yeni ortam/paket kurulmaz. Veri veya pooled config bu ortamdan alınmaz.

Tüm yeni source/data/selection/log/results MN5 kökü:
`/gpfs/projects/etur92/ozu150751/jad/dtec_xgb/`.

Her Slurm jobu `runs/<timestamp>_job<ID>/` oluşturur. Önceki attempt ezilmez;
yarım kalan attempt otomatik resume edilmez. Her konfigürasyonun kendi
output klasörü, SQLite study, optuna_trials.csv, best_params.json,
trial kayıtları, full/topk hasta ve segment tahminleri, patient_predictions.csv
(true_score/predicted_score/AE), final_metrics.csv, selected_features.csv,
best_model.pkl, best_model_info.json, execution.log ve COMPLETE.json çıktısı olur.
Global execution logları `runs/<attempt>/logs/<config>.log`;
12 satırlık özet `runs/<attempt>/experiment_summary.csv` olur.
Her config'te ortak seçim dosyasının aynı hash'li kopyası bulunur.

## Staging doğrulama komutları

```bash
python dtec_xgb/controller.py --prepare-only
python dtec_xgb/controller.py --validate-only
python -m unittest discover -s dtec_xgb -p test_protocol.py -v
bash -n dtec_xgb/run_dtec_four_gpu.slurm
python dtec_xgb/smoke_gpu.py
```

Testler sentetik CPU modelleriyle referans AST'sinden çıkarılan gerçek objective
ve final fold fonksiyonlarını karşılaştırır; üç Top-K objective, final Top-15
indeks/tahmin eşitliği, 12 final konfigürasyon, minimum-AE model reload, All'da
tek LOPO, scheduler'ın dört worker sınırı ve sıra değişiminden bağımsız ortak
seçim kontrol edilir. Gerçek metadata üzerinde model fit edilmez.

## MN5 çalıştırma — yalnız kullanıcı ayrıca başlatmayı onayladıktan sonra

GitHub commit'i ile birebir aynı kaynak, `SOURCE_COMMIT` ve `SHA256SUMS`
deployment sırasında oluşturulur. MN5'te bu dosyalar doğrulanmadan başlamaz.
Kaynak ve özel veri rsync ile transfer host üzerinden aktarılır; Git'e veri konmaz.

```bash
cd /gpfs/projects/etur92/ozu150751/jad/dtec_xgb
mkdir -p logs
sbatch run_dtec_four_gpu.slurm
```

Bu komut dokümantasyondur; hazırlık/doğrulama sırasında `sbatch` çalıştırılmaz.
Dört gerçek MN5 GPU'sunun izolasyonu ancak onaylanmış job içinde doğrulanabilir.
