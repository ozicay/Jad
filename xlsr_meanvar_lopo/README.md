# XLS-R 300M mean + variance ile LOPO regresyon

Bu klasör, mevcut wav2vec2/XLS-R deneyinin başka bir bilgisayarda kullanılabilir sürümüdür. Model **`facebook/wav2vec2-xls-r-300m`**. Her ses segmentinden **1024 zaman ortalaması + 1024 popülasyon varyansı = 2048 özellik** çıkarır. Sonra XGBoost ile depresyon veya anksiyete skoru tahmini yapar.

Gerçek metadata, hasta kaydı, ses, embedding veya eğitilmiş model içermez. `examples/metadata_example.csv` tamamen sentetiktir; örnek ses üretici de konuşma yerine yapay ton üretir.

## 1. Kurulum

Python 3.12/Linux üzerinde aşağıdaki sabit paket sürümleri ve PyTorch 2.5.1 ile yerel CPU kontrolleri yapıldı. Eğitim betiği `fcntl` kullandığından Windows üzerinde WSL kullanın.

```bash
git clone https://github.com/ozicay/Jad.git
cd Jad
cd xlsr_meanvar_lopo
python -m venv .venv
source .venv/bin/activate
# CPU için:
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

GPU kullanacaksanız CPU wheel komutu yerine donanımınıza uygun PyTorch CUDA wheel'ini kurun. XGBoost sürümü `3.1.2` olarak sabittir; ortamınızdaki sürücü/CUDA uyumunu kontrol edin. Mevcut uygun ortamınız varsa onu doğrudan kullanabilirsiniz. Eğitim ve feature çıkarımı farklı Python ortamlarında da çalışabilir.

## 2. Metadata biçimi

Her satır bir ses segmentidir. Aynı hastanın tüm segmentlerinde aynı `patient_id` ve hedef skoru bulunmalıdır.

```csv
file_name,patient_id,depresyon_skoru,anksiyete_skoru,label
p001_seg01.wav,p001,12,8,example
p001_seg02.wav,p001,12,8,example
p002_seg01.wav,p002,20,14,example
```

- `file_name`: tam ses yolu veya `--audio-root` altında göreli yol. `--audio-root` verilmezse CSV'nin bulunduğu klasör kullanılır. CSV başka bilgisayara taşınıyorsa tam yolları değiştirin ya da göreli yol kullanın.
- `patient_id`: dosya adından türetilmez; doğrudan bu sütun kullanılır. Baştaki/sondaki boşluklar ve büyük/küçük harf farkları normalize edilir. Farklı kişilerin kimlikleri normalize edilince aynı olmamalıdır.
- `depresyon_skoru` / `anksiyete_skoru`: ilgili eğitim için gereken hedef sütunu. Diğer skorun bulunması zorunlu değildir. Sayısal ölçek değiştirilmez.
- Diğer metadata sütunları korunur, ancak XGBoost girdisine eklenmez. `label`, yaş, diğer skorlar, önceki tahminler ve akustik özellikler model girdisi değildir.

Eğitim için en az üç hasta gerekir. Geriatri ve non-geriatri ayrı CSV ve ayrı eğitimlerdir; otomatik yaş/grup/etiket filtresi uygulanmaz. İki hedef için aynı sesler kullanılıyorsa feature'ları bir kez çıkarıp iki eğitimde kullanabilirsiniz; extractor skorları kullanmaz.

## 3. Modeli bir kez indir

```bash
python download_model.py --output models/xlsr300m
# Tam bir Hugging Face commit'ini sabitlemek için isteğe bağlı: --revision COMMIT_SHA
```

Bu komut model ve processor'ı indirir, model ağırlıklarını yerel safetensors biçiminde saklar. Boş bir çıktı klasörü ister. Model dosyaları Git'e girmez. Mevcut bir XLS-R300M klasörünü de doğrudan kullanabilirsiniz.

## 4. Feature çıkarımı

```bash
python extract_features.py \
  --csv /absolute/path/metadata.csv \
  --audio-root /absolute/path/audio \
  --output outputs/features.csv \
  --model models/xlsr300m --local-files-only \
  --device cuda:0
```

CPU için `--device cpu` kullanın. Yerel model yerine `--model facebook/wav2vec2-xls-r-300m` verilebilir; bu durumda ilk kullanım internet/model indirmesi gerektirir ve `--local-files-only` kaldırılmalıdır.

Çıkarım: frozen `eval()`, FP32, son `last_hidden_state`, tek segment, padding/truncation yok. Çok kanallı ses kanal ortalamasıyla mono olur. 16 kHz dışındaki sesler `scipy.signal.resample_poly` ile bellekte 16 kHz'e çevrilir; kaynak sesler değişmez. Segment süresi korunur; yeni 5/10/20 saniye bölme yapılmaz.

Üretilen CSV, orijinal sütunlara `xlsr_meanvar_features` ekler. Sütun değeri 2048 sayılık JSON dizisidir:

| İndeks | Anlam |
|---|---|
| 0–1023 | Zaman boyunca mean |
| 1024–2047 | Zaman boyunca variance; `ddof=0`, `unbiased=False` |

Varyans standart sapma değildir. PCA veya feature scaling uygulanmaz. İlk segmentin mean/variance hesabı NumPy float64 ile ayrıca karşılaştırılır. Tüm satırlarda boyut/finite/nonnegative-variance, ses hash'i ve metadata korunumu kontrol edilir.

- `outputs/features.csv`: doğrulanmış özellikli metadata.
- `outputs/features.csv.report.json`: model/processor/loaded-weight/CSV/kod/PCM hash'leri ve paket sürümleri.
- `outputs/features.csv.parts/`: kesilen çıkarımın aynı girdi/model/kodla devam edebilmesi için satır cache'i. Değişmiş audio/cache kimliği kabul edilmez.

Tamamlanmış çıktı CSV'si üzerine yazılmaz. Yeni bir sonuç için yeni `--output` kullanın. 4 GPU satır paylaşımı için `--rank`, `--world-size` ve `--merge-only` seçeneklerini Slurm örneği kullanır; tüm rank'ler tamamlanmadan merge çalışmaz.

## 5. Depresyon ve anksiyete eğitimi

```bash
python train_lopo.py --csv outputs/features.csv --output outputs/runs \
  --target depresyon_skoru --dataset geriatric --selection 5 --device cuda:0

python train_lopo.py --csv outputs/features.csv --output outputs/runs \
  --target anksiyete_skoru --dataset geriatric --selection 5 --device cuda:0
```

`--selection` seçenekleri: `all`, `5`, `10`, `15`. Diğer grup için `--dataset non_geriatric` ve o gruba ait CSV'yi kullanın. Varsayılanlar: **20 Optuna trial**, **5 fold worker**, estimator seed **42**. Optuna sampler seed'i referans kodda olduğu gibi sabit değildir. CPU çalışması için `--device cpu` kullanılabilir; CPU ve GPU sayısal sonuçlarının birebir aynı olması beklenmez.

Tek grupta dört seçeneği çalıştırmak için:

```bash
for selection in all 5 10 15; do
  python train_lopo.py --csv outputs/features.csv --output outputs/runs \
    --target depresyon_skoru --dataset geriatric --selection "$selection" --device cuda:0
done
```

### Korunan LOPO/Optuna protokolü

1. Her fold bir hastanın **tüm segmentlerini** eğitim dışında tutar.
2. Eğitim segment düzeyindedir. Test hastasının segment tahminlerinin **medyanı**, hasta tahmini olur.
3. Her Optuna trial önce tüm 2048 özellik ile LOPO çalıştırır. Top-K için bu fold modellerinin importance ortalamasıyla **global** feature sıralaması yapılır ve seçilen K özellik ile ikinci LOPO çalışır. `All` ikinci seçim/LOPO yapmaz.
4. Optuna hedefi hasta başına mutlak hataların ortalaması (**hasta MAE**)dır. Arama alanı: `learning_rate=1e-3..0.1` log, `max_depth=4..14`, `n_estimators=100..500`.
5. En iyi parametrelerle final full LOPO tekrar çalışır. Final Top-K sıralamasında importance değerleri referans gibi altı ondalığa yuvarlanır. Final hasta tahminlerinden RMSE, MAE ve iki yönlü Pearson r/p hesaplanır; her hasta eşit ağırlıklıdır.

Bu, mevcut **non-nested** referans protokolüdür: global feature seçimi ve hiperparametre seçimi aynı hasta popülasyonunu kullanır. Bağımsız bir dış-test/nested LOPO değerlendirmesi olarak sunulmaz. Paylaşılabilir sürümde fold, sıralama, objective ve metrik fonksiyonları korunmuştur; yollar/CLI, hedef seçimi ve açık CPU desteği taşınabilir hale getirilmiştir. Bu değişiklikler MN5'te gönderilmiş işlerin kaynaklarını değiştirmez.

### Eğitim çıktıları ve model

Örnek klasör:

```text
outputs/runs/depresyon_skoru/geriatric_xlsr300m_meanvar_top5/
  identity.json
  best_params.json
  selected_features.json
  final_full_importance.csv
  patient_predictions.csv
  metrics.json
  optuna_trials.csv
  study.sqlite
  minimum_error_fold_model.joblib
  best_fold_model.json
```

Her konfigürasyonda **yalnız test hastası AE'si en düşük final LOPO fold modeli** saklanır. Eşitlikte deterministik hasta sırasındaki ilk fold seçilir. Aynı parametre/feature'larla o hastanın tüm segmentleri dışarıda tutularak model refit edilir; hasta medyan tahmini kaydedilmiş LOPO tahminiyle `rtol=atol=1e-7` eşleşmeden checkpoint kabul edilmez. `best_fold_model.json` model checksum'ını kaydeder. `final_full.joblib`, final importance satırlarının devam cache'idir; ikinci bir eğitilmiş XGBoost modeli değildir.

Kaydedilen minimum-AE fold modeli tüm veri üzerinde eğitilmiş nihai dağıtım modeli değildir; hangi hastanın dışarıda tutulduğu model payload'ında bulunur. MAE ve RMSE için ayrı model kaydı yoktur. Tek hasta/fold için bu iki fold hatası aynı mutlak hataya eşittir.

Aynı girdi/kod/ayarlarla tekrar çağırmak tamamlanmış konfigürasyonu atlar, yarım kalan Optuna trial'larını sürdürür. Kimlik uyuşmazlığında devam etmez; yeni çıktı klasörü kullanın. Tamamlanmış bir sonucu yalnız kayıtlı parametre/feature indeksleriyle yeniden doğrulamak için aynı komuta `--rerun` ekleyin; yeni Optuna araması yapmaz.

## 6. Tek node / 4 GPU Slurm örneği

`run_four_gpu.slurm` her hedef için iki grup × dört seçim = **8 konfigürasyon** çalıştırır. Her GPU aynı anda bir konfigürasyon çalıştırır; her worker iki konfigürasyonu ardışık alır. Varsayılan beş fold thread korunur.

Girdileri iki ayrı klasörde hazırlayın. `file_name` değerleri tam yol olsun veya CSV'nin yanındaki seslere göre göreli olsun:

```text
/path/depression_inputs/geriatric.csv
/path/depression_inputs/non_geriatric.csv
/path/anxiety_inputs/geriatric.csv
/path/anxiety_inputs/non_geriatric.csv
```

```bash
mkdir -p logs
export MODEL_DIR=/absolute/path/to/models/xlsr300m
export INPUT_DIR=/path/depression_inputs
export TARGET=depresyon_skoru
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION run_four_gpu.slurm

export INPUT_DIR=/path/anxiety_inputs
export TARGET=anksiyete_skoru
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION run_four_gpu.slurm
```

Böylece depresyon ve anksiyete ayrı ayrı **1 node/4 GPU** ister. Script'teki CPU/walltime ve account/partition/QoS ayarlarını kendi kümenize göre düzenleyin. Üç gün bir scheduler üst sınırıdır; çalışma süresi tahmini değildir. Ortam aktivasyonunu job içinde yapın veya `FEATURE_PYTHON`/`TRAIN_PYTHON` ile mevcut Python yollarını verin. Ayrı site-packages kullanılıyorsa `FEATURE_PYTHONPATH`/`TRAIN_PYTHONPATH` verilebilir.

Sıra: iki CSV'de dört GPU ile satır paylaşarak çıkarım → merge/doğrulama → dört GPU'da eğitim → metrik/checkpoint checksum doğrulaması ve `experiment_summary.csv`. Feature/merge hata verirse eğitim başlamaz. Modeller önceden indirilmiş olmalıdır; job offline çalışır. Varsayılan çalışma kökleri `outputs/slurm_depresyon_skoru` ve `outputs/slurm_anksiyete_skoru` olur. Her gönderimde yeni `RUN_DIR` seçin; feature dosyaları üzerine yazılmaz.

## 7. Gerçek veri olmadan deneme

```bash
python examples/make_synthetic_audio.py
python extract_features.py --csv examples/metadata_example.csv \
  --audio-root examples/synthetic --output outputs/demo_features.csv \
  --model models/xlsr300m --local-files-only --device cpu
python train_lopo.py --csv outputs/demo_features.csv --output outputs/demo \
  --target depresyon_skoru --dataset synthetic --selection 5 \
  --device cpu --trials 1 --workers 2
```

Bu yalnız kullanım/smoke denemesidir; klinik sonuç üretmez. Gerçek protokolde varsayılan 20 trial/5 worker kullanılır. Otomatik yerel kontroller:

```bash
python -m unittest discover -s tests -v
```

## Kaynaklar

- [Resmî XLS-R300M model kartı](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
- [XGBoost GPU kullanımı](https://xgboost.readthedocs.io/en/release_3.1.0/gpu/index.html)
- Proje içindeki mevcut mean+variance XGBoost protokolünden türetilmiştir. Paylaşılan örnek metadata, ses ve test çıktıları sentetiktir. Gerçek klinik veri, ağırlık ve çalışma çıktıları bu klasörün Git kapsamına dahil değildir.
