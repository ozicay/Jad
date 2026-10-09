# Jad proje talimatları

## Deney günlüğü

Kullanıcı, bundan sonraki her deneyin günlükte kaydedilmesini istedi.
Yeni bir deney istendiğinde `docs/agent-journal/YYYY-MM-DD.md` dosyasına
Europe/Istanbul tarihini kullanarak kayıt ekle. Yöntem/config kararları,
doğrulama, kaynak commit'i, deployment, Slurm gönderimi ve doğrulanmış
sonuçlar gibi anlamlı aşamalarda aynı kaydı ek bilgilerle sürdür.

Kayıtta hedefi, veri/config yollarını, feature seçimini, LOPO ve metrik
protokolünü, kaynak commit'ini, çıktı/log yollarını, job ID'lerini, kanıtlanan
durumu ve kalan işi belirt. Gerçekten kontrol edilmemiş sonuçları tamamlandı
olarak yazma. Önceki kayıtları silmeden düzeltme veya yeni durum kaydı ekle.
Hasta bazlı hassas verileri, ses içeriklerini, kimlik bilgilerini ve büyük
çıktıları günlüğe/Git'e ekleme. Günlük, deney çalıştırma yetkisi sağlamaz.
