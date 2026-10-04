# Car Sales V2.5

Bu klasör Car Sales V2'nin çalıştırılabilir sade sürümüdür. Yalnızca müşteri demosunu çalıştırmak ve veri girdilerini yenilemek için gereken kod burada bulunur.

Bir araç ve tahmin tarihi girildiğinde iki ayrı sonuç üretilir:

- **Kimi arayalım?** Model A, dondurulmuş LambdaRank modeliyle firma öncelik listesini üretir.
- **Tahmini ihale satış fiyatı:** Model B, dondurulmuş `lgbm_noR` modeliyle TL fiyat tahmini üretir. Bu model Min.Fiyat kullanmaz.

Modellerin eğitimi, değerlendirmesi, raporlar ve çalışma kayıtları burada değildir; V2 deposunda kalır: https://github.com/serdarcekinmez/Car_Sales-V2

## Klasörler

| Yol | İçerik |
|---|---|
| `src/phase8_demo/` | Web uygulaması: giriş, form, sonuçlar, filtre, CSV ve geri bildirim |
| `src/phase7_inference/` | Ortak tahmin katmanı: araç girdisi, iki modelin bağlantısı, filtreler, paket kimliği |
| `src/common`, `src/phase1_preparation` … `src/phase5_pricing` | Tahmin katmanının ve veri yenilemenin kullandığı modüller (yalnızca gerekenler) |
| `pipeline/run_demo.py` | Parola özeti üretir ve sunucuyu başlatır |
| `pipeline/run_phase1.py`, `run_phase2.py`, `run_phase3_firm_registry.py` | Yeni dışa aktarımlardan veri girdilerini yeniden üretir |
| `docs/DEMO_KURULUM.md` | Kurulum, çalıştırma ve güvenlik (Türkçe) |
| `Dockerfile.demo`, `.dockerignore`, `.env.example`, `requirements-demo.txt` | Paketleme ve ayar dosyaları |

## Git'e girmeyenler

Veri ve model dosyaları iş verisidir. `.gitignore` bunları dışarıda tutar; yerel klasörde aşağıdaki yollarda bulunmaları gerekir:

- `Datasets/raw/` — ham Excel dışa aktarımları (satış, teklif, katılım, üye dizini)
- `Datasets/lookups/`, `Datasets/interim/…` — Faz 1–3 çıktıları
- `Datasets/external/fx/` — USDTRY kur kaydı
- `workflow/runs/phase4_experiment_build/results/final_model.json` — Model A
- `workflow/runs/phase5_experiment_build/results/models/lgbm_noR.json` — Model B

Model dosyaları, kod değişmeden çalışsın diye V2'deki göreli yollarıyla aynı yerde tutuldu (`BundleConfig` varsayılanları).

## Hızlı başlangıç

```bash
pip install -r requirements-demo.txt
python pipeline/run_demo.py hash-password      # çıktıyı .env dosyasına yapıştırın
cp .env.example .env                           # GARAJ_DEMO_USERNAME ve hash satırını doldurun
set -a; source .env; set +a
python pipeline/run_demo.py serve --insecure-local-http
# http://127.0.0.1:8000
```

Ayrıntılar ve Docker + HTTPS kurulumu için: [docs/DEMO_KURULUM.md](docs/DEMO_KURULUM.md).

## Veriyi yenileme

Yeni dışa aktarımlar `Datasets/raw/` klasörüne konur ve yenileme şu sırayla yapılır: `run_phase1.py` → `run_phase2.py` → `run_phase3_firm_registry.py`. Kur kaydı ayrıca çekilir: `python src/phase5_pricing/fx_rates.py fetch --start 2024-11-01`. Son olarak sunucu yeniden başlatılır. Bu işlem modelleri yeniden eğitmez.

Not: Katılım raporu, üye dizini ve kur dosyasının adları şu anda `BundleConfig` içinde tam dosya adıyla tanımlı. Yeni adla gelen dosyalar için bu yapılandırmanın güncellenmesi gerekir.

## Durum (4 Ekim 2026)

Faz 7 ve Faz 8'in kodu yazıldı ve Codex tarafından kaynak üzerinden incelendi. Kod **henüz hiç çalıştırılmadı**. İlk çalıştırma, `docs/DEMO_KURULUM.md` §6'daki kabul kontrolüyle yapılacak.

Bilinen sınırlar:

- Satış ve teklif geçmişi 4 Haziran 2026'da bitiyor.
- Kur kaydı 2 Ekim 2026'da bitiyor. Bu nedenle 9 Ekim 2026'dan sonraki tarihler için fiyat gösterilmez; firma listesi yine gelir.
