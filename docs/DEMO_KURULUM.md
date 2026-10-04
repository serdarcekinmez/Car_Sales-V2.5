# Müşteri demosu (Faz 8) — kurulum ve çalıştırma

Bu belge, Faz 8 müşteri web demosunun nasıl kurulacağını ve başlatılacağını anlatır.
Demo tek bir pilot müşteri hesabıyla çalışır. Tahminleri sunucudaki Faz 7 katmanı ve dondurulmuş modeller üretir. Tarayıcı yalnızca sonucu gösterir.

> **Durum (4 Ekim 2026):** Kod yazıldı ve yalnızca okunarak incelendi. Kurulum komutları, sunucu ve testler **hiç çalıştırılmadı**. Aşağıdaki komutlar ilk çalıştırma içindir. İlk denemede çıkan hatalar Faz 8 kabul kontrolünün parçasıdır.

## 1. Ne gerekir?

| Gereken | Açıklama |
|---|---|
| Python 3.10 | Proje 3.10 ile geliştirildi. Paket sürümleri `requirements-demo.txt` dosyasında sabit. |
| Veri ve model dosyaları | Depodaki mevcut yollar kullanılır. Ayrıntı: `src/phase7_inference/bundle.py` içindeki `BundleConfig`. |
| Bir kullanıcı adı ve parola | Parolanın kendisi hiçbir dosyaya yazılmaz. Yalnızca tuzlu scrypt özeti saklanır. |

Uygulamanın okuduğu dosyalar (proje köküne göre):

```
Datasets/interim/phase1_preparation/clean_sales.parquet
Datasets/interim/phase1_preparation/clean_reel_db.parquet
Datasets/interim/phase2_labels/labeled_candidates.parquet
Datasets/interim/phase3_firm_universe/firm_registry.parquet
Datasets/lookups/internal_firm_lookup.csv
Datasets/raw/2501-260604 AA Katılım Raporu.xlsx
Datasets/raw/Üye-Kurumsal Raporu_2026-06-07.xlsx
Datasets/external/fx/USDTRY_yahoo_daily_2024-11-01_2026-10-02_fetched20261003T074901.csv
workflow/runs/phase4_experiment_build/results/final_model.json
workflow/runs/phase5_experiment_build/results/models/lgbm_noR.json
```

Uygulama yalnızca şu dosyaya yazar: geri bildirim ve tahmin kayıtlarının tutulduğu SQLite dosyası. Varsayılan yeri `Datasets/phase8_demo/demo.sqlite3` dosyasıdır ve git dışında kalır.

## 2. Yerel çalıştırma (Linux veya Windows'ta WSL)

Windows'ta WSL önerilir. WSL içinde açılan `http://localhost:8000` adresi Windows tarayıcısından da açılır.

```bash
cd ~/projects/Garaj/Car_Sales-V2.5

# 1) Ayrı bir sanal ortam (önerilir; mevcut ortamda paketler zaten kuruluysa atlanabilir)
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-demo.txt

# 2) Parola özeti üret (parola ekrana yazılmaz; en az 12 karakter)
python pipeline/run_demo.py hash-password
#    Çıktı: GARAJ_DEMO_PASSWORD_HASH=scrypt:16384:8:1:...

# 3) Ayar dosyası
cp .env.example .env
#    .env içinde GARAJ_DEMO_USERNAME ve GARAJ_DEMO_PASSWORD_HASH satırlarını doldurun.

# 4) Ayarları kabuğa yükle ve başlat
set -a; source .env; set +a
python pipeline/run_demo.py serve --insecure-local-http
```

Tarayıcıda `http://127.0.0.1:8000` adresini açın ve `.env` içindeki kullanıcı adı ile parolayla giriş yapın.

- İlk açılışta modeller ve geçmiş veriler arka planda yüklenir. Bu sırada üst şeritte "yükleniyor" yazar. Yükleme bitince form kullanılabilir.
- `--insecure-local-http` yalnızca bu bilgisayardaki düz HTTP erişimi içindir. Başka bir adrese bağlanmayı başlatıcı reddeder.
- Başka bir veri kökü veya kayıt dosyası için `--root DİZİN` ve `--store DOSYA` kullanılır.
- Sunucu yeniden başlatılınca bütün oturumlar kapanır. Kayıtlı tahminler ve geri bildirimler SQLite dosyasında kalır.

Windows PowerShell'de doğrudan çalıştırma (WSL yoksa):

```powershell
cd C:\...\Car_Sales-V2.5
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-demo.txt
py pipeline\run_demo.py hash-password
$env:GARAJ_DEMO_USERNAME = "pilot"
$env:GARAJ_DEMO_PASSWORD_HASH = "scrypt:16384:8:1:..."   # hash-password çıktısındaki değer
py pipeline\run_demo.py serve --insecure-local-http
```

## 3. Sunucuda yayın (Docker + HTTPS ters vekil)

İmajda yalnızca kod bulunur. Veri, modeller ve gizli bilgiler imaja girmez. `.dockerignore` bir izin listesidir. `Datasets/`, `workflow/runs/`, `reports/`, `.git` ve `.env` derleme bağlamına gönderilmez.

```bash
cd ~/projects/Garaj/Car_Sales-V2.5

# 1) İmaj
docker build -f Dockerfile.demo -t garaj-demo:faz8 .

# 2) Parola özeti (imaj içinden de üretilebilir)
docker run --rm -it garaj-demo:faz8 python /app/pipeline/run_demo.py hash-password

# 3) .env: GARAJ_DEMO_USERNAME, GARAJ_DEMO_PASSWORD_HASH ve
#    GARAJ_DEMO_ALLOWED_HOSTS=demo.alanadiniz.com (tarayıcının kullanacağı ad) zorunlu.
#    Ters vekil aynı makinedeyse: GARAJ_DEMO_TRUSTED_PROXIES=172.17.0.1
#    (Docker köprü ağ geçidi; kendi kurulumunuzda `docker network inspect bridge` ile doğrulayın.)

# 4) Kayıt deposu ve başlatma
docker volume create garaj-demo-store
docker run -d --name garaj-demo --restart unless-stopped \
  --env-file .env \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/Datasets:/data/Datasets:ro" \
  -v "$PWD/workflow/runs/phase4_experiment_build/results:/data/workflow/runs/phase4_experiment_build/results:ro" \
  -v "$PWD/workflow/runs/phase5_experiment_build/results/models:/data/workflow/runs/phase5_experiment_build/results/models:ro" \
  -v garaj-demo-store:/var/lib/garaj-demo \
  garaj-demo:faz8

docker logs -f garaj-demo
```

Bağlama noktaları:

- Uygulama veri kökünü `/data` olarak görür. Dosyalar depodaki göreli yollarıyla aynı yere salt okunur bağlanır. Bu nedenle `BundleConfig` varsayılanları değiştirilmeden çalışır.
- Yazılabilir tek yer `/var/lib/garaj-demo` birimidir. Tahmin ve geri bildirim kayıtları buradaki SQLite dosyasında tutulur.
- Port yalnızca makinenin kendi döngü adresine (`127.0.0.1:8000`) açılır. Dışarıdan erişim yalnızca HTTPS ters vekil üzerinden olur.

Caddy ile HTTPS örneği (sertifikayı Caddy kendisi alır):

```
demo.alanadiniz.com {
    reverse_proxy 127.0.0.1:8000
}
```

Nginx kullanılırsa `Host` başlığı korunmalıdır:

```
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

**Henüz bir sunucu veya alan adı seçilmedi; demo hiçbir yerde yayında değildir.** Cloudflare Workers gibi sunucusuz ortamlar bu Python/LightGBM sunucusunu doğrudan çalıştıramaz. Docker çalıştırabilen bir sanal sunucu gerekir.

## 4. Güvenlik kuralları

- **Tek hesap, tek müşteri.** Hesap ortam değişkeninden gelir; kodda varsayılan parola yoktur. Başka bir müşteri için ayrı kurulum ve ayrı hesap gerekir. Aynı kurulumdaki tahminler bu hesaba aittir.
- **Oturum:** Rastgele oturum anahtarı HttpOnly, SameSite=Strict ve Secure çerezde tutulur. Oturum süresi varsayılan olarak 8 saattir. Çıkış oturumu sunucuda geçersiz kılar.
- **CSRF:** Değişiklik yapan her istek oturuma özel bir güvenlik belirteci, JSON gövdesi ve izinli kaynak (Origin) ister. CORS açılmamıştır.
- **Giriş denemeleri:** Aynı istemciden 15 dakikada 5 hatalı denemeden sonra giriş geçici olarak reddedilir. `X-Forwarded-For` başlığı yalnızca `GARAJ_DEMO_TRUSTED_PROXIES` içindeki vekilden gelirse dikkate alınır.
- **Erişim sınırı:** Tahmin, CSV ve geri bildirim yalnızca sahibi olan hesapla açılır. Başka hesabın kaydı "bulunamadı" görünür.
- **Veri sızıntısı:** Model dosyaları, ham satış/teklif geçmişi ve üye dizini tarayıcıya hiç gönderilmez. Sunucu yalnızca üç sabit sayfa dosyası (`index.html`, `app.js`, `style.css`) sunar.
- **Başlıklar:** Satır içi kod içermeyen sıkı CSP uygulanır. Ayrıca çerçeve yasağı, `nosniff`, `no-store` ve HTTPS'te HSTS kullanılır. İstek gövdesi en fazla 64 KB olabilir.
- **Günlükler:** Parola, çerez ve ham firma geçmişi günlüğe yazılmaz.

## 5. Yenileme ve sınırlar

- **Kur:** Kur dosyası 2 Ekim 2026'da bitiyor. Fiyat için tahmin tarihinden önceki en fazla 7 günlük kur gerekir. Bu nedenle 9 Ekim 2026'dan sonraki tahmin tarihlerinde fiyat "gösterilemiyor" olur ve firma listesi yine gelir. Yeni kur kaydı çekilip `BundleConfig.fx_file` güncellenmeden bu düzelmez. Uygulama kuru kendisi çekmez.
- **Geçmiş veri:** Satış ve teklif geçmişi 4 Haziran 2026'da bitiyor. Bugünkü araçlar için ekranda "eski veri" uyarısı görünür. Yeni dışa aktarımlar geldiğinde önce Faz 1–3 çıktıları yenilenir, sonra sunucu yeniden başlatılır. Bu işlem modeli yeniden eğitmez.
- **Paket kimliği:** Her tahmin; veri, model, Faz 7 kodu ve çalışma ortamı özetini (`bundle_id`), filtre kurallarını ve Faz 8 sunucu kodunun özetini birlikte saklar. Böylece bir sonucun neden değiştiği izlenebilir.

## 6. İlk çalıştırmada yapılacak kabul kontrolü (henüz yapılmadı)

1. Sunucu açılıyor ve üst şeritte iki model "hazır" görünüyor.
2. Yanlış parola reddediliyor ve 5 denemeden sonra giriş geçici olarak kapanıyor.
3. Yeni bir araç için firma listesi ve TL fiyatı aynı ekranda geliyor.
4. Bir firma "Listeden çıkar" ile dışlanıyor ve liste aşağıdan dolduruluyor. Yeni tahmin no oluşuyor; eski tahmin değişmiyor.
5. 10 Ekim 2026 veya sonrası bir tahmin tarihinde fiyat "gösterilemiyor", firma listesi geliyor.
6. CSV dosyası Excel'de Türkçe karakterlerle açılıyor.
7. Firma ve fiyat geri bildirimi kaydediliyor ve tahmin yeniden açıldığında görünüyor.
8. Çıkıştan sonra API istekleri 401 dönüyor.
