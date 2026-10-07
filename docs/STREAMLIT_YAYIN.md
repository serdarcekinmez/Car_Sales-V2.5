# Demoyu Streamlit Community Cloud üzerinden paylaşma

Bu belge, demonun GitHub deposundan Streamlit Community Cloud'a nasıl yayınlanacağını ve Türkiye'deki bir kullanıcıyla nasıl paylaşılacağını anlatır.

> **Durum (5 Ekim 2026):** `streamlit_app.py` yazıldı, ancak bu ortamda Streamlit kurulamadığı için **hiç çalıştırılmadı**. İlk yayından önce bilgisayarınızda `streamlit run streamlit_app.py` ile bir kez deneyin (§5).

## 1. Kısa cevap

Evet, mümkün. Yalnız üç noktaya dikkat etmek gerekiyor:

1. **Ana demo (FastAPI) Streamlit'te doğrudan çalışmaz.** Streamlit Cloud yalnızca `streamlit run` ile açılan uygulamaları çalıştırır. Bu yüzden depoya ayrı bir Streamlit ekranı eklendi: `streamlit_app.py`. Tahmini aynı Faz 7 modelleri ve aynı Faz 8 sunucu kodu üretir (doğrulama, `DemoRuntime`, CSV). Değişen yalnızca ekrandır.
2. **Veri ve model dosyaları git'e girmez.** `Datasets/` ve `workflow/runs/` iş verisidir ve `.gitignore` bunları dışarıda tutar. Streamlit Cloud ise yalnızca depodakileri görür. Bu dosyalar tek bir zip olarak **gizli** bir depolama alanına konur. Uygulama ilk açılışta zip'i indirir (§3).
3. **Erişim iki katmanla korunur.** Birinci katman Streamlit'in "private app" ayarıdır: uygulamayı yalnızca davet ettiğiniz e-posta adresleri açabilir. İkinci katman uygulamanın kendi girişidir: kullanıcı adı ve parola (scrypt özeti, FastAPI demosundakiyle aynı).

Türkiye'den `*.streamlit.app` adreslerine erişimi engelleyen bilinen bir kısıtlama yok. Yine de bağlantıyı paylaşmadan önce kullanıcıdan bir kez açmasını isteyin.

## 2. Depoya eklenen dosyalar

| Dosya | Görevi |
|---|---|
| `streamlit_app.py` | Streamlit ekranı. Alanlar yan yana sütunlarda. Birden fazla firma tabloda işaretlenip birlikte listeden çıkarılabilir. Sağ üstte "Oturumu kapat" düğmesi var. |
| `requirements.txt` | Streamlit Cloud'un kuracağı paketler. `requirements-demo.txt` ile aynı sabit sürümler kullanılır, üstüne `streamlit` eklenir. |
| `packages.txt` | Sistem paketi `libgomp1` (LightGBM bunu ister). |
| `.streamlit/config.toml` | Tema ve sunucu ayarları. |
| `.streamlit/secrets.toml.example` | Gizli ayarların örneği. Gerçek değerler git'e girmez. |
| `pipeline/pack_demo_data.py` | Uygulamanın okuduğu veri ve model dosyalarını tek bir zip'te toplar. |

## 3. Veri zip'ini hazırlama ve gizli yere koyma

Bilgisayarınızda, verilerin bulunduğu proje klasöründe şunu çalıştırın:

```bash
python pipeline/pack_demo_data.py --out garaj_demo_data.zip
```

Betik, uygulamanın kullandığı on dosyayı bulur ve depodaki göreli yollarıyla zip'e koyar: Faz 1–3 parquet dosyaları, lookup, katılım raporu, üye dizini, kur dosyası ve iki model. Zip `.gitignore` içinde olduğu için yanlışlıkla commit edilmez.

Zip'i **gizli** bir yere yükleyin. Önerilen yol, gizli bir Hugging Face dataset deposudur:

1. huggingface.co'da **Private** bir dataset deposu açın ve `garaj_demo_data.zip` dosyasını yükleyin.
2. Settings → Access Tokens bölümünden **Read** yetkili bir token oluşturun.
3. İndirme adresi şu biçimdedir: `https://huggingface.co/datasets/<kullanıcı>/<depo>/resolve/main/garaj_demo_data.zip`

`Authorization: Bearer <token>` başlığıyla indirmeye izin veren başka bir HTTPS adresi de kullanılabilir.

> Veriyi doğrudan GitHub'a koymayın. Depo herkese açıksa veri herkese açılır. Depo gizli olsa bile ham satış, teklif ve üye verisi git geçmişinde kalıcı olarak saklanır.

## 4. Streamlit Community Cloud'da yayınlama

1. https://share.streamlit.io adresine GitHub hesabınızla girin. Gizli bir depodan yayınlamak için Streamlit'e bu depoyu okuma izni verin.
2. **Create app** → **Deploy a public app from GitHub** seçin ve şu bilgileri girin:
   - Repository: `serdarcekinmez/Car_Sales-V2.5`
   - Branch: yayınlanacak dal (ör. `main`)
   - Main file path: `streamlit_app.py`
3. **Advanced settings** bölümünde:
   - **Python version:** listede varsa 3.10 seçin. Yoksa 3.11 veya 3.12 seçin. Paket sürümleri aynı kalır; modeller yalnızca LightGBM 4.6.0 ister.
   - **Secrets:** `.streamlit/secrets.toml.example` içeriğini yapıştırın ve doldurun:
     ```toml
     GARAJ_DEMO_USERNAME = "pilot"
     GARAJ_DEMO_PASSWORD_HASH = "scrypt:16384:8:1:..."   # run_demo.py hash-password çıktısı
     GARAJ_DEMO_DATA_URL = "https://huggingface.co/datasets/<kullanıcı>/<depo>/resolve/main/garaj_demo_data.zip"
     GARAJ_DEMO_DATA_TOKEN = "hf_..."
     ```
4. **Deploy** düğmesine basın. İlk açılışta zip indirilir ve modeller yüklenir. Bu birkaç dakika sürebilir.
5. Uygulama açıldıktan sonra **Settings → Sharing** bölümünde uygulamayı yalnızca davetli kişilere açık yapın. Türkiye'deki kullanıcının e-posta adresini (Google veya GitHub hesabı) ekleyin.
6. Kullanıcıya `https://<uygulama-adı>.streamlit.app` bağlantısını gönderin. Kullanıcı adı ve parolayı ayrı bir kanaldan iletin.

## 5. Yayından önce yerel deneme

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# secrets.toml: kullanıcı adı ve parola özetini girin. Veri bu klasörde olduğu için
# GARAJ_DEMO_DATA_URL boş kalabilir; uygulama veriyi depo klasöründen okur.
streamlit run streamlit_app.py
```

## 6. Sınırlar

- **Bellek:** Streamlit Community Cloud ücretsiz uygulamalara sınırlı bellek verir (yaklaşık 2–3 GB). Modeller ve geçmiş veri bu sınırı aşarsa uygulama açılmaz. O durumda FastAPI demosu (Docker + HTTPS, `docs/DEMO_KURULUM.md` §3) küçük bir sanal sunucuda yayınlanır.
- **Uyku:** Uzun süre kullanılmayan uygulama uykuya geçer. Kullanıcı açtığında "uyandır" düğmesine basar ve ilk yükleme yeniden birkaç dakika sürer.
- **Kayıtlar kalıcı değildir:** Tahmin kayıtları geçici bir SQLite dosyasında tutulur ve uygulama yeniden başlayınca silinir. Kalıcı geri bildirim toplamak için FastAPI demosu kullanılır.
- **"Oturumu kapat"** Streamlit'te yalnızca kullanıcının oturumunu kapatır. Bulut sunucusu açık kalır. Uygulamayı tamamen durdurmak için Streamlit panelinden **Reboot / Delete** kullanılır. Yerel kurulumda sunucuyu tamamen kapatan düğme FastAPI demosundadır (`DEMO_KURULUM.md` §2).
- **Hatalı giriş sınırı** tarayıcı oturumu başınadır. Asıl erişim koruması Streamlit'in davetli kullanıcı ayarıdır; bu ayarı açık tutun.
- Veri ve kur yenilemesinden sonra zip yeniden hazırlanır, gizli depoya yüklenir ve uygulama Streamlit panelinden **Reboot** edilir.
