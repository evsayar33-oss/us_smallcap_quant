# US SMALL-CAP QUANT — KUR-UNUT V1 MANUEL KURULUM

Bu paket, konuşmada yüklenen `us_smallcap_quant-main.zip` sürümü temel alınarak hazırlanmıştır.

## ÖNEMLİ

Canlı sistemin öğrenilmiş verilerini ezmemek için aşağıdaki dosyalar kurulum payload'ına dahil edilmedi:

- `us_ai_state.json`
- `gecmis_veri.csv`
- `signals_lifecycle.csv`
- `backtest_report.md`

Bu dört dosyanın orijinal yüklenen ZIP içindeki hali `02_REFERENCE_ONLY__DO_NOT_OVERWRITE_LIVE_DATA/` klasöründedir. Bunları canlı repo üzerine kopyalamayın.

## KURULUM

GitHub repo klasörünüzün içine `01_INSTALL_PAYLOAD/` altındaki dosyaları aynı yollarını koruyarak kopyalayın.

Değiştirilen mevcut dosyalar:

1. `main.py`
2. `longterm_auditor.py`
3. `backtest_optimizer.py`
4. `app.py`

Eklenen yeni dosya:

5. `autonomy_guard.py`

Ayrıca mevcut workflow dosyaları payload içinde güncel halleriyle bulunmaktadır:

- `.github/workflows/daily_scan.yml`
- `.github/workflows/backtest_optimization.yml`

`bootstrap_us_history.py`, `state_manager.py`, `.gitignore` ve `requirements.txt` da yüklenen proje sürümündeki halleriyle payload'a eklenmiştir; değişiklik yapılmamıştır.

## KUR-UNUT V1 NE YAPIYOR?

Akış:

DATA → REGIME → EXISTING QUANT SCORE → AUTONOMY GUARD → ENTRY CONTROL

Guard şu durum makinesini kullanır:

NORMAL → WATCH → SAFE → RECOVERY → NORMAL

### NORMAL

Mevcut Quant skorlaması ve normal giriş mantığı korunur.

### WATCH

Feature drift, performans drift'i veya piyasa stresi izleniyorsa yeni giriş standardı sıkılaşır:

- +10 puan ilave eşik
- 0.60x maruziyet katsayısı

### SAFE

Ciddi drift/veri/piyasa stresi oluşursa yeni girişler engellenir:

- yeni lifecycle sinyali oluşturulmaz
- mevcut skorlar silinmez
- mevcut öğrenme geçmişi silinmez

### RECOVERY

SAFE sonrası doğrudan tam kapasiteye dönülmez:

- 0.40x maruziyet
- +7 puan ilave eşik
- temiz gözlemler devam ederse NORMAL'e dönüş

## DRIFT

Feature dağılım drift'i PSI ile izlenir. Referans olarak yakın dönem gerçek geçmiş veri kullanılır.

İzlenen temel aileler:

- RVOL
- haftalık / aylık / yıllık performans
- mevcut Quant skor faktörleri
- Quant skor

Performans drift'i de ayrı izlenir.

## REGIME STRESS-TEST

`autonomy_guard.py` içindeki stress-test yalnızca motorun davranışını test eder. Sentetik senaryolar gerçek piyasa verisi yerine geçmez ve production backtest'e yazılmaz.

Test senaryoları:

- NORMAL
- EXPANSION
- ROTATION
- PANIC
- QUIET
- RECOVERY
- STRESS

## BACKTEST

Önceki sentetik fallback kaldırıldı.

Gerçek Yahoo/yfinance geçmiş verisi alınamazsa:

`BACKTEST ABORTED`

çalışır.

Bu durumda sistem sentetik fiyat üretip başarı raporu yazmaz.

## SİTE / STREAMLIT

Mevcut site görünümü korunmuştur.

Tek UI değişikliği:

`📊 Backtest Win Rate`

metriğinin üst kartlara eklenmesidir.

Değer mevcut `us_ai_state.json` içindeki:

`backtest_benchmark.win_rate`

alanından okunur.

Veri yoksa `Veri yok` gösterilir.

Başka sekme, tablo, başlık veya görünüm değişikliği yapılmamıştır.

## KURULUMDAN SONRA TEST

Repo kökünde:

```bash
python -m py_compile main.py longterm_auditor.py backtest_optimizer.py app.py autonomy_guard.py
```

Sonra:

```bash
python autonomy_guard.py
```

Beklenen:

```text
passed = True
passed_cases = 7
total_cases = 7
```

Canlı scan'i ayrıca çalıştırmak için:

```bash
python main.py
```

## NOT

Bu katman kârlılık garantisi değildir. Amacı sistemin piyasa/rejim/veri/model varsayımları bozulduğunda bunu algılayıp risk azaltması, yeni girişleri güvenli biçimde durdurması ve doğrulanmış koşullarda kontrollü şekilde geri açılmasıdır.
