# ABD Small-Cap Reel Getiri Motoru (BIST V3.10'un ABD uyarlaması)

BIST projesinde V3.10'a kadar gerçek veriyle doğrulanan her şey burada aynen çalışır. Değişen yalnızca hisse evreni ve veri kaynaklarıdır.

## Aynen taşınanlar (BIST V3.10)
* **%100 hisse portföyü:** Altın, nakit ya da strateji değiştirme yok.
* **Aylık dilimler:**
  * Her ayın ilk seansında en güçlü **5 likit hisse** alınır (aynı sektörden en fazla 2).
  * Her dilim **6 ay** tutulur. Portföyde ortalama 10–13 hisse olur, tek hisse en fazla %15.
  * Satışların parası hemen kalan hisselere dağıtılır.
* **Değer tuzağı koruması:** Ucuz ama hâlâ düşmekte olan hisseler (son 3 ay getirisi + 52 hafta dibinden uzaklık, en düşük üçte bir) alınmaz.
* **Kalibre güven oranı:** Her öneri için hissenin 12 ayda tipik bir ABD küçük hissesinden çok kazanma olasılığı ve yeni paranın önerilen payı verilir. Güven modeli canlı sonuçlardan kendini günceller.
* **Faktör ağırlıkları:** İleriye dönük (walk-forward) öğrenilir; 13 aylık boşluk, Newey-West t-istatistiği. Model her ayın 2'sinde yeniden eğitilir.
* **Sistem sağlığı:** 11 kontrol, canlı sonuç ile test karşılaştırması, kazanma oranları. Panelde **Sağlık** sekmesi, Telegram'da günlük satır ve haftalık tam rapor.
* **Stop yok:** Her hisse en geç 6 ayda yeniden değerlendirilir; 6. ay dolmadan bir ay önce uyarı gelir.

## ABD'ye uyarlananlar
| | BIST | ABD |
|---|---|---|
| Evren | Tüm BIST hisseleri | NYSE/NASDAQ/AMEX, piyasa değeri **250 milyon – 6 milyar $**, fiyat ≥ 3 $ |
| Likidite eşiği | Günlük medyan 20 milyon TL | Günlük medyan **5 milyon $** (backtest'te ABD TÜFE'ye göre geriye indirgenir) |
| Fiyat verisi | Yahoo (.IS) | Yahoo (küçük paketler, bekleme ve tekrar denemeli) |
| Bilanço verisi | İş Yatırım | **SEC EDGAR XBRL** (ücretsiz, anahtarsız). Her rakam SEC'e ilk dosyalandığı günden itibaren kullanılır; sonradan düzeltilen rakamlar geçmişe sızmaz. 4. çeyrek = yıllık − 9 aylık. |
| Enflasyon | TÜFE (EVDS) | **ABD TÜFE** (FRED CPIAUCSL, anahtarsız) |
| Mevduat | TCMB fonlama − vergi | **3 aylık Hazine bonosu** (FRED DTB3) |
| Endeks | BIST100 | **Russell 2000 (IWM)** |
| Hedef | TÜFE + dolar + altın + mevduat + %3 | ABD TÜFE + altın + hazine bonosu + %3 (yalnız gösterge) |
| Takvim / saat | BIST tatilleri, 18:25 TR | NYSE tatilleri, 21:30 UTC (New York kapanışından sonra) |

## Kurulum (Android, GitHub web)
1. Zip'i açın. Kök klasördeki tüm `.py` dosyalarını, `requirements.txt` ve `README_TR.md` dosyalarını repoya yükleyin; aynı isimli eski dosyaların üzerine yazılır.
2. `.github/workflows` klasörüne 3 `.yml` dosyasını yükleyin.
3. **Secrets:** `TELEGRAM_TOKEN` ve `CHAT_ID` zaten var. İsteğe bağlı olarak `SEC_USER_AGENT` ekleyebilirsiniz; SEC tanıtıcı bir kimlik ister, örneğin `adiniz iletisim@alanadi.com`. Eklemezseniz genel bir kimlik kullanılır.
4. **Actions → "US Small-Cap Engine - Walk Forward Backtest" → Run workflow.** İlk çalışma 1–3 saat sürebilir, çünkü ~1.500 hissenin fiyatı ve SEC bilançosu ilk kez indirilir. Sonraki çalışmalar önbellekten daha hızlıdır.
5. Ardından **"US Small-Cap Engine - Daily Run"** çalıştırın.

**Artık kullanılmayan eski dosyalar (silebilirsiniz):** `gecmis_veri.csv`, `us_ai_state.json`, `signals_lifecycle.csv`, `win_rate_optimizer.py`, `bootstrap_us_history.py`, `MANIFEST.json`, `VERIFY_RESULTS.txt`, `backtest_report.md`.

## Dürüstlük notları
* **Hayatta kalanlar yanılgısı:** Backtest bugün listede olan hisselerle yapılır; batmış ya da borsadan çıkmış şirketler eksiktir. Araştırmalara göre bu, ABD küçük hisselerinde sonucu yılda ~1,5–3 puan şişirir.
* **ABD küçük hisseleri BIST'ten farklı:** Russell 2000'in yaklaşık %40'ı zarar ediyor. Bu yüzden kârlılık faktörleri (ROE, kâr getirisi) burada daha önemli olabilir. Faktör ağırlıkları veriden öğrenildiği için sistem bunu kendisi ayarlar. Değer tuzağı kuralı BIST'te doğrulandı; ABD'de ilk backtest ile ölçülecek.
* **Düşüş beklentisi:** Russell 2000, 2008'de %59, 2020'de %41 düştü. 10–13 hisselik yoğun bir portföy benzer krizlerde daha da derin düşebilir.
* Yatırım tavsiyesi değildir.
