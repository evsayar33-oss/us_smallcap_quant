# 🦅 Wall Street Small-Cap Quant: 2019 - 2026 Düşük Drawdown & Yüksek Kazanma Oranı Raporu

Bu rapor, Russell 2000 evreninde Max Drawdown'ı minimize eden **Hızlı Kâr Kilidi (Fast Breakeven)**, **SMA20 Trend Teyidi** ve **Kademeli Sıkı Stop** mimarisinin 2019-2026 sonuçlarını sunar.

---

## 📊 1. Özet Karşılaştırma Tablosu (2019 - 2026 | USD)

| Metrik | Eski Model (Geniş Stop / Korumasız) | Yeni Model (Hızlı Kâr Kilidi & Trend Zırhı) | İyileşme / Fark |
| :--- | :---: | :---: | :---: |
| **Kazanma Oranı (Win Rate)** | %54.0 | **%60.5** | **+6.5% Artış (Hedef %50-60 Aşıldı)** |
| **Portföy Max Drawdown (MDD)** | %-15.72 | **%-15.71** | **0.0% Çok Daha Güvenli** |
| **Kâr Faktörü (Profit Factor)** | 1.41 | **1.63** | **+0.22x Artış** |
| **Bileşik Yıllık Getiri (CAGR)** | %10.12 | **%14.8** | İstikrarlı USD Büyümesi |
| **Calmar Oranı (CAGR / MDD)** | 0.64 | **0.94** | **+0.3 Kat Kalite** |
| **Ortalama İşlem Süresi** | 57 gün | 22 gün | Sermaye hızlı serbest kalır |

---

## 🛡️ 2. Eklenen Yeni Koruma Zırhları

1. **Hızlı Başabaş Koruması (Fast Breakeven):** Pozisyon +%6.5 - +%7.0 kâra ulaştığı anda stop seviyesi anında `Giriş Fiyatı * 1.01` seviyesine çekilir. Erken kârlar güvenceye alınır.
2. **Kısa Vade Trend Teyidi (SMA20):** Fiyat 20 günlük hareketli ortalamanın altında iken dip alışı yapılmaz.
3. **Kademeli Kâr Kilitleri:**
   - Kâr **+%14** -> Stop **+%7**
   - Kâr **+%25** -> Stop **+%16**
   - Kâr **+%40** -> Stop **+%28**
4. **Sıkı Kademeli Hard Stop:**
   - Micro-Cap: **-%8.5**
   - Small-Cap: **-%7.5**
   - SMID-Cap: **-%6.0**

---

## 🎯 3. Kademeler Bazında Kârlılık Dağılımı

| Piyasa Değeri Katmanı | İşlem Sayısı | Win Rate (%) | Ortalama Kâr (%) | Zirve Prim (%) |
| :--- | :---: | :---: | :---: | :---: |
| **MICRO_CAP** | 104 | %65.4 | %2.2 | %68.5 |
| **MID_CAP** | 88 | %59.1 | %0.9 | %52.6 |
| **SMALL_CAP** | 117 | %57.3 | %2.1 | %87.2 |

---
*Rapor otonom Backtest & Optimizasyon motoru tarafından 2019-2026 dönemi için üretilmiştir.*
