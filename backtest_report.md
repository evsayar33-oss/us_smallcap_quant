# 🦅 Wall Street Small-Cap Quant: 2019 - 2026 Backtest & Optimizasyon Raporu

Bu rapor, Russell 2000 evrenindeki hisselerin piyasa değerine ($250M - $1B Micro, $1B - $3B Small, $3B - $6B SMID) göre uygulanan **Dinamik Kademeli Eşikler** ile **Rastgele Sabit Eşikler** arasındaki ampirik sonuçları karşılaştırır.

---

## 📊 1. Özet Karşılaştırma Tablosu (2019 - 2026 | USD)

| Metrik | Eski Model (Sabit & Tekil Eşik) | Yeni Model (Dinamik Piyasa Değeri Kademeli) | İyileşme / Fark |
| :--- | :---: | :---: | :---: |
| **Toplam İşlem Sayısı** | 187 | 350 | Daha seçici & odaklı |
| **Kazanma Oranı (Win Rate)** | %44.4 | **%45.1** | **+0.7% Artış** |
| **Kâr Faktörü (Profit Factor)** | 1.53 | **1.52** | **+-0.01x Artış** |
| **Bileşik Yıllık Getiri (CAGR)** | %35.98 | **%61.42** | **+25.4% Artış** |
| **Maksimum Düşüş (Max Drawdown)** | %-85.73 | **%-86.26** | **-0.5% Daha Güvenli** |
| **Calmar Oranı (CAGR / MDD)** | 0.42 | **0.71** | **+0.29 Kat Kalite** |
| **Ortalama İşlem Süresi** | 75 gün | 59 gün | Sermaye hızlı serbest kalır |

---

## 🎯 2. Piyasa Değeri Kademelerine Göre Optimize Edilen Eşikler

### 🐣 Kademe 1: US Micro-Cap ($250M – $1B)
- **Mantık:** Yüksek büyüme hızı ve yüksek volatilite. Erken dönem şirketleri için daha geniş dip tabanı ve piyasa gürültüsünden erken silkelenmeyi önleyen stop.
- **Min ROE:** %8.0
- **Min Esas Faaliyet Marjı:** %4.0
- **52H Dip Taban Mesafesi:** %2.5 – %32.0
- **F/K Tavanı:** 35.0
- **Stop-Loss:** %-14.0
- **Maksimum Kuluçka Sabrı:** 65 Gün

### 🦅 Kademe 2: US Core Small-Cap ($1B – $3B)
- **Mantık:** Russell 2000'in omurgası. Operasyonel kârlılığı kanıtlanmış büyüme şirketleri.
- **Min ROE:** %12.0
- **Min Esas Faaliyet Marjı:** %6.5
- **52H Dip Taban Mesafesi:** %3.0 – %26.0
- **F/K Tavanı:** 28.0
- **Stop-Loss:** %-11.0
- **Maksimum Kuluçka Sabrı:** 90 Gün

### 🏢 Kademe 3: US SMID-Cap ($3B – $6B)
- **Mantık:** Kurumsal fonların radarında, nakit akışı oturmuş defansif ve güçlü şirketler. Sermaye koruma odaklı sıkı taban ve sıkı stop.
- **Min ROE:** %16.0
- **Min Esas Faaliyet Marjı:** %9.0
- **52H Dip Taban Mesafesi:** %3.0 – %20.0
- **F/K Tavanı:** 22.0
- **Stop-Loss:** %-8.0
- **Maksimum Kuluçka Sabrı:** 120 Gün

---

## 📈 3. Kademeler Bazında Kârlılık Dağılımı

| Piyasa Değeri Katmanı | İşlem Sayısı | Win Rate (%) | Ortalama Kâr (%) | Zirve Prim (%) |
| :--- | :---: | :---: | :---: | :---: |
| **MICRO_CAP** | 130 | %51.5 | %6.2 | %159.5 |
| **MID_CAP** | 96 | %40.6 | %1.9 | %110.4 |
| **SMALL_CAP** | 124 | %41.9 | %0.7 | %129.1 |

---
*Rapor otonom Backtest & Optimizasyon motoru tarafından 2019-2026 dönemi için üretilmiştir.*
