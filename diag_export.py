"""ABD tek seferlik araştırma verisi: fiyat paneli + split geçmişi (düzeltilmemiş fiyat için)
+ SEC EDGAR point-in-time bilanço + endeks. Çıktı ayrı `diag-data` dalına gider; main ve
./data'ya dokunulmaz."""
import json, os, time
import pandas as pd
import backtest_optimizer as B
import benchmarks as BM
import fundamentals_hist as FH
import market_data as MD
import regime_model as RM

OUT = "diag_out"
os.makedirs(OUT, exist_ok=True)
uni = B.backtest_universe()
data = MD.download_history(uni, "2010-01-01", None, min_rows=300)
print(f"indirilen hisse: {len(data)} / {len(uni)}")
for fld in ("open", "high", "low", "close", "volume"):
    pd.DataFrame({t: g[fld] for t, g in data.items()}).sort_index() \
      .to_csv(f"{OUT}/px_{fld}.csv.gz", compression="gzip", float_format="%.6g")

# split geçmişi -> araştırmada düzeltilmemiş (o günkü gerçek) fiyat yeniden kurulur
import yfinance as yf
rows = []
for i, t in enumerate(sorted(data)):
    for k in range(3):
        try:
            s = yf.Ticker(MD.yf_symbol(t)).splits
            for d, r in s.items():
                rows.append({"ticker": t, "date": pd.Timestamp(d).tz_localize(None).date(), "ratio": float(r)})
            break
        except Exception:
            time.sleep(2 * (k + 1))
    if i % 200 == 0:
        print(f"split {i}/{len(data)}")
pd.DataFrame(rows).to_csv(f"{OUT}/splits.csv", index=False)

try:
    fh = FH.load_history(sorted(data), 2010)
    fh.to_csv(f"{OUT}/fund_hist.csv.gz", compression="gzip", index=False)
    print(f"bilanço satırı: {len(fh)}  hisse: {fh['ticker'].nunique() if 'ticker' in fh else '?'}")
except Exception as exc:
    print(f"::warning::bilanço alınamadı: {exc}")

RM.download_regime_series(start="2009-01-01").to_csv(f"{OUT}/regime.csv")
BM.download_benchmarks(start="2009-01-01").to_csv(f"{OUT}/bench.csv")
try:
    snap, _ = MD.fetch_snapshot({})
    snap.to_csv(f"{OUT}/snapshot_today.csv", index=False)
except Exception as exc:
    print(f"::warning::snapshot: {exc}")
json.dump({"universe": uni, "downloaded": sorted(data)}, open(f"{OUT}/universe.json", "w"))
print("✅ tamam")
