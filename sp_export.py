"""Tek seferlik S&P 500 araştırma verisi (us_smallcap_quant reposunda çalışır, ana dala ve ./data'ya DOKUNMAZ).
* Evren: bugünkü S&P 500 + Wikipedia'daki geçmiş eklenen/çıkarılan şirketler (üyelik tarihleriyle)
* Yahoo günlük OHLCV (2005-), split geçmişi, SPY / RSP / ^GSPC
* SEC EDGAR point-in-time bilanço (gelir, net kâr, faaliyet kârı, BRÜT kâr, faaliyet nakit akışı, yatırım harcaması,
  özsermaye, borç, toplam varlık, hisse adedi)
Çıktı -> `sp-data` dalı."""
import io, json, os, sys, tempfile, time
os.environ["BOQ_DATA_DIR"] = tempfile.mkdtemp()           # SEC önbelleği geçici klasöre: repo verisine dokunma
import numpy as np, pandas as pd, requests
import config as C, fundamentals_hist as FH, market_data as MD

OUT = "sp_out"; os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research export)"}
html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA, timeout=60).text
tabs = pd.read_html(io.StringIO(html))
cur, chg = tabs[0], tabs[1]
cur.to_csv(f"{OUT}/sp500_current.csv", index=False)
chg.columns = ["_".join([str(x) for x in c if "Unnamed" not in str(x)]).strip("_") if isinstance(c, tuple) else str(c) for c in chg.columns]
chg.to_csv(f"{OUT}/sp500_changes.csv", index=False)
print("güncel", len(cur), "değişiklik", len(chg), list(chg.columns))
sym = lambda s: str(s).strip().upper().replace(".", "-")
tick = set(cur.iloc[:, 0].map(sym))
for c in chg.columns:
    if "Ticker" in c or "ticker" in c:
        tick |= set(chg[c].dropna().map(sym))
tick = sorted(t for t in tick if t and t != "NAN")
print("toplam aday", len(tick))

data = MD.download_history(tick + ["SPY", "RSP", "^GSPC"], "2005-01-01", None, min_rows=120)
print("fiyatı bulunan", len(data))
for f in ("open", "high", "low", "close", "volume"):
    pd.DataFrame({t: g[f] for t, g in data.items()}).sort_index().to_csv(f"{OUT}/px_{f}.csv.gz", compression="gzip", float_format="%.6g")

import yfinance as yf
rows = []
for i, t in enumerate(sorted(data)):
    for k in range(3):
        try:
            for d, r in yf.Ticker(MD.yf_symbol(t)).splits.items():
                rows.append({"ticker": t, "date": pd.Timestamp(d).tz_localize(None).date(), "ratio": float(r)})
            break
        except Exception:
            time.sleep(2 * (k + 1))
pd.DataFrame(rows).to_csv(f"{OUT}/splits.csv", index=False)

# SEC: add gross profit, operating cash flow, capex (flows) and total assets (instant)
FH.FLOW_TAGS.update({
    "gross_profit": ["GrossProfit"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "dividends": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "buyback": ["PaymentsForRepurchaseOfCommonStock"],
})
_orig = FH.parse_companyfacts
def parse2(ticker, facts):
    d = _orig(ticker, facts)
    if d is None or d.empty:
        return d
    a = FH._instant(facts, ["Assets"])
    d["assets"] = [FH._asof(a, pd.Timestamp(e), pd.Timestamp(av))[0] for e, av in zip(d["period_end"], d["avail_date"])]
    return d
FH.parse_companyfacts = parse2
names = [t for t in sorted(data) if t not in ("SPY", "RSP", "^GSPC")]
fh = FH.load_history(names, 2005)
fh.to_csv(f"{OUT}/fund_hist.csv.gz", index=False, compression="gzip")
print("bilanço satırı", len(fh), "şirket", fh["ticker"].nunique() if len(fh) else 0, "SEC tanı", json.dumps(FH.LAST_DIAG)[:300])
json.dump({"candidates": tick, "downloaded": sorted(data)}, open(f"{OUT}/universe.json", "w"))
print("✅ tamam")
