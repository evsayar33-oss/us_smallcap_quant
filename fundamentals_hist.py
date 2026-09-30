"""Point-in-time US fundamentals from SEC EDGAR XBRL (free, keyless) — US port of the BIST module.

Source: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json (one request per company,
<= 10 req/s, a descriptive User-Agent is REQUIRED: env SEC_USER_AGENT).
Ticker -> CIK map: https://www.sec.gov/files/company_tickers.json

No look-ahead:
  * every value is used only from the date it was FIRST filed with the SEC (+1 day);
    later restatements of the same period are ignored (the original number is what the market saw),
  * discrete quarters come from 10-Q (3-month facts); Q4 = fiscal year (10-K) - 9-month YTD;
    missing quarters are derived from YTD differences (H1 - Q1, 9M - H1),
  * TTM = sum of 4 contiguous quarters, available when the last of them was filed.
Output columns match the BIST module so the engine is unchanged:
  ticker, year, period, net_income_ttm, revenue_ttm, op_profit_ttm, equity, fin_debt,
  paid_in (= shares outstanding), rev_growth_pct, period_end, avail_date
Coverage: XBRL is mandatory for small companies from mid-2011, so realistic history starts ~2012.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

import config as C

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
REQ_PAUSE = 0.13                                   # ~7.5 requests/second (SEC limit: 10)

FLOW_TAGS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "RevenuesNetOfInterestExpense",
                "InterestAndDividendIncomeOperating"],
    "net_income": ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"],
    "op_profit": ["OperatingIncomeLoss", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"],
}
INSTANT_TAGS = {
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "debt_lt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "debt_st": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
}
SHARE_TAGS = [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")]
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT", "20-F", "40-F"}


def _headers() -> Dict[str, str]:
    return {"User-Agent": C.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def sec_key(ticker: str) -> str:
    return str(ticker).upper().replace(".", "-")


def load_cik_map() -> Dict[str, int]:
    r = requests.get(TICKERS_URL, headers=_headers(), timeout=30)
    r.raise_for_status()
    return {str(v["ticker"]).upper(): int(v["cik_str"]) for v in r.json().values()}


# ------------------------------------------------------------------ parsing one companyfacts JSON
def _entries(facts: Dict, ns: str, tag: str, unit: str) -> pd.DataFrame:
    try:
        rows = facts["facts"][ns][tag]["units"][unit]
    except (KeyError, TypeError):
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    if d.empty or "filed" not in d or "end" not in d:
        return pd.DataFrame()
    if "form" in d:
        d = d[d["form"].isin(FORMS)]
    d = d.copy()
    d["end"] = pd.to_datetime(d["end"], errors="coerce")
    d["filed"] = pd.to_datetime(d["filed"], errors="coerce")
    if "start" in d:
        d["start"] = pd.to_datetime(d["start"], errors="coerce")
    d["val"] = pd.to_numeric(d["val"], errors="coerce")
    d = d.dropna(subset=["end", "filed", "val"])
    d = d[d["filed"] >= d["end"] + pd.Timedelta(days=10)]          # guard against mis-dated facts
    return d


def _first_filed(d: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Point-in-time: keep the ORIGINAL (earliest filed) number for each period."""
    return d.sort_values("filed").drop_duplicates(subset=keys, keep="first")


def _flow_quarters(facts: Dict, tags: List[str]) -> pd.DataFrame:
    """Discrete quarterly values with their availability date: columns end, val, filed."""
    for tag in tags:
        d = _entries(facts, "us-gaap", tag, "USD")
        if d.empty or "start" not in d:
            continue
        d = _first_filed(d.dropna(subset=["start"]), ["start", "end"])
        d["days"] = (d["end"] - d["start"]).dt.days
        q = d[d["days"].between(80, 100)][["start", "end", "val", "filed"]]
        ytd = {lo: d[d["days"].between(lo, hi)] for lo, hi in ((170, 190), (260, 285), (350, 380))}
        have = set(q["end"])
        extra = []
        # Q4 = FY - 9M (same fiscal-year start); Q2 = H1 - Q1; Q3 = 9M - H1
        for big_lo, small_lo in ((350, 260), (260, 170), (170, None)):
            for _, b in ytd[big_lo].iterrows():
                if b["end"] in have:
                    continue
                if small_lo is None:
                    q1 = q[(q["start"] == b["start"])]
                    if q1.empty:
                        continue
                    sm = q1.iloc[0]
                else:
                    cand = ytd[small_lo][ytd[small_lo]["start"] == b["start"]]
                    if cand.empty:
                        continue
                    sm = cand.iloc[0]
                extra.append({"start": sm["end"] + pd.Timedelta(days=1), "end": b["end"], "val": b["val"] - sm["val"],
                              "filed": max(b["filed"], sm["filed"])})
                have.add(b["end"])
        if extra:
            q = pd.concat([q, pd.DataFrame(extra)], ignore_index=True)
        q = q.sort_values("end").drop_duplicates("end", keep="first")
        if len(q) >= 4:
            return q.reset_index(drop=True)
    return pd.DataFrame(columns=["start", "end", "val", "filed"])


def _instant(facts: Dict, tags: List[str], ns: str = "us-gaap", unit: str = "USD") -> pd.DataFrame:
    for tag in tags:
        d = _entries(facts, ns, tag, unit)
        if not d.empty:
            return _first_filed(d, ["end"])[["end", "val", "filed"]].sort_values("end").reset_index(drop=True)
    return pd.DataFrame(columns=["end", "val", "filed"])


def _asof(d: pd.DataFrame, end: pd.Timestamp, filed_by: pd.Timestamp, tol_days: int = 200) -> (float, Optional[pd.Timestamp]):
    """Latest instant value with end <= `end` (within tol) that was filed by `filed_by`."""
    if d.empty:
        return np.nan, None
    m = d[(d["end"] <= end) & (d["end"] >= end - pd.Timedelta(days=tol_days)) & (d["filed"] <= filed_by)]
    if m.empty:
        return np.nan, None
    r = m.sort_values("end").iloc[-1]
    return float(r["val"]), r["filed"]


def parse_companyfacts(ticker: str, facts: Dict) -> pd.DataFrame:
    flows = {k: _flow_quarters(facts, tags) for k, tags in FLOW_TAGS.items()}
    base = flows["revenue"] if len(flows["revenue"]) else flows["net_income"]
    if base.empty:
        return pd.DataFrame()
    inst = {k: _instant(facts, tags) for k, tags in INSTANT_TAGS.items()}
    shares = pd.DataFrame(columns=["end", "val", "filed"])
    for ns, tag in SHARE_TAGS:
        shares = _instant(facts, [tag], ns=ns, unit="shares")
        if not shares.empty:
            break
    recs = []
    for end in base["end"]:
        rec = {"ticker": ticker, "year": end.year, "period": end.month, "period_end": end}
        filed = []
        for k, q in flows.items():
            w = q[(q["end"] <= end) & (q["end"] > end - pd.Timedelta(days=340))].sort_values("end")
            if len(w) == 4 and (w["end"].diff().dt.days.dropna().between(70, 110)).all():
                rec[f"{k}_ttm"] = float(w["val"].sum())
                filed.append(w["filed"].max())
            else:
                rec[f"{k}_ttm"] = np.nan
        if not filed:
            continue
        avail = max(filed)
        eq, f_eq = _asof(inst["equity"], end, avail)
        lt, _ = _asof(inst["debt_lt"], end, avail)
        st, _ = _asof(inst["debt_st"], end, avail)
        sh, _ = _asof(shares, end + pd.Timedelta(days=120), avail + pd.Timedelta(days=1), tol_days=400)
        rec.update({"equity": eq, "fin_debt": np.nansum([lt, st]) if np.isfinite(lt) or np.isfinite(st) else np.nan,
                    "paid_in": sh, "avail_date": avail + pd.Timedelta(days=1)})
        recs.append(rec)
    d = pd.DataFrame(recs)
    if d.empty:
        return d
    d = d.sort_values("period_end").reset_index(drop=True)
    prev = d.set_index("period_end")["revenue_ttm"]
    growth = []
    for _, r in d.iterrows():
        m = prev[(prev.index <= r["period_end"] - pd.Timedelta(days=350)) & (prev.index >= r["period_end"] - pd.Timedelta(days=380))]
        g = (r["revenue_ttm"] / m.iloc[-1] - 1.0) * 100.0 if len(m) and m.iloc[-1] and np.isfinite(m.iloc[-1]) and m.iloc[-1] > 0 else np.nan
        growth.append(g)
    d["rev_growth_pct"] = growth
    return d.replace([np.inf, -np.inf], np.nan)


def fetch_company(ticker: str, cik: int, session: Optional[requests.Session] = None) -> pd.DataFrame:
    s = session or requests
    for attempt in range(3):
        try:
            r = s.get(FACTS_URL.format(cik=cik), headers=_headers(), timeout=30)
            if r.status_code == 404:
                return pd.DataFrame()
            if r.status_code in (403, 429):
                time.sleep(15 * (attempt + 1))
                continue
            r.raise_for_status()
            return parse_companyfacts(ticker, r.json())
        except Exception:
            time.sleep(2 * (attempt + 1))
    return pd.DataFrame()


# ------------------------------------------------------------------ engine API (same as BIST module)
def build_point_in_time(raw: pd.DataFrame) -> pd.DataFrame:
    """The cache already stores the point-in-time table; just restore dtypes."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    d = raw.copy()
    for c in ("period_end", "avail_date"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    return d.dropna(subset=["period_end", "avail_date"])


def load_history(tickers: List[str], start_year: int, refresh_years: int = 2, max_workers: int = 1) -> pd.DataFrame:
    """Refresh every ticker from EDGAR (sequential, rate-limited); fall back to the cache per ticker."""
    cache = pd.DataFrame()
    if os.path.exists(C.FUNDAMENTALS_CACHE_FILE):
        try:
            cache = pd.read_csv(C.FUNDAMENTALS_CACHE_FILE)
        except Exception:
            cache = pd.DataFrame()
    try:
        cik = load_cik_map()
    except Exception as exc:
        print(f"⚠️ SEC ticker listesi alınamadı ({exc}); bilanço önbelleği kullanılıyor.")
        return build_point_in_time(cache)
    sess = requests.Session()
    got, ok = [], 0
    t0 = time.time()
    for i, t in enumerate(tickers):
        c = cik.get(sec_key(t))
        if c is None:
            continue
        d = fetch_company(t, c, sess)
        if not d.empty:
            got.append(d)
            ok += 1
        time.sleep(REQ_PAUSE)
        if (i + 1) % 250 == 0:
            print(f"   SEC EDGAR: {i + 1}/{len(tickers)} şirket ({ok} veri) · {time.time() - t0:.0f}s")
    fresh = pd.concat(got, ignore_index=True) if got else pd.DataFrame()
    if not cache.empty:
        keep = cache[~cache["ticker"].isin(set(fresh["ticker"]) if not fresh.empty else set())]
        fresh = pd.concat([fresh, keep], ignore_index=True) if not fresh.empty else keep
    fresh = fresh[pd.to_datetime(fresh["period_end"], errors="coerce").dt.year >= start_year - 1] if not fresh.empty else fresh
    try:
        os.makedirs(C.DATA_DIR, exist_ok=True)
        fresh.to_csv(C.FUNDAMENTALS_CACHE_FILE, index=False, float_format="%.6g")
    except Exception:
        pass
    print(f"📚 SEC EDGAR bilanço: {ok}/{len(tickers)} şirket")
    return build_point_in_time(fresh)


def point_in_time(pit: pd.DataFrame, date, tickers: List[str]) -> pd.DataFrame:
    """Latest statement per ticker that was public on `date`."""
    if pit is None or pit.empty:
        return pd.DataFrame(columns=["ticker"])
    d = pit[(pit["avail_date"] <= pd.Timestamp(date)) & pit["ticker"].isin(tickers)]
    if d.empty:
        return pd.DataFrame(columns=["ticker"])
    return d.sort_values("period_end").groupby("ticker").tail(1)
