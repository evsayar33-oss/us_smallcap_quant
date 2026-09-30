"""US CPI (CPIAUCSL) and 3-month T-bill (DTB3) from FRED — keyless CSV, cached in data/.

CPI chain: data/cpi_manual.csv (optional) -> FRED CPIAUCSL -> FRED CPIAUCNS -> cache.
Cash (idle money / benchmark): FRED DTB3 daily -> monthly average, cached in data/tbill_us.csv.
The engine never invents inflation figures; an unpublished month stays unresolved.
(Module API kept identical to the BIST engine.)
"""
from __future__ import annotations

import io
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import config as C

# TCMB moved EVDS to evds3 in 2025 and TÜİK re-based CPI to 2025=100:
# old code TP.FG.J0 (2003=100, archived) is spliced with TP.TUKFIY2025.GENEL.
EVDS_BASES = ["https://evds3.tcmb.gov.tr/igmevdsms-dis/", "https://evds2.tcmb.gov.tr/service/evds/"]
EVDS_QUERY = "series={code}&startDate=01-01-2003&endDate=01-12-2035&type=json&frequency=5"
EVDS_CODES = [c.strip() for c in os.environ.get("EVDS_CPI_SERIES", "TP.FG.J0,TP.TUKFIY2025.GENEL").split(",") if c.strip()]
FRED_URLS = ["https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL",
             "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS"]
FRED_TBILL_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
FRED_API = "https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={key}&file_type=json&observation_start=1990-01-01"
BLS_API = "https://api.bls.gov/publicAPI/v1/timeseries/data/CUSR0000SA0?startyear={a}&endyear={b}"


def _fred_api(sid: str) -> Optional[pd.Series]:
    """Official FRED API (needs the free FRED_API_KEY secret). Much more reliable from GitHub
    runners than fredgraph.csv, which often times out."""
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return None
    last = None
    for _ in range(3):
        try:
            r = requests.get(FRED_API.format(sid=sid, key=key), timeout=45)
            r.raise_for_status()
            obs = r.json().get("observations", [])
            s = pd.Series({pd.Timestamp(o["date"]): pd.to_numeric(o["value"], errors="coerce") for o in obs}).dropna()
            if len(s) > 24:
                return s.sort_index()
        except Exception as exc:
            last = exc
    if last:
        raise last
    return None


def _from_fred_api() -> Optional[pd.Series]:
    s = _fred_api("CPIAUCSL")
    return None if s is None else _clean(s)


def _from_bls() -> Optional[pd.Series]:
    """BLS public API v1 (keyless, 10 years per call): CPI-U all items, seasonally adjusted."""
    this = pd.Timestamp.now().year
    parts = []
    for a in range(this - 19, this + 1, 10):
        r = requests.get(BLS_API.format(a=a, b=min(a + 9, this)), timeout=45, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        for srs in r.json().get("Results", {}).get("series", []):
            for d in srs.get("data", []):
                if str(d.get("period", "")).startswith("M") and d["period"] != "M13":
                    parts.append((pd.Timestamp(int(d["year"]), int(d["period"][1:]), 1), float(d["value"])))
    if not parts:
        return None
    return _clean(pd.Series(dict(parts)))
DBNOMICS_URLS = ["https://api.db.nomics.world/v22/series/IMF/CPI/M.TR.PCPI_IX?observations=1&format=json",
                 "https://api.db.nomics.world/v22/series/OECD/MEI/TUR.CPALTT01.IXOB.M?observations=1&format=json"]
PROXY_SAFETY_PP = 0.0     # no FX proxy for the US


def _month(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return pd.Timestamp(year=t.year, month=t.month, day=1)


def _clean(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0]
    s.index = [_month(i) for i in s.index]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _from_manual() -> Optional[pd.Series]:
    if not os.path.exists(C.CPI_MANUAL_FILE):
        return None
    df = pd.read_csv(C.CPI_MANUAL_FILE)
    return _clean(pd.Series(df.iloc[:, 1].to_numpy(), index=pd.to_datetime(df.iloc[:, 0])))


def _from_evds() -> Optional[pd.Series]:
    key = os.environ.get("EVDS_API_KEY")
    if not key:
        return None
    out, errors = None, []
    for code in EVDS_CODES:
        field = code.replace(".", "_")
        for base in EVDS_BASES:
            try:
                r = requests.get(base + EVDS_QUERY.format(code=code), headers={"key": key}, timeout=40)
                r.raise_for_status()
                idx, val = [], []
                for it in r.json().get("items", []):
                    t = str(it.get("Tarih", ""))
                    v = it.get(field)
                    if not t or v in (None, ""):
                        continue
                    y, m = t.split("-")[:2]
                    idx.append(pd.Timestamp(int(y), int(m), 1))
                    val.append(float(v))
                if idx:
                    s = _clean(pd.Series(val, index=idx))
                    out = s if out is None else splice(out, s)
                    break
            except Exception as exc:
                errors.append(f"{code}@{base.split('/')[2]}: {str(exc)[:60]}")
    if out is None and errors:
        raise RuntimeError("; ".join(errors[:3]))
    return out


def splice(base: pd.Series, ext: pd.Series) -> pd.Series:
    """Extend `base` with the later months of `ext`, rescaled on the overlap (base change safe)."""
    if base is None or base.empty:
        return ext
    if ext is None or ext.empty or ext.index[-1] <= base.index[-1]:
        return base
    overlap = base.index.intersection(ext.index)
    if len(overlap) == 0:
        return base
    k = float(base[overlap[-1]] / ext[overlap[-1]])
    tail = ext[ext.index > base.index[-1]] * k
    return pd.concat([base, tail]).sort_index()


def _from_fred() -> Optional[pd.Series]:
    last_exc = None
    for url in FRED_URLS:
        for _ in range(2):
            try:
                r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                s = _clean(pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),
                                     index=pd.to_datetime(df.iloc[:, 0])))
                if len(s) >= 24:
                    return s
            except Exception as exc:
                last_exc = exc
    if last_exc:
        raise last_exc
    return None


def _from_dbnomics() -> Optional[pd.Series]:
    last_exc = None
    for url in DBNOMICS_URLS:
        try:
            r = requests.get(url, timeout=40, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            docs = r.json().get("series", {}).get("docs", [])
            if not docs:
                continue
            per, val = docs[0].get("period", []), docs[0].get("value", [])
            s = _clean(pd.Series([v if v not in ("NA", None) else np.nan for v in val],
                                 index=pd.to_datetime([p + "-01" if len(p) == 7 else p for p in per])))
            if len(s) >= 24:
                return s
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    return None


def proxy_from_fx(fx: Optional[pd.Series]) -> pd.Series:
    """LAST RESORT: monthly average USDTRY as an inflation proxy (clearly flagged).
    TRY depreciation tracks the inflation differential only loosely, so a safety
    margin is added to the expected inflation when this proxy is in use."""
    return pd.Series(dtype=float)          # US: no FX-based inflation proxy
    if fx is None or len(fx) < 300:
        return pd.Series(dtype=float)
    m = pd.to_numeric(fx, errors="coerce").dropna()
    m.index = pd.to_datetime(m.index)
    m = m.resample("MS").mean().dropna()
    return m[m > 0]


def _from_cache() -> Optional[pd.Series]:
    if not os.path.exists(C.CPI_CACHE_FILE):
        return None
    df = pd.read_csv(C.CPI_CACHE_FILE)
    return _clean(pd.Series(df["cpi"].to_numpy(), index=pd.to_datetime(df["tarih"])))


def load_cpi(allow_network: bool = True) -> Tuple[pd.Series, Dict]:
    tried = {}
    sources = [("manual", _from_manual)]
    if allow_network:
        sources += [("fred_api", _from_fred_api), ("fred", _from_fred), ("bls", _from_bls)]
    sources += [("cache", _from_cache)]
    best, best_name = None, None
    for name, fn in sources:
        try:
            s = fn()
        except Exception as exc:
            tried[name] = f"error: {str(exc)[:120]}"
            continue
        if s is None or len(s) < 24:
            tried[name] = "unavailable"
            continue
        tried[name] = f"ok (last {s.index[-1]:%Y-%m})"
        if best is None:
            best, best_name = s, name
        elif s.index[-1] > best.index[-1]:
            best = splice(best, s)          # fresher months from a later source
            best_name = f"{best_name}+{name}"
    meta = {"source": best_name, "tried": tried}
    if best is None:
        meta["status"] = "UNAVAILABLE"
        return pd.Series(dtype=float), meta
    meta["last_month"] = str(best.index[-1].date())
    meta["status"] = "OK"
    if best_name != "cache":
        try:
            os.makedirs(C.DATA_DIR, exist_ok=True)
            pd.DataFrame({"tarih": best.index.strftime("%Y-%m-%d"), "cpi": best.values}).to_csv(C.CPI_CACHE_FILE, index=False)
        except Exception:
            pass
    return best, meta


def cpi_ratio(cpi: pd.Series, start, end) -> float:
    """CPI(end month)/CPI(start month); NaN if a month is not published."""
    if cpi is None or cpi.empty:
        return np.nan
    a, b = _month(start), _month(end)
    if a not in cpi.index or b not in cpi.index:
        return np.nan
    return float(cpi[b] / cpi[a])


def cpi_ratio_vec(cpi: pd.Series, starts, ends) -> np.ndarray:
    if cpi is None or cpi.empty:
        return np.full(len(starts), np.nan)
    m = cpi.to_dict()
    out = []
    for a, b in zip(starts, ends):
        va, vb = m.get(_month(a)), m.get(_month(b))
        out.append(vb / va if va and vb else np.nan)
    return np.asarray(out, float)


def load_cpi_or_proxy(fx: Optional[pd.Series] = None, allow_network: bool = True) -> Tuple[pd.Series, Dict]:
    cpi, meta = load_cpi(allow_network)
    if meta.get("status") == "OK":
        # an official series that has stopped updating (> 4 months) is still used for history,
        # but flagged so the dashboard/Telegram can warn
        if cpi.index[-1] < pd.Timestamp.now().normalize() - pd.DateOffset(months=4):
            meta["status"] = "STALE"
        return cpi, meta
    px = proxy_from_fx(fx)
    if len(px) >= 24:
        meta.update({"source": "USDTRY_PROXY", "status": "PROXY", "last_month": str(px.index[-1].date())})
        return px, meta
    return cpi, meta


def inflation_stats(cpi: pd.Series, proxy: bool = False) -> Dict:
    if cpi is None or len(cpi) < 13:
        return {"yoy_pct": None, "ann6m_pct": None, "expected_12m_pct": None, "is_proxy": proxy}
    last = cpi.index[-1]
    yoy = (cpi.iloc[-1] / cpi.iloc[-13] - 1.0) * 100.0
    ann6 = ((cpi.iloc[-1] / cpi.iloc[-7]) ** 2 - 1.0) * 100.0 if len(cpi) >= 7 else yoy
    # Expected next-12m inflation: blend of trailing year and recent 6m pace. Under-estimating
    # inflation is the costly error for a beat-inflation objective -> proxy gets a safety margin.
    exp = max(0.5 * yoy + 0.5 * ann6, 0.0) + (PROXY_SAFETY_PP if proxy else 0.0)
    return {"last_month": str(last.date()), "yoy_pct": round(float(yoy), 2),
            "ann6m_pct": round(float(ann6), 2), "expected_12m_pct": round(float(exp), 2), "is_proxy": proxy}


# ------------------------------------------------------------------ TL money-market proxy for idle cash
def load_cash_rate(allow_network: bool = True) -> Tuple[pd.Series, Dict]:
    """Monthly average 3-month T-bill rate (%, FRED DTB3) -> idle-cash yield and deposit benchmark."""
    s, meta = None, {"series": C.CASH_RATE_SERIES}
    if allow_network:
        errs = {}
        try:                                                    # 1) official FRED API (key)
            d = _fred_api("DTB3")
            if d is not None:
                s, meta["source"] = d.resample("MS").mean().dropna(), "fred_api_dtb3"
        except Exception as exc:
            errs["fred_api"] = str(exc)[:80]
        if s is None:                                           # 2) fredgraph CSV (keyless)
            try:
                r = requests.get(FRED_TBILL_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                d = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(df.iloc[:, 0])).dropna()
                m = d.resample("MS").mean().dropna()
                if len(m) > 24:
                    s, meta["source"] = m, "fred_dtb3"
            except Exception as exc:
                errs["fred"] = str(exc)[:80]
        if s is None:                                           # 3) Yahoo ^IRX (13-week T-bill yield, %)
            try:
                import yfinance as yf
                raw = yf.download("^IRX", start="2005-01-01", progress=False, auto_adjust=False, threads=False)
                c = raw["Close"]
                if isinstance(c, pd.DataFrame):
                    c = c.iloc[:, 0]
                c.index = pd.to_datetime(c.index).tz_localize(None)
                m = c.dropna().resample("MS").mean().dropna()
                if len(m) > 24:
                    s, meta["source"] = m, "yahoo_irx"
            except Exception as exc:
                errs["yahoo"] = str(exc)[:80]
        if errs:
            meta["errors"] = errs
    if s is not None and len(s):
        try:
            os.makedirs(C.DATA_DIR, exist_ok=True)
            pd.DataFrame({"tarih": s.index.strftime("%Y-%m-%d"), "rate": s.values}).to_csv(C.POLICY_RATE_CACHE_FILE, index=False)
        except Exception:
            pass
    elif os.path.exists(C.POLICY_RATE_CACHE_FILE):
        try:
            df = pd.read_csv(C.POLICY_RATE_CACHE_FILE)
            s = pd.Series(df["rate"].to_numpy(float), index=pd.to_datetime(df["tarih"]))
            meta["source"] = "cache"
        except Exception:
            s = None
    if s is None or not len(s):
        meta["source"] = "none"
        return pd.Series(dtype=float), meta
    meta["last_month"] = str(s.index[-1].date())
    return s, meta


def cash_yield_at(rate: pd.Series, date) -> float:
    """Net annual % yield of idle cash at `date` (month known at that time)."""
    if rate is None or not len(rate):
        return float(C.CASH_YIELD_ANNUAL_PCT)
    r = rate[rate.index <= _month(date)]
    if not len(r):
        return float(C.CASH_YIELD_ANNUAL_PCT)
    return float(max(0.0, (float(r.iloc[-1]) - C.CASH_HAIRCUT_PP)) * (1.0 - C.CASH_TAX))
