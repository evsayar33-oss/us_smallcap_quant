"""System health — one place that says whether the engine can be trusted today.

Checks (each GREEN / YELLOW / RED, the overall status is the worst one):
  data      : TradingView scan size, CPI source, fundamentals coverage
  pipeline  : last after-close run, this month's review, stuck orders, weekly self-test
  model     : live signal strength (IC) vs the backtest, live confidence calibration, model age
  live test : "canlı backtest" — is the live portfolio inside the range the backtest says is normal
              for a portfolio of the same age? (percentile of live excess return vs Russell 2000 among all
              backtest windows of the same length)
Also returns win-rate tables (live vs backtest) for the dashboard.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config as C

GREEN, YELLOW, RED = "OK", "UYARI", "KRİTİK"
_ORDER = {GREEN: 0, YELLOW: 1, RED: 2}


def _chk(key: str, name: str, status: str, msg: str, value=None) -> Dict:
    return {"key": key, "name": name, "status": status, "msg": msg, "value": value}


def _bdays(a, b) -> int:
    try:
        return int(np.busday_count(pd.Timestamp(a).date(), pd.Timestamp(b).date()))
    except Exception:
        return 999


def load_backtest_nav() -> pd.DataFrame:
    p = getattr(C, "BACKTEST_NAV_FILE", None)
    if not p or not os.path.exists(p):
        return pd.DataFrame()
    try:
        d = pd.read_csv(p, parse_dates=["tarih"])
        return d.dropna(subset=["nav"])
    except Exception:
        return pd.DataFrame()


def live_vs_test(nav_df: Optional[pd.DataFrame], bt_nav: pd.DataFrame) -> Dict:
    """Where does the live portfolio sit inside the backtest's distribution for the same holding age?"""
    if nav_df is None or len(nav_df) < 2 or bt_nav.empty:
        return {"status": "WAIT"}
    n = nav_df.copy()
    n["tarih"] = pd.to_datetime(n["tarih"])
    n = n.set_index("tarih").sort_index()
    days = (n.index[-1] - n.index[0]).days
    h = days / 30.44
    live_ret = (n["nav"].iloc[-1] / n["nav"].iloc[0] - 1) * 100
    x = pd.to_numeric(n.get("xu100"), errors="coerce").ffill().bfill() if "xu100" in n else pd.Series(dtype=float)
    live_x = (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) and x.iloc[0] > 0 else np.nan
    out = {"months": round(h, 1), "live_ret_pct": round(float(live_ret), 2),
           "live_xu100_pct": None if not np.isfinite(live_x) else round(float(live_x), 2)}
    if h < 1:
        out["status"] = "WAIT"
        return out
    b = bt_nav.set_index("tarih").sort_index()
    step = max(int(round(days * 252 / 365.25)), 1)
    if len(b) <= step + 20:
        out["status"] = "WAIT"
        return out
    r = (b["nav"].shift(-step) / b["nav"] - 1) * 100
    bx = pd.to_numeric(b["xu100"], errors="coerce").ffill()
    rx = (bx.shift(-step) / bx - 1) * 100
    ex = (r - rx).dropna()
    r = r.dropna()
    if len(r) < 50:
        out["status"] = "WAIT"
        return out
    out["band_ret_pct"] = {k: round(float(np.percentile(r, q)), 1) for k, q in (("p10", 10), ("p50", 50), ("p90", 90))}
    if np.isfinite(live_x) and len(ex):
        live_ex = live_ret - live_x
        out["live_excess_pp"] = round(float(live_ex), 2)
        out["band_excess_pp"] = {k: round(float(np.percentile(ex, q)), 1) for k, q in (("p10", 10), ("p50", 50), ("p90", 90))}
        out["percentile"] = round(float((ex < live_ex).mean() * 100), 1)
    else:
        out["percentile"] = round(float((r < live_ret).mean() * 100), 1)
    out["status"] = "OK"
    return out


def confidence_live_check(snaps: pd.DataFrame) -> Dict:
    """Live calibration: confidence given at the time vs what happened 12 (and 3) months later."""
    out = {"n_resolved": 0}
    if snaps is None or snaps.empty or "confidence" not in snaps:
        return out
    s = snaps.copy()
    s["confidence"] = pd.to_numeric(s["confidence"], errors="coerce")
    for h, col in (("12m", "fwd_ret"), ("3m", "fwd_3m")):
        if col not in s:
            continue
        d = s.dropna(subset=["confidence", col]).copy()
        if d.empty:
            continue
        med = d.groupby("tarih")[col].transform("median")
        d["y"] = (d[col] > med).astype(float)
        top = d[d.get("selected", pd.Series(False, index=d.index)).astype(str).str.lower() == "true"]
        out[h] = {"n": int(len(d)), "months": int(d["tarih"].nunique()),
                  "all_pred": round(float(d["confidence"].mean()), 3), "all_real": round(float(d["y"].mean()), 3),
                  "picks_n": int(len(top)),
                  "picks_pred": None if top.empty else round(float(top["confidence"].mean()), 3),
                  "picks_real": None if top.empty else round(float(top["y"].mean()), 3)}
        if h == "12m":
            out["n_resolved"] = int(len(d))
    return out


def evaluate(state: Dict, snaps: pd.DataFrame, nav_df: Optional[pd.DataFrame], report: Optional[Dict],
             today, bt_nav: Optional[pd.DataFrame] = None) -> Dict:
    today = pd.Timestamp(today).normalize()
    checks: List[Dict] = []
    dq = state.get("data_quality") or {}
    rows = int(dq.get("rows") or 0)
    fb = (dq.get("fetch") or {}).get("fallback")
    st = GREEN if rows >= 300 and not fb else (YELLOW if rows >= 150 else RED)
    checks.append(_chk("scan", "Veri: hisse taraması", st,
                       f"TradingView {rows} hisse döndürdü" + (" (yedek alan seti)" if fb else ""), rows))

    inf = state.get("inflation") or {}
    ist = inf.get("status")
    st = GREEN if ist == "OK" else (YELLOW if ist in ("STALE", "PROXY") else RED)
    checks.append(_chk("cpi", "Veri: TÜFE", st, {"OK": f"resmî TÜFE ({inf.get('source')}, son ay {inf.get('last_month')})",
                                                  "STALE": "TÜFE 4 aydan eski", "PROXY": "TÜFE yerine dolar vekili"}.get(
        ist, "TÜFE alınamadı (yalnız hedef kartını etkiler)"), ist))

    cov = (state.get("last_rebalance") or {}).get("coverage") or {}
    fund = [v for k, v in cov.items() if k in C.FUNDAMENTAL_FACTORS]
    fc = float(np.mean(fund)) if fund else None
    st = GREEN if fc is not None and fc >= 0.7 else (YELLOW if fc is not None and fc >= 0.4 else (YELLOW if fc is None else RED))
    checks.append(_chk("fund", "Veri: bilanço verileri", st,
                       "henüz tarama yok" if fc is None else f"hisselerin %{fc * 100:.0f}'inde bilanço verisi var", fc))

    lr = state.get("last_run") or {}
    last_eod = state.get("last_eod_date") or (lr.get("date") if lr.get("mode") == "EOD" else None)
    gap = _bdays(last_eod, today) if last_eod else 999
    st = GREEN if gap <= 2 else (YELLOW if gap <= 5 else RED)
    checks.append(_chk("run", "Günlük çalışma", st,
                       "hiç kapanış sonrası çalışma yok" if not last_eod else f"son kapanış işlemi {last_eod} ({gap} iş günü önce)", gap))

    pf = state.get("portfolio") or {}
    month = today.strftime("%Y-%m")
    reviewed = pf.get("last_rebalance_month") == month
    sessions_in = _bdays(today.replace(day=1), today)
    st = GREEN if reviewed or sessions_in <= 2 else (YELLOW if sessions_in <= 5 else RED)
    checks.append(_chk("review", "Aylık tarama", st, "bu ayın taraması yapıldı" if reviewed else
                       f"bu ayın taraması henüz yapılmadı (ayın {sessions_in}. iş günü)", reviewed))

    stuck = [o["ticker"] for o in pf.get("pending", []) if int(o.get("age", 0)) >= 3]
    checks.append(_chk("orders", "Bekleyen emirler", YELLOW if stuck else GREEN,
                       ("3+ gündür gerçekleşmeyen: " + ", ".join(stuck)) if stuck else "takılı emir yok", len(stuck)))

    g = state.get("autonomy_guard") or {}
    ok = (g.get("self_test") or {}).get("passed")
    checks.append(_chk("selftest", "Öz test", GREEN if ok else (YELLOW if ok is None else RED),
                       "hesaplamalar doğrulandı" if ok else ("henüz çalışmadı" if ok is None else "öz test BAŞARISIZ"), ok))

    ic = (state.get("model") or {}).get("ic_live_12m") or {}
    bt_ic = ((report or {}).get("oos_composite_ic_12m") or {}).get("mean")
    n_ic = int(ic.get("n_dates") or 0)
    if n_ic < 6:
        checks.append(_chk("ic", "Sinyal gücü (canlı)", GREEN,
                           f"canlı sonuç birikiyor ({n_ic}/6 ay; ilk ölçüm alımdan 12 ay sonra)", n_ic))
    else:
        m, t = float(ic.get("ic_mean") or 0), float(ic.get("t_nw") or 0)
        ref = float(bt_ic or 0.1)
        st = GREEN if m >= 0.5 * ref else (YELLOW if m >= 0 or t > -1.5 else RED)
        checks.append(_chk("ic", "Sinyal gücü (canlı)", st, f"canlı IC {m:.3f} (test {ref:.3f}, t={t:.1f}, {n_ic} ay)".replace(".", ","), m))

    cl = state.get("confidence_live") or {}
    c12 = cl.get("12m") or {}
    if (c12.get("picks_n") or 0) >= 30:
        d = abs((c12.get("picks_pred") or 0) - (c12.get("picks_real") or 0))
        st = GREEN if d <= 0.08 else (YELLOW if d <= 0.15 else RED)
        checks.append(_chk("conf", "Güven oranı tutarlılığı", st,
                           f"öneriler için tahmin %{c12['picks_pred'] * 100:.0f}, gerçekleşen %{c12['picks_real'] * 100:.0f}"
                           f" ({c12['picks_n']} öneri)", d))
    else:
        checks.append(_chk("conf", "Güven oranı tutarlılığı", GREEN,
                           f"canlı ölçüm birikiyor ({c12.get('picks_n', 0)}/30 sonuçlanmış öneri)", None))

    gen = (report or {}).get("generated_at")
    age = (today - pd.Timestamp(gen[:10])).days if gen else None
    st = GREEN if age is not None and age <= 45 else (YELLOW if age is not None and age <= 100 else RED)
    checks.append(_chk("model_age", "Model güncelliği", st,
                       "backtest hiç çalışmadı" if age is None else f"model {age} gün önce yeniden eğitildi (her ayın 2'sinde otomatik)", age))

    btn = bt_nav if bt_nav is not None else load_backtest_nav()
    lv = live_vs_test(nav_df, btn)
    if lv.get("status") != "OK":
        why = ("test NAV'ı yok — Walk Forward Backtest çalışınca ölçülür" if btn is None or btn.empty
               else f"canlı geçmiş kısa ({lv.get('months', 0)} ay); 1 aydan sonra ölçülür")
        checks.append(_chk("live", "Canlı sonuç vs test", GREEN if not (btn is None or btn.empty) else YELLOW, why, None))
    else:
        p = lv.get("percentile", 50)
        st = GREEN if p >= 15 else (YELLOW if p >= 5 else RED)
        what = "Russell 2000'e göre fark" if "live_excess_pp" in lv else "getiri"
        checks.append(_chk("live", "Canlı sonuç vs test", st,
                           f"{str(lv['months']).replace('.', ',')} aylık {what}, testteki aynı süreli dönemlerin %{p:.0f}'inden iyi"
                           + (" (normal aralıkta)" if p >= 15 else " (testin alt ucunda)" if p >= 5 else " (testte neredeyse hiç görülmeyen kadar kötü)"), p))
    overall = max((c["status"] for c in checks), key=lambda s: _ORDER[s])
    live_lots = ((state.get("performance") or {}).get("lots") or {})
    bt_lots = (report or {}).get("closed_lots") or {}
    win = {k: {"live": live_lots.get(k), "test": bt_lots.get(k)} for k in
           ("closed_lots", "hit_nominal_pct", "hit_beat_xu100_pct", "hit_beat_cpi_pct", "hit_beat_all_pct", "avg_nominal_pct", "avg_months_held")}
    return {"date": str(today.date()), "overall": overall, "checks": checks, "live_vs_test": lv, "win_rates": win,
            "red": [c["name"] for c in checks if c["status"] == RED],
            "yellow": [c["name"] for c in checks if c["status"] == YELLOW]}


def emoji(status: str) -> str:
    return {GREEN: "🟢", YELLOW: "🟡", RED: "🔴"}.get(status, "⚪")
