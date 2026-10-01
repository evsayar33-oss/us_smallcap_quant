"""Hedef Avcısı (target hunter): a separate stock sleeve that sells at a fixed multiple.

Why it exists
    The main engine holds monthly cohorts for a fixed time. The user also wants a system that buys a
    few names and sells only when they reach a target multiple (BIST 5x, US 2x), with as few losers
    as possible. Research on the real panels (2014-2026 BIST, 2011-2026 US; walk-forward split, every
    configuration compared with RANDOM picks under the SAME exit rules) found one entry family that
    beat random in both halves of the sample in both markets: SMALL + CHEAP (book/earnings yield)
    (+ profitable margin in the US). Nominal-price features were rejected: on adjusted prices they
    are look-ahead (bonus issues / reverse splits).

Rules (config.py, TH_*)
    * Each month fill the free slots (TH_SLOTS) with the best-scored eligible names not held.
    * Buy at the next session's open.
    * Sell when: level >= TH_TARGET               -> "hedef"
                 trailing (after TH_TRAIL[0]x, fall TH_TRAIL[1] from the peak)   -> "iz"
                 TH_MAX_MONTHS reached             -> "süre"
                 TH_STOP (None = no stop; research: stops RAISE the loss rate) -> "stop"
    * Levels are chained from daily change_pct (robust to splits / bonus issues) and re-synced to
      adjusted history at each monthly review.

Slow, evidence-based adaptation (champion / challenger)
    * Each monthly backtest also tests a few nearby rule sets (TH_CHALLENGERS). A challenger qualifies only if
      it is better in BOTH halves of the history, beats random by >= TH_CHALLENGER_MIN_EDGE_PP and the champion's
      edge, and does not close more trades at a loss. Live rules change only after TH_CHALLENGER_CONFIRM
      consecutive confirmations and at most once per TH_MIN_MONTHS_BETWEEN_CHANGES; every change is reported.

Stopping (guard)
    * New buys stop (exits keep running) when the latest test no longer beats random, or live results after
      12 months / 8 trades are clearly worse than the test. Holdings down 50%, holdings not trading and market
      crashes raise alerts. A crash alone does not pause buying (that made results worse in both markets).

Self-checking
    * The monthly walk-forward backtest re-runs this sleeve on real data with the SAME code
      (simulate) and a random-pick control; health.py flags it if the edge over random disappears.
    * Every month the full eligible universe is logged with its score (data/th_universe_log.csv.gz).
      Delisted names stay in that log, so after 12+ months the live hit/loss rates are free of
      survivorship bias (live_stats).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as C

KEY = "target_hunter"


_ACTIVE: Dict = {}          # live/backtest overrides chosen by the champion-challenger process


def _cfg(name, default):
    if name in _ACTIVE:
        return _ACTIVE[name]
    return getattr(C, name, default)


def set_active(over: Optional[Dict]) -> None:
    _ACTIVE.clear()
    for k, v in (over or {}).items():
        _ACTIVE[k] = tuple(v) if k == "TH_TRAIL" and isinstance(v, list) else v


def currency() -> str:
    c = str(getattr(C, "CURRENCY", "TRY")).upper()
    return "$" if c == "USD" else "TL"


def enabled() -> bool:
    return bool(_cfg("TH_ENABLED", True))


# ----------------------------------------------------------------------------- scoring
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def components(df: pd.DataFrame) -> pd.DataFrame:
    """Raw component values (higher = better) from whatever columns exist (live snapshot or backtest)."""
    out = pd.DataFrame(index=df.index)
    mc = _col(df, "market_cap").where(lambda x: x > 0)
    out["small"] = -np.log(mc)
    if "equity" in df:
        out["book"] = pd.to_numeric(df["equity"], errors="coerce") / mc
    else:
        pb = _col(df, "pb")
        out["book"] = (1.0 / pb).where(pb.abs() > 1e-9)
    if "net_income" in df:
        out["earn"] = pd.to_numeric(df["net_income"], errors="coerce") / mc
    else:
        pe = _col(df, "pe")
        out["earn"] = (1.0 / pe).where(pe.abs() > 1e-9)
    if "revenue" in df:
        out["sales"] = pd.to_numeric(df["revenue"], errors="coerce") / mc
    else:
        ps = _col(df, "ps")
        out["sales"] = (1.0 / ps).where(ps.abs() > 1e-9)
    out["margin"] = _col(df, "op_margin")
    for c in ("vol60", "vol_ann_pct"):
        if c in df:
            out["volatile"] = pd.to_numeric(df[c], errors="coerce")
    if "from_high" in df:
        out["far_high"] = -pd.to_numeric(df["from_high"], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def score(df: pd.DataFrame, comps: Optional[List[str]] = None) -> pd.Series:
    """Mean cross-sectional percentile of the components; a missing value counts as neutral (0.5)."""
    comps = comps or _cfg("TH_COMPONENTS", ["small", "book", "earn"])
    x = components(df)
    r = [x[c].rank(pct=True).fillna(0.5) for c in comps if c in x]
    return (sum(r) / len(r)) if r else pd.Series(np.nan, index=df.index)


def eligible(df: pd.DataFrame, liq_floor: Optional[float] = None) -> pd.Series:
    ok = _col(df, "market_cap") > 0
    floor = _cfg("TH_MIN_VALUE_TRADED", None)
    floor = liq_floor if floor is None else floor
    if floor is not None and "med_value_traded" in df:
        ok &= pd.to_numeric(df["med_value_traded"], errors="coerce") >= float(floor)
    lo, hi = _cfg("TH_MCAP_MIN", None), _cfg("TH_MCAP_MAX", None)
    mc = _col(df, "market_cap")
    if lo is not None:
        ok &= mc >= lo
    if hi is not None:
        ok &= mc <= hi
    mp = _cfg("TH_MIN_PRICE", None)
    px_col = "close" if "close" in df else ("close_adj" if "close_adj" in df else None)
    if mp is not None and px_col:
        ok &= pd.to_numeric(df[px_col], errors="coerce") >= mp
    return ok.fillna(False)


def ranked(df: pd.DataFrame, liq_floor: Optional[float] = None) -> pd.DataFrame:
    d = df[eligible(df, liq_floor)].copy()
    if d.empty:
        return d.assign(th_score=pd.Series(dtype=float))
    d["th_score"] = score(d).to_numpy()
    return d.sort_values("th_score", ascending=False)


# ----------------------------------------------------------------------------- exit rule (shared)
def exit_reason(level: float, peak: float, entry_date, today) -> Optional[str]:
    T = float(_cfg("TH_TARGET", 5.0))
    stop = _cfg("TH_STOP", None)
    trail = _cfg("TH_TRAIL", None)
    months = _cfg("TH_MAX_MONTHS", None)
    if level >= T:
        return "hedef"
    if stop is not None and level <= 1 - float(stop):
        return "stop"
    if trail is not None and peak >= float(trail[0]) and level <= peak * (1 - float(trail[1])):
        return "iz"
    if months is not None and pd.Timestamp(entry_date) + pd.DateOffset(months=int(months)) <= pd.Timestamp(today):
        return "süre"
    return None


# ----------------------------------------------------------------------------- backtest (same rules)
def simulate(open_w: pd.DataFrame, close_w: pd.DataFrame, picks: Dict[pd.Timestamp, List[str]],
             cost: float = None, start=None, paused=None) -> Tuple[pd.Series, pd.DataFrame]:
    """Slot portfolio on ADJUSTED prices. picks[d] = ranked candidate list decided at the close of d;
    entries at the next session's open, equal slot weight NAV/S, exits at the signal day's close."""
    S = int(_cfg("TH_SLOTS", 8))
    cost = (float(getattr(C, "COST_ROUND_TRIP_PCT", 0.4)) / 200.0) if cost is None else cost
    idx = close_w.index if start is None else close_w.index[close_w.index >= pd.Timestamp(start)]
    O = open_w.reindex(idx).to_numpy(float)
    Cl = close_w.reindex(idx).ffill(limit=10).to_numpy(float)
    col = {t: j for j, t in enumerate(close_w.columns)}
    sig = {}
    for d, lst in picks.items():
        if paused is not None and pd.Timestamp(d) in paused:
            continue
        p = idx.searchsorted(pd.Timestamp(d), side="right")          # next session
        if p < len(idx):
            sig[p] = lst
    cash, pos, trades, nav = 1.0, {}, [], []
    for i in range(len(idx)):
        day = idx[i]
        if i in sig:                                                  # buy at open
            val = cash + sum(q["sh"] * q["last"] for q in pos.values())
            for t in sig[i]:
                if len(pos) >= S:
                    break
                j = col.get(t)
                if t in pos or j is None or not np.isfinite(O[i, j]) or O[i, j] <= 0:
                    continue
                w = min(val / S, cash)
                if w <= 1e-9:
                    break
                cash -= w
                pos[t] = {"j": j, "px": O[i, j], "sh": w * (1 - cost) / O[i, j], "d0": day, "peak": 1.0, "last": O[i, j], "stale": 0}
        for t in list(pos):
            q = pos[t]
            c = Cl[i, q["j"]]
            if np.isfinite(c):
                q["last"], q["stale"] = c, 0
            else:
                q["stale"] += 1
                if q["stale"] < 15:
                    continue
                c = q["last"]                                         # delisted/halted: written at last price
            lvl = c / q["px"]
            q["peak"] = max(q["peak"], lvl)
            why = exit_reason(lvl, q["peak"], q["d0"], day) if q["stale"] < 15 else "veri_yok"
            if why:
                cash += q["sh"] * c * (1 - cost)
                trades.append({"ticker": t, "entry": q["d0"], "exit": day, "ret_pct": (lvl * (1 - cost) ** 2 - 1) * 100,
                               "peak_x": q["peak"], "reason": why})
                del pos[t]
        nav.append(cash + sum(q["sh"] * q["last"] for q in pos.values()))
    for t, q in pos.items():
        trades.append({"ticker": t, "entry": q["d0"], "exit": idx[-1], "ret_pct": (q["last"] / q["px"] - 1) * 100,
                       "peak_x": q["peak"], "reason": "açık"})
    return pd.Series(nav, idx), pd.DataFrame(trades)


def summarize(nav: pd.Series, trades: pd.DataFrame) -> Dict:
    if nav is None or len(nav) < 30:
        return {}
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    out = {"cagr_pct": round((nav.iloc[-1] ** (1 / yrs) - 1) * 100, 2), "max_dd_pct": round(float((nav / nav.cummax() - 1).min() * 100), 2),
           "years": round(yrs, 1), "n_trades": int(len(trades))}
    if len(trades):
        r = trades["ret_pct"]
        out.update({"win_rate_pct": round(float((r > 0).mean() * 100), 1),
                    "loss_rate_pct": round(float((r <= 0).mean() * 100), 1),
                    "target_hit_pct": round(float((trades["reason"] == "hedef").mean() * 100), 1),
                    "heavy_loss_pct": round(float((r < -50).mean() * 100), 1),
                    "avg_win_pct": round(float(r[r > 0].mean()), 1) if (r > 0).any() else None,
                    "avg_loss_pct": round(float(r[r <= 0].mean()), 1) if (r <= 0).any() else None,
                    "avg_trade_pct": round(float(r.mean()), 1),
                    "reasons": trades["reason"].value_counts().to_dict(),
                    "median_months": round(float(((pd.to_datetime(trades["exit"]) - pd.to_datetime(trades["entry"])).dt.days / 30.44).median()), 1)})
    return out


def _picks(cands, liq_floor_fn):
    picks = {}
    for d, df in cands.items():
        r = ranked(df, liq_floor_fn(d) if liq_floor_fn else None)
        if len(r) >= 2 * int(_cfg("TH_SLOTS", 8)):
            picks[d] = r["ticker"].tolist()
    return picks


def _halves(nav: pd.Series, mid) -> List[Optional[float]]:
    out = []
    for g in (nav[nav.index < mid], nav[nav.index >= mid]):
        if len(g) < 60:
            out.append(None)
            continue
        y = (g.index[-1] - g.index[0]).days / 365.25
        out.append(round(((g.iloc[-1] / g.iloc[0]) ** (1 / y) - 1) * 100, 2))
    return out


def backtest(open_w, close_w, cands: Dict[pd.Timestamp, pd.DataFrame], liq_floor_fn=None,
             n_random: int = 20, start=None) -> Dict:
    """cands[d] = cross-section at d (market_cap, equity/pb, net_income/pe, op_margin, med_value_traded, close_adj).
    Returns the sleeve's summary, a random-pick control (same exits, same eligibility), both halves of the
    sample and per-year returns. Uses the CURRENT active rules (config + overrides)."""
    picks = _picks(cands, liq_floor_fn)
    if not picks:
        return {"status": "NO_CANDIDATES"}
    start = start or min(picks)
    nav, tr = simulate(open_w, close_w, picks, start=start)
    rng = np.random.default_rng(7)
    rnd, rnd_h = [], []
    mid = nav.index[0] + (nav.index[-1] - nav.index[0]) / 2
    for _ in range(n_random):
        rp = {d: list(rng.permutation(v)) for d, v in picks.items()}
        nv, tt = simulate(open_w, close_w, rp, start=start)
        rnd.append(summarize(nv, tt))
        rnd_h.append(_halves(nv, mid))
    rc = float(np.mean([x["cagr_pct"] for x in rnd if x]))
    rl = float(np.mean([x.get("loss_rate_pct", np.nan) for x in rnd if x]))
    per_year, prev = {}, None
    for yr, g in nav.groupby(nav.index.year):
        base = prev if prev is not None else g.iloc[0]
        per_year[int(yr)] = round(float((g.iloc[-1] / base - 1) * 100), 1)
        prev = g.iloc[-1]
    s = summarize(nav, tr)
    h = _halves(nav, mid)
    rh = [round(float(np.nanmean([x[i] for x in rnd_h if x[i] is not None])), 2) if rnd_h else None for i in (0, 1)]
    return {"status": "OK", "rules": rules(), **s, "random_cagr_pct": round(rc, 2), "random_loss_rate_pct": round(rl, 1),
            "edge_vs_random_pp": round(s.get("cagr_pct", np.nan) - rc, 2), "per_year": per_year,
            "halves": {"split": str(pd.Timestamp(mid).date()), "cagr_pct": h, "random_cagr_pct": rh},
            "_nav": nav, "_trades": tr}


def _label(over: Dict) -> str:
    parts = []
    for k, v in over.items():
        if k == "TH_TRAIL":
            parts.append("kâr kilidi yok" if not v else f"kâr kilidi {v[0]:g}× sonrası zirveden %{v[1] * 100:.0f}")
        elif k == "TH_MAX_MONTHS":
            parts.append(f"süre {v} ay")
        elif k == "TH_SLOTS":
            parts.append(f"{v} hisse")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def backtest_suite(open_w, close_w, cands, liq_floor_fn=None, champion: Optional[Dict] = None,
                   challengers: Optional[List[Dict]] = None, n_random: int = 20, n_random_ch: int = 10) -> Dict:
    """Champion (active rules) + challengers. A challenger QUALIFIES only if it is better than the champion in
    BOTH halves of the sample, beats random by at least TH_CHALLENGER_MIN_EDGE_PP, beats the champion's edge
    over random, and does not close more trades at a loss. The best qualifier is reported; live adoption needs
    TH_CHALLENGER_CONFIRM consecutive monthly confirmations and TH_MIN_MONTHS_BETWEEN_CHANGES (main.py / monthly)."""
    champion = dict(champion or {})
    set_active(champion)
    try:
        champ = backtest(open_w, close_w, cands, liq_floor_fn, n_random=n_random)
        if champ.get("status") != "OK":
            return champ
        min_edge = float(_cfg("TH_CHALLENGER_MIN_EDGE_PP", 3.0))
        rows = []
        for v in (challengers if challengers is not None else _cfg("TH_CHALLENGERS", [])):
            over = {**champion, **v}
            if over == champion:
                continue
            set_active(over)
            r = backtest(open_w, close_w, cands, liq_floor_fn, n_random=n_random_ch)
            set_active(champion)
            if r.get("status") != "OK":
                continue
            ch, cc = r["halves"]["cagr_pct"], champ["halves"]["cagr_pct"]
            ok = (None not in ch and None not in cc and ch[0] > cc[0] and ch[1] > cc[1]
                  and r["edge_vs_random_pp"] >= max(min_edge, champ["edge_vs_random_pp"])
                  and r.get("loss_rate_pct", 999) <= champ.get("loss_rate_pct", 0))
            rows.append({"change": v, "label": _label(v), "cagr_pct": r["cagr_pct"], "halves_cagr_pct": ch,
                         "edge_vs_random_pp": r["edge_vs_random_pp"], "loss_rate_pct": r.get("loss_rate_pct"),
                         "max_dd_pct": r.get("max_dd_pct"), "qualifies": bool(ok)})
        q = [x for x in rows if x["qualifies"]]
        best = max(q, key=lambda x: x["cagr_pct"]) if q else None
        champ["challengers"] = rows
        champ["best_challenger"] = best
        champ["champion_overrides"] = champion
        return champ
    finally:
        set_active(champion)


def rules() -> Dict:
    return {"slots": _cfg("TH_SLOTS", 8), "target_x": _cfg("TH_TARGET", 5.0), "trail": _cfg("TH_TRAIL", None),
            "max_months": _cfg("TH_MAX_MONTHS", None), "stop": _cfg("TH_STOP", None),
            "components": _cfg("TH_COMPONENTS", ["small", "book", "earn"])}


# ----------------------------------------------------------------------------- live
def _st(state: Dict) -> Dict:
    th = state.setdefault(KEY, {})
    th.setdefault("positions", {})
    th.setdefault("pending", [])
    th.setdefault("closed", [])
    th.setdefault("active", {})
    set_active(th["active"])
    return th


def daily(state: Dict, snap: pd.DataFrame, today, refresh: bool = False, index_close: Optional[pd.Series] = None) -> List[Dict]:
    """Fill pending buys at today's open, chain levels with change_pct, emit exit signals at the close,
    and raise alerts for unexpected events (a holding down 50%, a holding not trading, a market crash)."""
    if not enabled() or refresh or snap is None or snap.empty:
        return []
    th = _st(state)
    today = pd.Timestamp(today)
    s = snap.set_index("ticker")
    ev, keep = [], []
    for o in th["pending"]:
        t = o["ticker"]
        if t in s.index and np.isfinite(s.at[t, "open"]) and s.at[t, "open"] > 0 and np.isfinite(s.at[t, "close"]):
            op, cl = float(s.at[t, "open"]), float(s.at[t, "close"])
            th["positions"][t] = {"entry_date": str(today.date()), "entry_px": round(op, 4), "level": cl / op,
                                  "peak": max(1.0, cl / op), "score_pct": o.get("score_pct"), "last_date": str(today.date()),
                                  "last_close": round(cl, 4), "missing": 0}
            ev.append({"type": "AL", "ticker": t, "px": op})
        elif pd.Timestamp(o["date"]) + pd.Timedelta(days=7) >= today:
            keep.append(o)
    th["pending"] = keep
    for t, p in list(th["positions"].items()):
        if p.get("last_date") != str(today.date()):
            if t in s.index and np.isfinite(s.at[t, "change_pct"]):
                p["level"] *= 1 + float(s.at[t, "change_pct"]) / 100.0
                p["last_date"] = str(today.date())
                p["peak"] = max(p["peak"], p["level"])
                p["missing"] = 0
                if np.isfinite(s.at[t, "close"]):
                    p["last_close"] = round(float(s.at[t, "close"]), 4)
            else:
                p["missing"] = int(p.get("missing", 0)) + 1
                if p["missing"] == 5:
                    ev.append({"type": "UYARI", "ticker": t, "kind": "işlem_yok"})
        if p["level"] <= 0.5 and not p.get("alert50"):
            p["alert50"] = True
            ev.append({"type": "UYARI", "ticker": t, "kind": "düşüş50", "level": p["level"]})
        why = exit_reason(p["level"], p["peak"], p["entry_date"], today)
        if why:
            ev.append(_close(th, t, p, today, why))
    if index_close is not None and len(index_close) > 60:
        x = pd.Series(index_close).dropna()
        dd = float(x.iloc[-1] / x.iloc[-60:].max() - 1)
        mk = today.strftime("%Y-%m")
        if dd <= -0.20 and th.get("crash_alert_month") != mk:
            th["crash_alert_month"] = mk
            ev.append({"type": "UYARI", "ticker": "", "kind": "çöküş", "dd": dd})
    for e in ev:
        if e["type"] == "SAT":
            _append_trade(e)
    return ev


def _close(th, t, p, today, why) -> Dict:
    r = (p["level"] - 1) * 100
    rec = {"type": "SAT", "ticker": t, "reason": why, "ret_pct": round(r, 1), "peak_x": round(p["peak"], 2),
           "entry_date": p["entry_date"], "exit_date": str(pd.Timestamp(today).date()),
           "exit_px": p.get("last_close")}
    th["closed"] = (th["closed"] + [rec])[-200:]
    del th["positions"][t]
    return rec


def _append_trade(e: Dict) -> None:
    path = _cfg("TH_TRADES_FILE", os.path.join(C.DATA_DIR, "th_trades.csv"))
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        pd.DataFrame([e]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    except Exception as exc:
        print(f"⚠️ Hedef avcısı işlem kaydı yazılamadı: {exc}")


def resync(state: Dict, hist: Dict[str, pd.DataFrame], today) -> None:
    """Correct chained levels with adjusted history (missed runs, corporate actions)."""
    th = _st(state)
    for t, p in th["positions"].items():
        g = hist.get(t)
        if g is None or g.empty:
            continue
        g = g[g.index <= pd.Timestamp(today)]
        if g.empty or g.index[0] > pd.Timestamp(p["entry_date"]):
            continue                                                  # history does not reach the entry day
        e = g[g.index >= pd.Timestamp(p["entry_date"])]
        if e.empty or e.index[0] != pd.Timestamp(p["entry_date"]):
            continue
        if len(e) < 2 or not np.isfinite(e["open"].iloc[0]) or e["open"].iloc[0] <= 0:
            continue
        lv = e["close"] / e["open"].iloc[0]
        p["level"], p["peak"] = float(lv.iloc[-1]), max(1.0, float(lv.max()))
        p["last_date"] = str(e.index[-1].date())


def prices(p: Dict) -> Dict:
    """Today's-price-terms levels (robust to splits / bonus issues): price = last_close * (multiple / level)."""
    lc, lv = p.get("last_close"), p.get("level")
    if not lc or not lv or lv <= 0:
        return {}
    k = float(lc) / float(lv)                      # today's price of '1.0x' (entry price in today's terms)
    out = {"entry_now": k, "target": k * float(_cfg("TH_TARGET", 5.0))}
    tr = _cfg("TH_TRAIL", None)
    if tr:
        if p["peak"] >= tr[0]:
            out["lock"] = k * p["peak"] * (1 - tr[1])
        else:
            out["lock_trigger"] = k * tr[0]
    m = _cfg("TH_MAX_MONTHS", None)
    if m:
        out["last_day"] = pd.Timestamp(p["entry_date"]) + pd.DateOffset(months=int(m))
    return out


def guard(state: Dict, report_th: Optional[Dict], today) -> Dict:
    """Stop NEW buys (exits keep running) when the evidence says the sleeve no longer works:
       * the latest walk-forward test no longer beats random picks under the same rules, or
       * after 12+ months and 8+ trades live: losing-trade share 20pp above the test, or average trade < 0.
    A market crash alone does NOT pause buying: in the tests that made results worse (BIST and US)."""
    th = _st(state)
    why = []
    rt = report_th or th.get("backtest") or {}
    if rt.get("status") == "OK" and float(rt.get("edge_vs_random_pp") or 0) < 0:
        why.append(f"son testte rastgele seçimden kötü (fark {rt.get('edge_vs_random_pp')} puan)")
    rows = [c["ret_pct"] for c in th["closed"]] + [(p["level"] - 1) * 100 for p in th["positions"].values()]
    first = min([pd.Timestamp(c["entry_date"]) for c in th["closed"]] + [pd.Timestamp(p["entry_date"]) for p in th["positions"].values()],
                default=None)
    live = {"n": len(rows)}
    if rows:
        r = np.array(rows, float)
        live.update({"loss_rate_pct": round(float((r <= 0).mean() * 100), 1), "avg_trade_pct": round(float(r.mean()), 1)})
    if first is not None and len(rows) >= 8 and first + pd.DateOffset(months=12) <= pd.Timestamp(today):
        tl = rt.get("loss_rate_pct")
        if tl is not None and live["loss_rate_pct"] > float(tl) + 20:
            why.append(f"canlıda zararla kapanan %{live['loss_rate_pct']} (test %{tl})")
        if live["avg_trade_pct"] < 0:
            why.append(f"canlıda işlem başı ortalama %{live['avg_trade_pct']}")
    was = bool(th.get("paused"))
    th["paused"] = bool(why)
    th["pause_reasons"] = why
    th["live_trades"] = live
    return {"paused": th["paused"], "changed": was != th["paused"], "why": why}


def adopt_challenger(state: Dict, report_th: Optional[Dict], today) -> Optional[Dict]:
    """Slow, evidence-based adaptation: switch rules only if the SAME challenger qualified in
    TH_CHALLENGER_CONFIRM consecutive (new) monthly tests and the last change is TH_MIN_MONTHS_BETWEEN_CHANGES old."""
    th = _st(state)
    rt = report_th or {}
    gen = rt.get("generated_at")
    if not gen or th.get("last_report_seen") == gen:
        return None
    th["last_report_seen"] = gen
    best = rt.get("best_challenger")
    if not best:
        th["challenger_streak"] = {}
        return None
    key = str(sorted(best["change"].items()))
    st_ = th.get("challenger_streak") or {}
    st_ = {"key": key, "n": st_.get("n", 0) + 1 if st_.get("key") == key else 1, "label": best["label"]}
    th["challenger_streak"] = st_
    since = pd.Timestamp(th.get("active_since") or th.setdefault("installed", str(pd.Timestamp(today).date())))
    gap_ok = since + pd.DateOffset(months=int(_cfg("TH_MIN_MONTHS_BETWEEN_CHANGES", 12))) <= pd.Timestamp(today)
    if st_["n"] >= int(_cfg("TH_CHALLENGER_CONFIRM", 3)) and gap_ok:
        old = dict(th["active"])
        th["active"] = {**th["active"], **best["change"]}
        th["active_since"] = str(pd.Timestamp(today).date())
        th["challenger_streak"] = {}
        th.setdefault("changes", []).append({"date": th["active_since"], "from": old, "to": th["active"], "label": best["label"],
                                             "evidence": {k: best[k] for k in ("cagr_pct", "halves_cagr_pct", "edge_vs_random_pp", "loss_rate_pct")}})
        set_active(th["active"])
        return {"label": best["label"], **{k: best[k] for k in ("cagr_pct", "halves_cagr_pct", "edge_vs_random_pp", "loss_rate_pct")}}
    return None


def monthly(state: Dict, cand: pd.DataFrame, today, liq_floor: Optional[float] = None,
            report_th: Optional[Dict] = None) -> Dict:
    """Adapt (rarely), check the guard, log the market (survivorship-free) and queue buys for the free slots."""
    if not enabled():
        return {"status": "DISABLED"}
    th = _st(state)
    month = pd.Timestamp(today).strftime("%Y-%m")
    if th.get("last_month") == month:
        return {"status": "DONE"}
    if report_th:
        th["backtest"] = {k: v for k, v in report_th.items() if not str(k).startswith("_")}
    change = adopt_challenger(state, report_th, today)
    g = guard(state, report_th, today)
    r = ranked(cand, liq_floor)
    S = int(_cfg("TH_SLOTS", 8))
    taken = set(th["positions"]) | {o["ticker"] for o in th["pending"]}
    free = max(0, S - len(taken))
    buys = []
    if len(r) >= 2 * S and not g["paused"]:
        r["score_pct"] = r["th_score"].rank(pct=True) * 100
        for _, x in r.iterrows():
            if len(buys) >= free:
                break
            if x["ticker"] in taken:
                continue
            buys.append({"ticker": x["ticker"], "date": str(pd.Timestamp(today).date()), "score_pct": round(float(x["score_pct"]), 1)})
        th["pending"] += buys
    _log_universe(cand, r, today, {b["ticker"] for b in buys} | taken)
    th["last_month"] = month
    th["last_review"] = {"month": month, "n_eligible": int(len(r)), "buys": [b["ticker"] for b in buys], "free_slots": free,
                         "paused": g["paused"]}
    th["live"] = live_stats()
    return {"status": "OK", "buys": buys, "n_eligible": int(len(r)), "guard": g, "change": change}


def _log_universe(cand: pd.DataFrame, r: pd.DataFrame, today, held: set) -> None:
    """Every scanned name with its price (eligible or not) so later returns of names that drop out of the
    universe or the market are still measurable; score only for the eligible ones."""
    path = _cfg("TH_UNIVERSE_LOG", os.path.join(C.DATA_DIR, "th_universe_log.csv.gz"))
    if cand is None or cand.empty:
        return
    px = "close" if "close" in cand else ("close_adj" if "close_adj" in cand else None)
    sc = r.set_index("ticker")["th_score"] if r is not None and not r.empty else pd.Series(dtype=float)
    rec = pd.DataFrame({"tarih": str(pd.Timestamp(today).date()), "ticker": cand["ticker"].to_numpy(),
                        "score": cand["ticker"].map(sc).round(4).to_numpy(),
                        "close": pd.to_numeric(cand[px], errors="coerce").to_numpy() if px else np.nan,
                        "market_cap": _col(cand, "market_cap").to_numpy(),
                        "selected": cand["ticker"].isin(held).to_numpy()})
    try:
        old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        if not old.empty:
            old = old[old["tarih"] != rec["tarih"].iloc[0]]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        pd.concat([old, rec], ignore_index=True).to_csv(path, index=False, compression="gzip")
    except Exception as exc:
        print(f"⚠️ Hedef avcısı evren kaydı yazılamadı: {exc}")


def live_stats(min_age_months: int = 12) -> Dict:
    """Survivorship-free live check from our own monthly log: for months at least `min_age_months` old,
    the best later multiple and the latest multiple of the TOP-S names vs the whole eligible universe.
    A name that later vanished from the scan (delisted / halted) keeps its last logged price."""
    path = _cfg("TH_UNIVERSE_LOG", os.path.join(C.DATA_DIR, "th_universe_log.csv.gz"))
    if not os.path.exists(path):
        return {"months": 0}
    try:
        L = pd.read_csv(path, parse_dates=["tarih"])
    except Exception:
        return {"months": 0}
    W = L.pivot_table(index="tarih", columns="ticker", values="close", aggfunc="last").sort_index()
    months = W.index
    cut = months.max() - pd.DateOffset(months=min_age_months) if len(months) else None
    old = [m for m in months if cut is not None and m <= cut]
    if not old:
        return {"months": len(months), "resolved_months": 0}
    T = float(_cfg("TH_TARGET", 5.0))
    S = int(_cfg("TH_SLOTS", 8))
    rows = []
    for m in old:
        g = L[(L["tarih"] == m) & L["score"].notna()].sort_values("score", ascending=False)
        fut = W[W.index > m].ffill()
        if fut.empty:
            continue
        for k, (t, c0) in enumerate(zip(g["ticker"], g["close"])):
            if not np.isfinite(c0) or c0 <= 0 or t not in fut:
                continue
            path_ = fut[t] / c0
            rows.append({"top": k < S, "best": float(path_.max()), "last": float(path_.iloc[-1]),
                         "gone": bool(pd.isna(W[t].iloc[-1]))})
    if not rows:
        return {"months": len(months), "resolved_months": 0}
    D = pd.DataFrame(rows)
    f = lambda d: {"n": int(len(d)), "target_hit_pct": round(float((d["best"] >= T).mean() * 100), 1),
                   "loss_pct": round(float((d["last"] < 1).mean() * 100), 1),
                   "heavy_loss_pct": round(float((d["last"] < 0.5).mean() * 100), 1),
                   "gone_pct": round(float(d["gone"].mean() * 100), 1)}
    return {"months": len(months), "resolved_months": len(old), "top": f(D[D["top"]]), "universe": f(D)}


# ----------------------------------------------------------------------------- telegram
def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _px(v) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    s = f"{v:,.2f}" if v >= 1 else f"{v:.4f}"
    if currency() == "TL":
        return s.replace(",", "_").replace(".", ",").replace("_", ".") + " TL"
    return "$" + s


def _n(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.1f}".replace(".", ",")
    except (TypeError, ValueError):
        return "—"


def _pct(v) -> str:
    return ("+" if v >= 0 else "−") + "%" + f"{abs(v):.1f}".replace(".", ",")


WHY_TR = {"hedef": "🎯 HEDEF — kâr al", "iz": "🔒 kâr kilidi (zirveden geri çekilme)", "süre": "⏰ süre doldu",
          "stop": "🛑 stop", "veri_yok": "veri yok"}


def position_lines(th: Dict) -> List[str]:
    out = []
    for t, p in sorted(th["positions"].items(), key=lambda kv: -kv[1]["level"]):
        q = prices(p)
        line = f"• <b>{_esc(t)}</b> " + f"{p['level']:.2f}".replace(".", ",") + f"× · son {_px(p.get('last_close'))}"
        if q:
            line += f" · 🎯 hedef {_px(q['target'])}"
            if "lock" in q:
                line += f" · 🔒 kâr kilidi {_px(q['lock'])} (altına kapanırsa sat)"
            elif "lock_trigger" in q:
                line += f" · kâr kilidi {_px(q['lock_trigger'])} üstünde devreye girer"
            if "last_day" in q:
                line += f" · son gün {q['last_day']:%d.%m.%Y}"
        out.append(line)
    return out


def telegram(state: Dict, events: List[Dict], monthly_res: Optional[Dict], today) -> Optional[str]:
    if not enabled():
        return None
    th = _st(state)
    T = _cfg("TH_TARGET", 5.0)
    title = f"🏹 <b>Hedef Avcısı ({T:g}×)</b> — {pd.Timestamp(today):%d.%m.%Y}"
    lines = []
    for e in [e for e in events if e["type"] == "SAT"]:
        lines.append(f"🔴 <b>SAT {_esc(e['ticker'])}</b> (yarın açılışta) — {WHY_TR.get(e['reason'], e['reason'])} · "
                     f"kapanış {_px(e.get('exit_px'))} · getiri {_pct(e['ret_pct'])} (zirve " + f"{e['peak_x']:.2f}".replace(".", ",")
                     + f"×, giriş {pd.Timestamp(e['entry_date']):%d.%m.%Y})")
    for e in [e for e in events if e["type"] == "AL"]:
        tr = _cfg("TH_TRAIL", None)
        lines.append(f"🟢 Alındı <b>{_esc(e['ticker'])}</b> @ {_px(e['px'])} → 🎯 hedef {_px(e['px'] * T)}"
                     + (f" · kâr kilidi {_px(e['px'] * tr[0])} üstünde devreye girer" if tr else ""))
    for e in [e for e in events if e["type"] == "UYARI"]:
        if e["kind"] == "düşüş50":
            lines.append(f"⚠️ <b>{_esc(e['ticker'])} girişten %{(1 - e['level']) * 100:.0f} düştü.</b> Şirkete özel kötü bir haber "
                         "(iflas/konkordato, işlem yasağı, olumsuz denetim görüşü, büyük dava) varsa SAT. Yoksa kural: tut "
                         "(testte bu düşüşlerin çoğu sonradan toparlandı).")
        elif e["kind"] == "işlem_yok":
            lines.append(f"⚠️ <b>{_esc(e['ticker'])} 5 gündür işlem görmüyor</b> (devre kesici / işlem yasağı / borsadan çıkma). "
                         "Aracı kurumundan durumunu kontrol et.")
        elif e["kind"] == "çöküş":
            lines.append(f"⚠️ <b>Piyasa son 3 ayın zirvesinden %{abs(e['dd']) * 100:.0f} düştü.</b> Kurallar aynen devam: testte "
                         "çöküş sonrası alımı durdurmak sonucu kötüleştirdi. Sistemin kendi sağlığı bozulursa alımları ayrıca durdurur.")
    if monthly_res and monthly_res.get("status") == "OK":
        ch = monthly_res.get("change")
        if ch:
            lines.append(f"🔧 <b>Kural güncellendi:</b> {_esc(ch['label'])}. Kanıt: geçmişin iki yarısında da daha iyi "
                         f"(yıllık %{_n(ch['halves_cagr_pct'][0])} / %{_n(ch['halves_cagr_pct'][1])}), rastgeleye fark {_n(ch['edge_vs_random_pp'])} puan, "
                         f"zararla kapanan %{_n(ch['loss_rate_pct'])}. Üç ay üst üste doğrulandı.")
        g = monthly_res.get("guard") or {}
        if g.get("paused"):
            lines.append("⛔ <b>Yeni alımlar durduruldu</b> (açık pozisyonların satış kuralları sürüyor): " + "; ".join(map(_esc, g["why"])))
        elif g.get("changed"):
            lines.append("✅ Koşullar düzeldi; yeni alımlar yeniden başladı.")
        b = monthly_res.get("buys", [])
        if b:
            lines.append("🆕 <b>Yarın açılışta AL</b> (boş slot): " + ", ".join(f"{_esc(x['ticker'])} (skor %{x['score_pct']:.0f})" for x in b))
        elif not g.get("paused"):
            lines.append("Bu ay boş slot yok; yeni alım yok.")
        st_ = th.get("challenger_streak") or {}
        if st_.get("n"):
            lines.append(f"🔬 Aday kural: {_esc(st_.get('label'))} — {st_['n']}/{_cfg('TH_CHALLENGER_CONFIRM', 3)} ay doğrulandı.")
    if not lines:
        return None
    if th["positions"]:
        lines.append("<b>Açık pozisyonlar</b> (fiyatlar bugünkü fiyat cinsinden):")
        lines += position_lines(th)
    lv = th.get("live") or {}
    if lv.get("resolved_months"):
        tp = lv["top"]
        lines.append(f"📏 Canlı (yanlılıksız) kayıt: {lv['resolved_months']} ay · hedefe ulaşan %{_n(tp['target_hit_pct'])} · "
                     f"zararda %{_n(tp['loss_pct'])} · tüm uygun hisselerde hedef %{_n(lv['universe']['target_hit_pct'])}")
    lt = th.get("live_trades") or {}
    bt = th.get("backtest") or {}
    if lt.get("n"):
        lines.append(f"📊 Canlı işlemler: {lt['n']} · zararla %{_n(lt.get('loss_rate_pct'))}"
                     + (f" (test %{_n(bt.get('loss_rate_pct'))})" if bt.get("loss_rate_pct") is not None else "")
                     + f" · işlem başı ort. %{_n(lt.get('avg_trade_pct'))}"
                     + (f" (test %{_n(bt.get('avg_trade_pct'))})" if bt.get("avg_trade_pct") is not None else ""))
    if bt.get("cagr_pct") is not None:
        lines.append(f"🧪 Test: yıllık %{_n(bt['cagr_pct'])} (rastgele %{_n(bt.get('random_cagr_pct'))}) · zararla kapanan %{_n(bt.get('loss_rate_pct'))} · "
                     f"hedefe ulaşan %{_n(bt.get('target_hit_pct'))}")
    return title + "\n" + "\n".join(lines)
