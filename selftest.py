"""End-to-end self-test of V3 on SYNTHETIC data in a temporary folder.

`python main.py --self-test`. Network calls are replaced by in-memory
generators; ./data is never touched. The synthetic market has a persistent
'quality' edge (visible in fundamentals and prices), inflation, a bonus issue,
and runs ~20 months of daily sessions so monthly reviews, learning,
calibration, orders, NAV and the trade log are all exercised. Also runs unit
checks of the catastrophe stop, CPI splicing, fundamentals TTM logic and a
small walk-forward backtest.
"""
from __future__ import annotations

import os
import sys
import tempfile

if "BOQ_DATA_DIR" not in os.environ:
    os.environ["BOQ_DATA_DIR"] = tempfile.mkdtemp(prefix="boq3_selftest_")
os.environ.pop("EVDS_API_KEY", None)      # the self-test must never touch real data sources

import numpy as np
import pandas as pd

import calendar_tr as cal
import config as C


def synthetic_world(n=80, start="2019-01-02", end="2026-06-30", seed=5):
    rng = np.random.default_rng(seed)
    days = pd.DatetimeIndex(cal.sessions_between(start, end))
    T = len(days)
    q = rng.normal(0, 1, n)                                   # latent quality
    mkt = rng.normal(0.0011, 0.014, T)                        # ~30%/yr nominal market
    close = np.zeros((T, n))
    close[0] = rng.uniform(5, 150, n)
    mom = np.zeros(n)
    for t in range(1, T):
        mom = 0.995 * mom + rng.normal(0, 0.00008, n)
        r = mkt[t] * rng.uniform(0.7, 1.3, n) + 0.0005 * q + mom + rng.normal(0, 0.02, n)
        close[t] = close[t - 1] * np.exp(np.clip(r, -0.095, 0.095))
    opn = np.vstack([close[0], close[:-1] * np.exp(rng.normal(0, 0.005, (T - 1, n)))])
    vol = rng.lognormal(14, 0.6, (T, n))
    tick = [f"SYN{i:03d}" for i in range(n)]
    hist = {t: pd.DataFrame({"open": opn[:, j], "high": np.maximum(opn[:, j], close[:, j]) * 1.01,
                             "low": np.minimum(opn[:, j], close[:, j]) * 0.99, "close": close[:, j],
                             "volume": vol[:, j]}, index=days) for j, t in enumerate(tick)}
    idx = pd.Series(1000 * np.exp(np.cumsum(mkt)), index=days)
    fx = pd.Series(5 * np.exp(np.cumsum(rng.normal(0.0008, 0.005, T))), index=days)
    months = pd.date_range("2016-01-01", "2026-12-01", freq="MS")
    cpi = pd.Series(100 * np.exp(np.cumsum(np.full(len(months), 0.02) + rng.normal(0, 0.004, len(months)))), index=months)
    sectors = [f"S{j % 7}" for j in range(n)]
    gold_usd = pd.Series(1300 * np.exp(np.cumsum(rng.normal(0.0003, 0.009, T))), index=days)
    crate = pd.Series(np.full(len(months), 25.0), index=months)
    return {"days": days, "hist": hist, "idx": idx, "fx": fx, "cpi": cpi, "q": q, "tick": tick,
            "sectors": sectors, "rng": rng, "gold_usd": gold_usd, "crate": crate}


def bench_of(W):
    import benchmarks as BMk
    return BMk.assemble(W["fx"], W["gold_usd"], W["idx"])


def _run(W, **kw):
    """main.run with every external source replaced by synthetic data."""
    import main as M
    kw.setdefault("bench_fn", lambda: bench_of(W))
    kw.setdefault("cash_fn", lambda: (W["crate"], {"source": "synthetic"}))
    return M.run(**kw)


def snapshot_for(W, d, split=None):
    """TradingView-like snapshot. `split` = (ticker, date, ratio) applied to RAW prices after date."""
    rows = []
    for j, t in enumerate(W["tick"]):
        g = W["hist"][t]
        if d not in g.index:
            continue
        i = g.index.get_loc(d)
        o, c = g["open"].iloc[i], g["close"].iloc[i]
        chg = (c / g["close"].iloc[i - 1] - 1) * 100 if i > 0 else 0.0
        k = 1.0
        if split and t == split[0] and d >= split[1]:
            k = split[2]
        q = W["q"][j]
        rows.append({"ticker": t, "open": o * k, "high": g["high"].iloc[i] * k, "low": g["low"].iloc[i] * k,
                     "close": c * k, "volume": g["volume"].iloc[i], "change_pct": chg,
                     "value_traded": c * g["volume"].iloc[i], "market_cap": c * 1e8, "sector": W["sectors"][j],
                     "roe": 20 + 8 * q + np.random.normal(0, 3), "pe": max(3.0, 9 - 2 * q + np.random.normal(0, 1)),
                     "pb": max(0.3, 1.5 - 0.3 * q), "ps": 1.0, "net_income": np.nan, "revenue": np.nan,
                     "rev_growth": 35 + 10 * q, "op_margin": 12 + 4 * q, "debt_to_equity": max(0.0, 1 - 0.3 * q),
                     "div_yield": max(0.0, 2 + q)})
    return pd.DataFrame(rows)


def unit_checks() -> dict:
    import inflation as INF
    import fundamentals_hist as FH
    from portfolio import apply_day, new_portfolio
    res = {}
    # drawdown: -32% -> FLAG only (no sell); -51% from entry -> hard stop sell at next open
    pf = new_portfolio("2025-01-02")
    pf["pending"] = [{"ticker": "X", "action": "BUY", "reason": "T", "target_w": 0.1}]
    b = pd.DataFrame({"open": [10.0], "close": [10.0], "chg_pct": [0.0]}, index=["X"])
    apply_day(pf, b, "2025-01-02")
    path = [10.0, 9.0, 8.0, 7.2, 6.8]
    for i in range(1, len(path)):
        px, prev = path[i], path[i - 1]
        apply_day(pf, pd.DataFrame({"open": [px], "close": [px], "chg_pct": [(px / prev - 1) * 100]}, index=["X"]),
                  f"2025-01-0{2 + i}")
    res["drawdown_flag_not_sold"] = bool(pf["positions"]["X"].get("dd_flag")) and not pf["pending"]
    apply_day(pf, pd.DataFrame({"open": [4.9], "close": [4.9], "chg_pct": [(4.9 / 6.8 - 1) * 100]}, index=["X"]), "2025-01-08")
    # V3.8: no forced stop — a -51% name stays until its cohort expires (research design; flag is informational)
    res["no_forced_stop"] = not any(o["reason"] == "CATASTROPHE_STOP" for o in pf["pending"]) and "X" in pf["positions"]
    # flagged position with intact thesis is KEPT at the review; with weak thesis it is SOLD
    from meta_engine import plan_rebalance
    pf3 = new_portfolio("2025-01-02")
    pf3["cash"] = 0.8
    pf3["positions"] = {"A": {"value": 0.1, "cost_basis": 0.15, "entry_date": "2025-01-02", "level": 0.6, "peak": 1.0, "dd_flag": True},
                        "B": {"value": 0.1, "cost_basis": 0.15, "entry_date": "2025-01-02", "level": 0.6, "peak": 1.0, "dd_flag": True}}
    fr = pd.DataFrame({"ticker": ["A", "B", "C"], "composite_pct": [95.0, 70.0, 50.0], "composite": [2, 1, 0],
                       "med_value_traded": [1e9] * 3, "fund_break": [False] * 3, "exp_real_12m": [10.0] * 3,
                       "vol_ann_pct": [40.0] * 3, "sector": ["S1", "S2", "S3"]})
    orders, summ = plan_rebalance(pf3, fr, {"calibration": {"pct_cutoff": 85.0, "status": "CALIBRATED"}}, 1.0)
    res["dd_thesis_intact_kept"] = "A" in summ["holds"]
    res["dd_thesis_failed_sold"] = any(o["ticker"] == "B" and o["reason"] == "DRAWDOWN_CONFIRMED" for o in orders)
    # cash yield: US T-bill 5% -> 5% (no haircut, no withholding)
    import inflation as I2
    res["cash_yield"] = abs(I2.cash_yield_at(pd.Series([5.0], index=[pd.Timestamp("2025-01-01")]), "2025-03-10") - 5.0) < 1e-9
    # bonus issue neutrality: raw price halves, adjusted change +1% -> value +1%
    pf2 = new_portfolio("2025-01-02")
    pf2["pending"] = [{"ticker": "Y", "action": "BUY", "reason": "T", "target_w": 0.5}]
    apply_day(pf2, pd.DataFrame({"open": [20.0], "close": [20.0], "chg_pct": [0.0]}, index=["Y"]), "2025-01-02")
    v0 = pf2["positions"]["Y"]["value"]
    apply_day(pf2, pd.DataFrame({"open": [10.0], "close": [10.1], "chg_pct": [1.0]}, index=["Y"]), "2025-01-03")
    res["bonus_issue_neutral"] = abs(pf2["positions"]["Y"]["value"] / v0 - 1.01) < 1e-9
    # CPI splice
    a = pd.Series(np.linspace(100, 200, 40), index=pd.date_range("2020-01-01", periods=40, freq="MS"))
    bb = pd.Series(np.linspace(50, 120, 50), index=pd.date_range("2021-01-01", periods=50, freq="MS"))
    res["cpi_splice"] = len(INF.splice(a, bb)) == 62
    # SEC EDGAR point-in-time parsing: Q4 = FY - 9M, restatements ignored, TTM, availability date
    def q(s_, e, v, f, form="10-Q"):
        return {"start": s_, "end": e, "val": v, "filed": f, "form": form}
    rev = [q("2022-01-01", "2022-03-31", 100, "2022-05-05"), q("2022-04-01", "2022-06-30", 110, "2022-08-05"),
           q("2022-07-01", "2022-09-30", 120, "2022-11-05"), q("2022-01-01", "2022-09-30", 330, "2022-11-05"),
           q("2022-01-01", "2022-12-31", 460, "2023-02-25", "10-K"), q("2023-01-01", "2023-03-31", 140, "2023-05-05"),
           q("2022-01-01", "2022-03-31", 999, "2023-05-05")]
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": rev}},
                                   "NetIncomeLoss": {"units": {"USD": [dict(r_, val=r_["val"] / 10) for r_ in rev]}},
                                   "StockholdersEquity": {"units": {"USD": [{"end": "2022-12-31", "val": 1000, "filed": "2023-02-25", "form": "10-K"}]}}},
                       "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
                           {"end": "2023-02-10", "val": 50, "filed": "2023-02-25", "form": "10-K"}]}}}}}
    pit = FH.build_point_in_time(FH.parse_companyfacts("Z", facts))
    r = pit.set_index("period_end")
    res["ttm_logic"] = (abs(r.loc["2022-12-31", "revenue_ttm"] - 460) < 1e-9 and abs(r.loc["2023-03-31", "revenue_ttm"] - 500) < 1e-9
                        and abs(r.loc["2023-03-31", "net_income_ttm"] - 50) < 1e-9)
    res["pit_lag"] = FH.point_in_time(pit, "2023-05-01", ["Z"]).iloc[0]["period_end"] == pd.Timestamp("2022-12-31")
    return res


def inflation_gate_check(W) -> dict:
    """V3.8: day 1 without CPI -> the stock-only engine STILL buys (CPI is a benchmark, not a gate).
    Day 2 (same month): CPI back -> the review is redone (idempotent: same-month cohort replaced)."""
    import main as M
    import state_manager as SM
    days = [d for d in W["days"] if d >= pd.Timestamp("2025-06-02")][:2]
    reg_df = pd.DataFrame({"idx": W["idx"], "fx": W["fx"]})
    cur = {"d": days[0]}
    fetch = lambda st: (snapshot_for(W, cur["d"]), {"source": "synthetic"})
    hist_fn = lambda tickers, start: {t: g[(g.index >= pd.Timestamp(start)) & (g.index <= cur["d"])]
                                      for t, g in W["hist"].items() if t in tickers}

    def no_regime():
        raise RuntimeError("offline")
    r1 = _run(W, today=days[0], fetch=fetch, hist_fn=hist_fn, regime_fn=no_regime,
               cpi_fn=lambda: (pd.Series(dtype=float), {"status": "UNAVAILABLE"}))
    st1 = SM.load_state()
    blocked = r1["review"] is not None and any(o["action"] == "BUY" for o in st1["portfolio"]["pending"]) and \
        st1["last_rebalance"].get("expected_inflation_12m") is None
    cur["d"] = days[1]
    last = pd.Timestamp(days[1]) - pd.DateOffset(months=1)
    r2 = _run(W, today=days[1], fetch=fetch, hist_fn=hist_fn, regime_fn=lambda: reg_df[reg_df.index <= cur["d"]],
               cpi_fn=lambda: (W["cpi"][W["cpi"].index <= pd.Timestamp(last.year, last.month, 1)], {"status": "OK", "source": "synthetic"}))
    st2 = SM.load_state()
    redone = r2["review"] is not None and r2["review"].get("status") == "OK" and \
        st2["last_rebalance"].get("expected_inflation_12m") is not None
    # proxy path: CPI unavailable but FX series present
    r3 = _run(W, today=pd.Timestamp(cal.add_sessions(days[1], 1)), fetch=lambda st: (snapshot_for(W, cal.add_sessions(days[1], 1)), {}),
               hist_fn=hist_fn, regime_fn=lambda: reg_df, cpi_fn=lambda: (pd.Series(dtype=float), {"status": "UNAVAILABLE"}))
    st3 = SM.load_state()
    # US: no FX-based inflation proxy -> CPI stays unavailable (never invented), engine still runs
    proxy = st3["inflation"].get("status") != "PROXY" and st3["inflation"].get("expected_12m_pct") is None
    for f in (C.STATE_FILE, C.NAV_FILE, C.MONTHLY_SNAPSHOT_FILE, C.TRADE_LOG_FILE):
        if os.path.exists(f):
            os.remove(f)
    # intraday REFRESH: ungated review (no CPI) is redone during the session without trading
    cur["d"] = days[0]
    _run(W, today=days[0], fetch=fetch, hist_fn=hist_fn, regime_fn=no_regime,
          cpi_fn=lambda: (pd.Series(dtype=float), {"status": "UNAVAILABLE"}))
    n_nav = len(SM.load_nav())
    cur["d"] = days[1]
    rr = _run(W, today=days[1], refresh=True, fetch=fetch, hist_fn=hist_fn,
               regime_fn=lambda: reg_df[reg_df.index <= cur["d"]],
               cpi_fn=lambda: (W["cpi"][W["cpi"].index <= pd.Timestamp(last.year, last.month, 1)], {"status": "OK", "source": "synthetic"}))
    str_ = SM.load_state()
    refresh_ok = rr["review"] is not None and len(SM.load_nav()) == n_nav and \
        str_["last_rebalance"].get("expected_inflation_12m") is not None and str_["last_run"].get("mode") == "REFRESH" \
        and str_["last_rebalance"].get("date") == str(days[0].date())
    # engine upgrade after a review: the old review's unexecuted buys are cancelled and it is redone
    str_["last_rebalance"]["engine_version"] = "3.3.0"
    str_["portfolio"]["pending"] = [{"ticker": "ZZZOLD", "action": "BUY", "reason": "NEW_ENTRY", "target_w": 0.1}]
    SM.save_state(str_)
    ru = _run(W, today=days[1], refresh=True, fetch=fetch, hist_fn=hist_fn, regime_fn=lambda: reg_df[reg_df.index <= cur["d"]],
              cpi_fn=lambda: (W["cpi"][W["cpi"].index <= pd.Timestamp(last.year, last.month, 1)], {"status": "OK", "source": "synthetic"}))
    su = SM.load_state()
    upgrade_ok = ru["review"] is not None and su["last_rebalance"].get("engine_version") == C.ENGINE_VERSION and \
        not any(o["ticker"] == "ZZZOLD" for o in su["portfolio"]["pending"])
    rs = _run(W, today=days[1], refresh=True, rescan=True, fetch=fetch, hist_fn=hist_fn, regime_fn=lambda: reg_df[reg_df.index <= cur["d"]],
              cpi_fn=lambda: (W["cpi"][W["cpi"].index <= pd.Timestamp(last.year, last.month, 1)], {"status": "OK", "source": "synthetic"}))
    rescan_ok = rs["review"] is not None
    for f in (C.STATE_FILE, C.NAV_FILE, C.MONTHLY_SNAPSHOT_FILE, C.TRADE_LOG_FILE):
        if os.path.exists(f):
            os.remove(f)
    return {"no_cpi_still_buys": bool(blocked), "ungated_review_redone": bool(redone), "no_invented_cpi": bool(proxy),
            "intraday_refresh": bool(refresh_ok), "upgrade_redoes_review": bool(upgrade_ok), "manual_rescan": bool(rescan_ok)}


def _gold_ok() -> bool:
    """dual overlay rotating into gold: plan must sell stocks and buy the gold sleeve."""
    from meta_engine import plan_rebalance
    from portfolio import new_portfolio
    pf = new_portfolio("2025-01-02")
    pf["cash"] = 0.9
    pf["positions"] = {"A": {"value": 0.1, "cost_basis": 0.1, "entry_date": "2025-01-02", "level": 1.0, "peak": 1.0}}
    fr = pd.DataFrame({"ticker": ["A", "B"], "composite_pct": [95.0, 96.0], "composite": [1, 2], "med_value_traded": [1e9] * 2,
                       "fund_break": [False] * 2, "exp_real_12m": [10.0] * 2, "vol_ann_pct": [40.0] * 2, "sector": ["S1", "S2"]})
    orders, summ = plan_rebalance(pf, fr, {"calibration": {"status": "CALIBRATED"}}, 1.0,
                                  params={"n_positions": 5, "buy_pct": 90, "gate": "rank", "overlay": "dual"},
                                  equity_frac=0.0, gold_w=1.0)
    return any(o["ticker"] == "A" and o["action"] == "SELL" for o in orders) and \
        any(o["ticker"] == "ALTIN" and o["action"] == "BUY" for o in orders) and not summ["buys"]


def _tg_ok(st) -> bool:
    import telegram_report as TG
    rev = {"target_weights": {"AAA": 0.1, "DDD": 0.1, "CCC": 0.1}, "sells": ["BBB"], "sell_reasons": {"BBB": "COHORT_EXPIRY"},
           "holds": ["CCC", "DDD"], "picks": ["AAA", "DDD"], "confidence": {"AAA": 0.64, "DDD": 0.58},
           "confidence_grade": {"AAA": "Yüksek", "DDD": "Orta"}, "suggested_split": {"AAA": 0.64, "DDD": 0.36}}
    msg = TG.monthly_report(pd.Timestamp("2026-10-01"), st, rev, [{"ticker": "AAA", "exp_nominal_12m": 55.2, "p_beat_all": 0.61}])
    ev = TG.events_report(pd.Timestamp("2026-10-02"), [{"ticker": "AAA", "type": "BUY", "w": 0.1},
                                                       {"ticker": "BBB", "type": "SELL_RANK_EXIT", "ret": 12.3}], st, 0.8)
    print("\n--- TELEGRAM (aylık) ---\n" + msg + "\n--- TELEGRAM (olay) ---\n" + ev + "\n---")
    legacy = {**st, "strategy": "ADAPTIVE_BIST_REAL_RETURN_ENGINE_V3"}
    legacy.pop("active_strategy", None)
    TG.monthly_report(pd.Timestamp("2026-10-01"), legacy, rev, [])     # old string key must not crash
    stt = TG.status_report(pd.Timestamp("2026-10-02"), legacy, True)
    print("--- TELEGRAM (durum) ---\n" + stt + "\n---")
    return any(w in msg for w in ("Çıta", "çıta", "hedef"))


def _confidence_ok() -> bool:
    """Calibrated confidence: fit recovers a monotone relation and predictions stay inside (0,1)."""
    import confidence as CF
    rng = np.random.default_rng(3)
    rows = []
    for m in range(40):
        d = pd.Timestamp("2015-01-01") + pd.DateOffset(months=m)
        n = 120
        pctv = rng.uniform(0, 100, n)
        ret = (pctv - 50) * 0.4 + rng.normal(0, 30, n)
        rows.append(pd.DataFrame({"tarih": d, "ticker": [f"T{i}" for i in range(n)], "composite_pct": pctv,
                                  "fwd_ret": ret, "max_1m": rng.uniform(0, 10, n),
                                  **{k: rng.normal(size=n) for k in CF.KEY_FACTORS}}))
    R = pd.concat(rows, ignore_index=True)
    m = CF.walk_forward(R, 2017, 2018)
    hi = CF.predict(pd.DataFrame({"composite_pct": [99.0], "max_1m": [1.0], **{k: [1.0] for k in CF.KEY_FACTORS}}), m).iloc[0]
    lo = CF.predict(pd.DataFrame({"composite_pct": [1.0], "max_1m": [1.0], **{k: [-1.0] for k in CF.KEY_FACTORS}}), m).iloc[0]
    return bool(m["coef"]["pct"] + m["coef"]["pct3"] > 0 and 0 < lo < 0.5 < hi < 1 and m.get("oos_reliability"))


def _health_units_ok() -> bool:
    import health as HL
    import confidence as CF
    idx = pd.bdate_range("2018-01-01", "2024-01-01")
    rng = np.random.default_rng(5)
    bt = pd.DataFrame({"tarih": idx, "nav": np.cumprod(1 + rng.normal(0.0015, 0.015, len(idx))),
                       "xu100": np.cumprod(1 + rng.normal(0.001, 0.013, len(idx)))})
    live = pd.DataFrame({"tarih": pd.bdate_range("2025-01-01", periods=130), "nav": np.linspace(1, 1.10, 130),
                         "xu100": np.linspace(100, 104, 130)})
    lv = HL.live_vs_test(live, bt)
    bad = live.assign(nav=np.linspace(1, 0.4, 130))
    lvb = HL.live_vs_test(bad, bt)
    rows = []
    for m in range(24):
        n = 120
        pctv = rng.uniform(0, 100, n)
        rows.append(pd.DataFrame({"tarih": pd.Timestamp("2025-01-01") + pd.DateOffset(months=m), "ticker": [f"T{i}" for i in range(n)],
                                  "composite_pct": pctv, "fwd_ret": (pctv - 50) * 0.5 + rng.normal(0, 30, n), "confidence": 0.5,
                                  "max_1m": rng.uniform(0, 9, n), **{k: rng.normal(size=n) for k in CF.KEY_FACTORS}}))
    model, meta = CF.live_update(None, pd.concat(rows, ignore_index=True))
    return (lv.get("status") == "OK" and 0 <= lv["percentile"] <= 100 and lvb["percentile"] < lv["percentile"]
            and meta["live_weight"] > 0 and model["source"] == "backtest+live_blend")


def _liq_ok() -> bool:
    import backtest_optimizer as B
    cpi = pd.Series([100.0, 200.0, 400.0], index=pd.to_datetime(["2015-01-01", "2020-01-01", "2025-01-01"]))
    f15, f25 = B.liq_floor_at(cpi, "2015-01-15"), B.liq_floor_at(cpi, "2025-01-15")
    return abs(f15 - C.MIN_MEDIAN_VALUE_TRADED_TL / 4) < 1 and abs(f25 - C.MIN_MEDIAN_VALUE_TRADED_TL) < 1


def _sum_hurdle_ok() -> bool:
    import benchmarks as BM
    if getattr(C, "HURDLE_MODE", "max") != "sum":
        return True
    idx = pd.bdate_range("2020-01-01", "2022-01-01")
    bm = pd.DataFrame({"usdtry": np.linspace(10, 12, len(idx)), "gold_try": np.linspace(100, 125, len(idx)),
                       "xu100": np.linspace(1, 2, len(idx))}, index=idx)
    cpi = pd.Series(np.linspace(100, 140, 25), index=pd.date_range("2020-01-01", periods=25, freq="MS"))
    w = BM.window_returns(bm, cpi, None, idx[0], idx[260])
    comps = [w[k] for k in C.HURDLE_COMPONENTS if np.isfinite(w.get(k, np.nan))]
    e = BM.expected_hurdles(bm, {"expected_12m_pct": 30.0}, 29.8, as_of=idx[-1])
    return (abs(w["hurdle"] - (sum(comps) + C.MIN_EDGE_OVER_HURDLE_PCT)) < 1e-9 and w["floor"] == max(comps)
            and abs(e["hurdle"] - round(30.0 + e["gold"] + 29.8 + C.MIN_EDGE_OVER_HURDLE_PCT, 2)) < 0.02
            and e["mode"] == "sum")


def run_self_test() -> bool:
    import main as M
    import autonomy_guard as AG
    assert os.path.abspath(C.DATA_DIR) != os.path.abspath("data"), "self-test must not use ./data"
    W = synthetic_world()
    gate = inflation_gate_check(W)
    split = ("SYN003", pd.Timestamp("2025-03-03"), 0.5)
    live_days = [d for d in W["days"] if pd.Timestamp("2024-11-01") <= d <= pd.Timestamp("2026-05-29")]
    reg_df = pd.DataFrame({"idx": W["idx"], "fx": W["fx"]})
    cur = {"d": None}

    def fetch(state):
        return snapshot_for(W, cur["d"], split), {"source": "synthetic"}

    def hist_fn(tickers, start):
        return {t: g[(g.index >= pd.Timestamp(start)) & (g.index <= cur["d"])] for t, g in W["hist"].items() if t in tickers}

    def cpi_fn():
        last = pd.Timestamp(cur["d"]) - pd.DateOffset(months=1)
        s = W["cpi"][W["cpi"].index <= pd.Timestamp(last.year, last.month, 1)]
        return s, {"source": "synthetic", "status": "OK"}

    def regime_fn():
        return reg_df[reg_df.index <= cur["d"]]

    statuses, reviews = [], 0
    for d in live_days:
        cur["d"] = d
        r = _run(W, today=d, fetch=fetch, hist_fn=hist_fn, cpi_fn=cpi_fn, regime_fn=regime_fn)
        statuses.append(r["status"])
        reviews += int(bool(r.get("review")))
    # stale detection: re-run same data on next calendar session
    r2 = _run(W, today=cal.add_sessions(live_days[-1], 1), fetch=fetch, hist_fn=hist_fn, cpi_fn=cpi_fn, regime_fn=regime_fn)

    from state_manager import load_monthly_snapshots, load_nav, load_state, load_trade_log
    st, snaps, nav, tl = load_state(), load_monthly_snapshots(), load_nav(), load_trade_log()
    u = unit_checks()

    # small walk-forward backtest on the same synthetic history (no fundamentals history)
    import backtest_optimizer as B
    sub = {t: g[g.index <= pd.Timestamp("2026-05-29")] for t, g in list(W["hist"].items())[:70]}
    bt = B.run("2019-01-01", None, save=True, data=sub, regime_df=reg_df, cpi=W["cpi"], pit=pd.DataFrame(),
               bm=bench_of(W), crate=W["crate"])
    rep = bt["report"]

    checks = {
        "all_runs_ok": all(s == "OK" for s in statuses),
        "stale_detected": r2["status"] == "STALE",
        "monthly_reviews": reviews >= 18,
        "positions_held": len(st["portfolio"]["positions"]) > 0,
        "nav_recorded": len(nav) >= len(live_days) - 1,
        "labels_resolved": "fwd_3m" in snaps and snaps["fwd_3m"].notna().sum() > 0,
        "learning_ran": not str(st["model"].get("status", "")).startswith("BOOTSTRAP"),
        "guard_self_test": AG.run_self_test()["passed"],
        "real_return_reported": st["performance"]["nav"].get("real_total_pct") is not None,
        "backtest_ran": rep["portfolio"].get("days", 0) > 200,
        "hurdles_computed": (st.get("hurdles") or {}).get("hurdle") is not None and (st.get("hurdles") or {}).get("gold") is not None,
        "beat_all_labels": "beat_all" in snaps and snaps["beat_all"].notna().sum() > 0,
        "multi_bench_report": "rolling12m_beat" in rep["portfolio"] and "beat_all" in next(iter(rep["per_year"].values())),
        "tranche_backtest": (rep.get("rules") or {}).get("engine") == "stock-only monthly cohorts"
                            and bool((rep.get("confidence_model") or {}).get("coef")),
        "gold_sleeve_unit": _gold_ok(),
        "telegram_monthly_ok": _tg_ok(st),
        "hurdle_is_sum": _sum_hurdle_ok(),
        "confidence_unit": _confidence_ok(),
        "health_units": _health_units_ok(),
        "health_report": (st.get("health") or {}).get("overall") in ("OK", "UYARI", "KRİTİK")
                         and len((st.get("health") or {}).get("checks", [])) >= 10,
        "backtest_nav_saved": os.path.exists(C.BACKTEST_NAV_FILE),
        "liq_floor_scaled": _liq_ok(),
        "loser_extension_reported": "extended" in (st.get("last_rebalance") or {}),
        "cohorts_live": bool(st["portfolio"].get("cohorts")) and all(len(c["tickers"]) <= 2 * C.TRANCHE_N for c in st["portfolio"]["cohorts"])
                        and len(st["portfolio"]["cohorts"]) <= C.TRANCHE_MONTHS,
        "stock_only_fully_invested": st["portfolio"]["cash"] / max(st["portfolio"]["nav"], 1e-9) < 0.08,
        "picks_have_confidence": all(0 < x["confidence"] < 1 for x in (st.get("last_picks") or {}).get("picks", [{"confidence": 0}])),
        **{f"unit_{k}": bool(v) for k, v in u.items()},
        **{f"gate_{k}": v for k, v in gate.items()},
    }
    print("SELF-TEST checks:", checks)
    print("live performance:", st["performance"]["nav"])
    print("lots:", st["performance"]["lots"])
    print("model:", st["model"].get("status"), "| calibration:", st["calibration"].get("status"))
    print("backtest:", rep["portfolio"], rep["oos_composite_ic_12m"])
    ok = all(checks.values())
    print("✅ SELF-TEST PASSED" if ok else "❌ SELF-TEST FAILED", "| temp:", C.DATA_DIR)
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_self_test() else 1)
