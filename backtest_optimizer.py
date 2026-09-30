"""Walk-forward research + backtest of the V3.8 stock-only engine (real data, free sources).

* Adjusted daily OHLCV (yfinance) for EVERY US small cap TradingView lists (point-in-time liquidity floor),
  point-in-time fundamentals (Is Yatirim, 75/100-day publication lags), CPI with a 1-month lag.
* Factor weights are learned walk-forward (yearly folds, 13-month purge) — never on the test year.
* The portfolio is simulated with the SAME functions the live engine uses
  (meta_engine.plan_tranche + portfolio.apply_day; orders fill at the next session's open).
* Rules are fixed in config.py (no strategy search). A calibrated confidence model is fitted on
  out-of-sample scores only (confidence.walk_forward) and published in the research prior.
Known optimism: survivorship (delisted names are missing), a few design parameters were chosen on this
sample -> treat the headline CAGR as an upper estimate.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config as C
import fundamentals_hist as FH
import inflation as INF
import market_data as MD
import regime_model as RM
import benchmarks as BM
from backtest_validator import enrich_lots, lot_metrics, nav_metrics
from calibration import add_excess, bucket_table, calibrate, cutoff_stats, market_stats
from factors import build_frame, composite, price_factors_at, wide_from_history
from labels import forward_labels, month_start_sessions
from learner_engine import daily_rank_ic, factor_corr, fit_weights, get_prior, newey_west
import confidence as CF
from meta_engine import plan_tranche, score_universe
from portfolio import apply_day, new_portfolio, weights as pf_weights
from state_manager import atomic_json_write, load_state

UNIVERSE = [   # fallback only (used when the TradingView list cannot be fetched): long-listed US small/mid caps
    "AAON", "ABM", "ACIW", "AEIS", "ALE", "ALKS", "AMKR", "ANF", "APOG", "ARCB", "ASGN", "AVA", "AVNT", "AWR", "AX",
    "BCPC", "BDC", "BHE", "BKH", "BLKB", "BMI", "BOOT", "BRC", "CABO", "CALM", "CATY", "CBU", "CENTA", "CNMD", "COHU",
    "CRS", "CSGS", "CVBF", "CVLT", "CW", "DIOD", "DORM", "ENS", "EPC", "ESE", "EXLS", "EXPO", "FELE", "FFIN", "FIZZ",
    "FORM", "FSS", "FUL", "GBCI", "GEF", "GHC", "HELE", "HI", "HLIO", "HNI", "HUBG", "IBOC", "IDCC", "IOSP", "ITRI",
    "JBSS", "JJSF", "KAI", "KALU", "KFY", "KLIC", "KWR", "LANC", "LCII", "LXP", "MATX", "MGEE", "MLI", "MMSI", "MOG.A",
    "MTX", "MYRG", "NHC", "NJR", "NPO", "NSIT", "NWE", "NWN", "OFG", "OSIS", "OTTR", "PATK", "PLXS", "PLUS", "POWI",
    "PRGS", "PRK", "RLI", "ROG", "SAFT", "SANM", "SCSC", "SFNC", "SJW", "SKYW", "SMP", "SPSC", "SR", "STRA", "SXI",
    "TDS", "TNC", "TRMK", "UFPI", "UNF", "UTL", "VIAV", "WDFC", "WERN", "WTS",
]
MIN_TRAIN_MONTHS = 36


def backtest_universe() -> List[str]:
    """V3.7: every BIST stock TradingView lists today (+ the classic large caps as a safety net).
    Survivorship remains (delisted names are missing) but the universe is no longer 103 large caps:
    mid and small caps — where most of BIST's big winners come from — are tested too.
    Point-in-time liquidity is enforced month by month (CPI-scaled floor, see liq_floor_at)."""
    try:
        live = MD.list_all_tickers(C.BACKTEST_UNIVERSE_MAX)
    except Exception as exc:
        print(f"⚠️ Tam hisse listesi alınamadı ({exc}); sabit yedek liste kullanılıyor.")
        live = []
    uni = list(dict.fromkeys(live + UNIVERSE))[: max(C.BACKTEST_UNIVERSE_MAX, len(UNIVERSE))]
    print(f"🌐 Backtest evreni: {len(uni)} hisse (TradingView listesi {len(live)})")
    return uni


def _downtrend(index_close, day) -> bool:
    if index_close is None or not len(index_close):
        return False
    s = index_close[index_close.index <= pd.Timestamp(day)].dropna()
    return bool(len(s) >= 200 and s.iloc[-1] < s.iloc[-200:].mean())


def liq_floor_at(cpi: Optional[pd.Series], day) -> float:
    """Minimum median daily value traded in THAT month's lira: today's floor deflated by CPI.
    (A fixed 20M TL floor would wrongly exclude almost every mid cap in 2013-2019.)"""
    if cpi is None or cpi.empty:
        return C.MIN_MEDIAN_VALUE_TRADED_TL
    r = INF.cpi_ratio(cpi, day, cpi.index[-1])
    if not np.isfinite(r) or r <= 0:
        return C.MIN_MEDIAN_VALUE_TRADED_TL
    return float(C.MIN_MEDIAN_VALUE_TRADED_TL / max(r, 1.0))


def fund_inputs_at(pit: pd.DataFrame, date, price_f: pd.DataFrame, latest_paid: Dict[str, float]) -> pd.DataFrame:
    if pit is None or pit.empty or price_f.empty:
        return pd.DataFrame(columns=["ticker"])
    d = FH.point_in_time(pit, date, price_f["ticker"].tolist())
    if d.empty:
        return pd.DataFrame(columns=["ticker"])
    px = price_f.set_index("ticker")["close_adj"]
    out = pd.DataFrame({"ticker": d["ticker"].to_numpy()})
    eq = d["equity"].to_numpy(float)
    ni = d["net_income_ttm"].to_numpy(float)
    rev = d["revenue_ttm"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = out["ticker"].map(latest_paid).to_numpy(float)
        out["market_cap"] = out["ticker"].map(px).to_numpy(float) * shares
        out["net_income"] = ni
        out["equity"] = eq
        out["revenue"] = rev
        out["roe"] = np.where(eq > 0, ni / eq * 100.0, np.nan)
        out["op_margin"] = np.where(rev > 0, d["op_profit_ttm"].to_numpy(float) / rev * 100.0, np.nan)
        out["debt_to_equity"] = np.where(eq > 0, d["fin_debt"].to_numpy(float) / eq, np.nan)
        out["rev_growth"] = d["rev_growth_pct"].to_numpy(float)
    return out.replace([np.inf, -np.inf], np.nan)


def cpi_stats_at(cpi: pd.Series, date, proxy: bool = False) -> Dict:
    if cpi is None or cpi.empty:
        return {"yoy_pct": None, "expected_12m_pct": None}
    last_pub = pd.Timestamp(date) - pd.DateOffset(months=1)        # ~1 month publication lag
    s = cpi[cpi.index <= pd.Timestamp(last_pub.year, last_pub.month, 1)]
    return INF.inflation_stats(s, proxy=proxy)


def run(start: str = "2010-01-01", end: Optional[str] = None, save: bool = True,
        data: Optional[Dict] = None, regime_df: Optional[pd.DataFrame] = None,
        cpi: Optional[pd.Series] = None, pit: Optional[pd.DataFrame] = None,
        bm: Optional[pd.DataFrame] = None, crate: Optional[pd.Series] = None,
        variants: Optional[List[Dict]] = None) -> Dict:
    state = load_state()
    sector_map = state.get("sector_map", {})
    data = data if data is not None else MD.download_history(backtest_universe(), start, end, min_rows=300)
    if len(data) < C.MIN_CROSS_SECTION:
        raise RuntimeError(f"Gerçek veri yetersiz: {len(data)} hisse")
    rdf = None
    try:
        rdf = regime_df if regime_df is not None else RM.download_regime_series(start=start, end=end)
        index_close, X = rdf["idx"], RM.make_features(rdf)
    except Exception as exc:
        print(f"⚠️ Endeks/rejim serisi yok: {exc}")
        index_close, X = None, None
    cpi_meta = {"source": "given"}
    if cpi is None:
        cpi, cpi_meta = INF.load_cpi_or_proxy(fx=rdf["fx"] if rdf is not None else None)
        if cpi_meta.get("status") == "PROXY":
            print("⚠️ Resmî TÜFE alınamadı; USDTRY vekili kullanılıyor (EVDS_API_KEY ekleyin).")
    cr_meta = {"source": "given"}
    if crate is None:
        crate, cr_meta = INF.load_cash_rate()
    if bm is None:
        try:
            bm = BM.download_benchmarks(start=str(int(start[:4]) - 1) + "-01-01", end=end)
        except Exception as exc:
            print(f"⚠️ Kıyas serileri alınamadı ({exc}); USDTRY rejim serisinden, altın yok.")
            bm = BM.assemble(rdf["fx"], None, rdf["idx"]) if rdf is not None else None
    bench = {"bm": bm, "crate": crate}
    fund_cov = 0.0
    if pit is None:
        try:
            pit = FH.load_history(list(data.keys()), int(start[:4]))
        except Exception as exc:
            print(f"⚠️ Temel veri geçmişi alınamadı: {exc}")
            pit = pd.DataFrame()
    latest_paid = {}
    if pit is not None and not pit.empty and "paid_in" in pit:
        lp = pit.dropna(subset=["paid_in"]).sort_values("period_end").groupby("ticker")["paid_in"].last()
        latest_paid = lp.to_dict()

    wide = wide_from_history(data)
    idx = wide["close"].index
    reb = month_start_sessions(idx[C.MIN_HISTORY_SESSIONS:])
    inputs, zrows = {}, []
    for d in reb:
        pf_ = price_factors_at(wide, index_close, d)
        if len(pf_) < C.MIN_CROSS_SECTION:
            continue
        fi = fund_inputs_at(pit, d, pf_, latest_paid)
        cs = cpi_stats_at(cpi, d, proxy=cpi_meta.get("status") == "PROXY")
        inputs[d] = (pf_, fi, cs)
        fr, cov = build_frame(pf_, fi, cs.get("yoy_pct"), sector_map)
        fund_cov = max(fund_cov, float(np.mean([cov.get(k, 0) for k in C.FUNDAMENTAL_FACTORS])))
        zrows.append(fr[["tarih", "ticker"] + [f"z_{k}" for k in C.FACTORS]])
    dates = sorted(inputs.keys())
    if len(dates) < MIN_TRAIN_MONTHS + 14:
        raise RuntimeError(f"Walk-forward için yetersiz ay: {len(dates)}")
    Z = pd.concat(zrows, ignore_index=True)
    lab = forward_labels(wide, dates, cpi, index_close, bm, crate)
    ds = Z.merge(lab, on=["tarih", "ticker"], how="inner")
    hand = get_prior(None)

    # ------------------------------------------------ walk-forward folds (yearly)
    first_test_i = MIN_TRAIN_MONTHS + 13
    test_starts = dates[first_test_i::12]
    folds, oos_rows = [], []
    fold_of_date = {}
    for k, T0 in enumerate(test_starts):
        T1 = test_starts[k + 1] if k + 1 < len(test_starts) else None
        train_dates = [d for d in dates if d + pd.DateOffset(months=13) <= T0]
        w, meta = fit_weights(ds[ds["tarih"].isin(train_dates)], hand)
        params = None
        if X is not None and len(X[X.index < T0]) > 500:
            params = RM.fit_hmm(X[X.index < T0])
        for d in dates:
            if d >= T0 and (T1 is None or d < T1):
                fold_of_date[d] = (k, w, params)
        folds.append({"test_start": str(T0.date()), "train_months": len(train_dates), "weights": w,
                      "factor_ic_train": {f: meta["factor_stats"][f]["ic_mean"] for f in C.FACTORS}})

    # ---------------- scoring pass: one scored cross-section per rebalance date (strategy-independent)
    alpha_cache = {}
    sim_days = idx[(idx >= dates[first_test_i])]
    O, Cl = wide["open"], wide["close"]
    chg = Cl.pct_change(fill_method=None) * 100.0
    cal_state = {"calibration": {}}
    frames = {}
    for day in sorted(fold_of_date.keys()):
        k, w, params = fold_of_date[day]
        if params is not None:
            if k not in alpha_cache:
                alpha_cache[k] = RM.filtered_probs(params, X)
            pos = X.index.searchsorted(day, side="right") - 1
            reg = RM.summarize(params, alpha_cache[k][: pos + 1], X.index[: pos + 1])
            reg["degraded"] = False
        else:
            reg = {"label": "UNKNOWN", "probs": {}, "p_risk_off": 0.5, "degraded": True, "exp_mkt_12m_pct": None}
        known = [r for r in oos_rows if r["tarih"].iloc[0] + pd.DateOffset(months=13) <= day]
        if known:
            kd = pd.concat(known, ignore_index=True).merge(lab, on=["tarih", "ticker"], how="inner")
            calibrate(cal_state, kd, None)
        pf_, fi, cs = inputs[day]
        st = {"model": {"champion_weights": w, "champion_version": f"wf{k}", "regime_weights": {}},
              "calibration": dict(cal_state["calibration"]), "regime": reg,
              "_no_inflation": cs.get("expected_12m_pct") is None, "liq_floor_tl": liq_floor_at(cpi, day),
              "downtrend": _downtrend(index_close, day),
              "hurdles": BM.expected_hurdles(bm, cs, INF.cash_yield_at(crate, day) if len(crate) else None, as_of=day)}
        frame, info = score_universe(pf_, fi, st, cs, sector_map)
        if frame.empty:
            continue
        keep_cols = ["tarih", "ticker", "composite", "composite_pct", "regime_label", "med_value_traded", "max_1m"] + \
            [c for c in CF.KEY_FACTORS if c in frame.columns]
        oos_rows.append(frame[[c for c in keep_cols if c in frame.columns]].assign(tarih=day))
        frames[day] = (frame, st, reg)

    # ---------------- V3.8: stock-only monthly-cohort simulation — SAME code path as live
    # (meta_engine.plan_tranche + portfolio.apply_day). No strategy search: the rules are fixed
    # in config.py (TRANCHE_N / TRANCHE_MONTHS / TRANCHE_SECTOR_CAP), so nothing is tuned on the test years.
    print(f"🧪 Simülasyon: her ay en iyi {C.TRANCHE_N} hisse, her dilim {C.TRANCHE_MONTHS} ay, %100 hisse")
    pf = new_portfolio(sim_days[0])
    nav_rows, lots_all, n_names = [], [], []
    for day in sim_days:
        bars = pd.DataFrame({"open": O.loc[day], "close": Cl.loc[day], "chg_pct": chg.loc[day]}).dropna()
        _, lots = apply_day(pf, bars, day, cash_yield_pct=INF.cash_yield_at(crate, day) if len(crate) else None)
        lots_all.extend(lots)
        if day in frames:
            frame, st, reg = frames[day]
            orders, summ = plan_tranche(pf, frame, st, day, today_change=bars["chg_pct"])
            pf["pending"] = orders
            n_names.append(len(summ["target_weights"]))
        wts = pf_weights(pf)
        nav_rows.append({"tarih": day, "nav": pf["nav"], "exposure": round(sum(wts.values()), 4),
                         "xu100": float(index_close.asof(day)) if index_close is not None else np.nan})
    nav_df = pd.DataFrame(nav_rows)
    lots_df = enrich_lots(pd.DataFrame(lots_all), cpi, index_close, bench)
    open_now = {t: round((p["level"] - 1) * 100, 2) for t, p in pf["positions"].items()}

    # ---------------- calibrated confidence model (walk-forward on OUT-OF-SAMPLE scores, investable names)
    conf_model = dict(CF.DEFAULT_MODEL)
    try:
        oo = pd.concat(oos_rows, ignore_index=True).merge(lab[["tarih", "ticker", "fwd_ret"]], on=["tarih", "ticker"], how="inner")
        oo = oo[pd.to_numeric(oo["med_value_traded"], errors="coerce") >= oo["tarih"].map(lambda d: liq_floor_at(cpi, d))]
        y0 = pd.Timestamp(oo["tarih"].min()).year + 2
        conf_model = CF.walk_forward(oo.dropna(subset=["fwd_ret"]), y0, pd.Timestamp(idx[-1]).year)
    except Exception as exc:
        print(f"⚠️ Güven modeli kurulamadı ({exc}); araştırma varsayılanı kullanılıyor.")

    # OOS IC of the composite (12m) per fold
    oos = pd.concat(oos_rows, ignore_index=True) if oos_rows else pd.DataFrame()
    oos_l = oos.merge(lab, on=["tarih", "ticker"], how="inner") if not oos.empty else pd.DataFrame()
    ic12 = daily_rank_ic(oos_l.dropna(subset=["fwd_ret"]), ["composite"], "fwd_ret") if not oos_l.empty else pd.DataFrame()
    m, se, t, n = newey_west(ic12["composite"], 11) if not ic12.empty else (0.0, 0, 0.0, 0)

    # per-year table
    per_year = {}
    if not nav_df.empty:
        y = nav_df.set_index("tarih")
        prev_end = None
        for yr, g in y.groupby(y.index.year):
            g0 = y.loc[[prev_end]] if prev_end is not None else g.iloc[[0]]
            prev_end = g.index[-1]
            if len(g) < 20:
                continue
            a0 = g0.index[0]
            nom = (g["nav"].iloc[-1] / g0["nav"].iloc[0] - 1) * 100
            c = INF.cpi_ratio(cpi, a0, g.index[-1])
            xr = (g["xu100"].iloc[-1] / g0["xu100"].iloc[0] - 1) * 100 if g["xu100"].notna().all() and np.isfinite(g0["xu100"].iloc[0]) else np.nan
            per_year[int(yr)] = {"nominal_pct": round(float(nom), 2),
                                 "cpi_pct": round(float((c - 1) * 100), 2) if np.isfinite(c) else None,
                                 "real_pct": round(float(((1 + nom / 100) / c - 1) * 100), 2) if np.isfinite(c) else None,
                                 "xu100_pct": round(float(xr), 2) if np.isfinite(xr) else None}
            w = BM.window_returns(bm, cpi, crate, a0, g.index[-1])
            per_year[int(yr)].update({"usd_pct": None if not np.isfinite(w["usd"]) else round(float(w["usd"]), 2),
                                      "gold_pct": None if not np.isfinite(w["gold"]) else round(float(w["gold"]), 2),
                                      "deposit_pct": None if not np.isfinite(w["deposit"]) else round(float(w["deposit"]), 2),
                                      "hurdle_pct": None if not np.isfinite(w["hurdle"]) else round(float(w["hurdle"]), 2),
                                      "beat_each": None if not np.isfinite(w.get("floor", np.nan)) else bool(nom > w["floor"]),
                                      "beat_all": None if not np.isfinite(w["hurdle"]) else bool(nom > w["hurdle"])})

    prior = build_research_prior(ds, X, oos_l)
    prior[C.CONFIDENCE_FILE_KEY] = conf_model
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z", "engine_version": C.ENGINE_VERSION,
        "hurdle_mode": getattr(C, "HURDLE_MODE", "max"), "hurdle_edge_pct": C.MIN_EDGE_OVER_HURDLE_PCT,
        "objective": {"horizon_months": C.HORIZON_MONTHS,
                      "primary": "beat ALL of " + ", ".join(C.HURDLE_COMPONENTS) + f" (USD incl. {C.US_INFLATION_PCT}% US inflation)",
                      "secondary": "excess vs IWM (Russell 2000)"},
        "period": f"{str(pd.Timestamp(dates[first_test_i]).date())}..{str(pd.Timestamp(idx[-1]).date())}",
        "universe_downloaded": len(data), "rebalance_months": len(dates),
        "cpi_source": cpi_meta, "cash_rate_source": cr_meta, "fundamentals_coverage": round(fund_cov, 3),
        "data_source": "Yahoo Finance adjusted OHLCV + Is Yatirim statements + CPI (EVDS/FRED)",
        "synthetic_data_used": False, "cost_round_trip_pct": C.COST_ROUND_TRIP_PCT,
        "rules": {"engine": "stock-only monthly cohorts", "picks_per_month": C.TRANCHE_N,
                  "cohort_months": C.TRANCHE_MONTHS, "sector_cap_per_cohort": C.TRANCHE_SECTOR_CAP,
                  "max_name_weight": C.MAX_NAME_W, "rebalance_band": C.REBALANCE_BAND,
                  "liquidity_floor_today_tl": C.MIN_MEDIAN_VALUE_TRADED_TL},
        "portfolio": nav_metrics(nav_df, cpi, bench),
        "portfolio_basis": "walk-forward scores, fixed rules (no strategy search), same code path as live",
        "avg_names_held": round(float(np.mean(n_names)), 1) if n_names else None,
        "closed_lots": lot_metrics(lots_df),
        "confidence_model": conf_model,
        "open_positions_end": open_now,
        "oos_composite_ic_12m": {"mean": round(float(m), 4), "t_nw": round(float(t), 2), "n_months": int(n)},
        "per_year": per_year,
        "folds": folds,
        "factor_ic_full_sample_12m": prior.get("ic_mean"),
        "factor_t_full_sample_12m": prior.get("ic_t_nw"),
        "limitations": ["survivorship: universe = today's liquid names",
                        "dividends are in adjusted prices (reinvested); idle cash earns TCMB funding rate - 2pp, after 15% tax"
                        if len(crate) else "dividends are in adjusted prices (reinvested); idle cash earns 0 (rate series unavailable)",
                        f"fundamentals coverage {round(fund_cov, 2)} (Is Yatirim best effort)",
                        "autonomy guard neutral in backtest"],
    }
    print(json.dumps({k: report[k] for k in ("period", "portfolio", "avg_names_held", "oos_composite_ic_12m", "per_year")},
                     ensure_ascii=False, indent=2, default=str))
    if save:
        atomic_json_write(C.RESEARCH_PRIOR_FILE, prior)
        atomic_json_write(C.BACKTEST_REPORT_FILE, report)
        try:
            nav_df[["tarih", "nav", "xu100"]].to_csv(C.BACKTEST_NAV_FILE, index=False, float_format="%.6g")
        except Exception as exc:
            print(f"⚠️ backtest NAV yazılamadı: {exc}")
        if os.path.exists(C.STRATEGY_CONFIG_FILE):
            os.remove(C.STRATEGY_CONFIG_FILE)          # V3.8: no strategy switching
    return {"report": report, "prior": prior, "nav": nav_df, "lots": lots_df}


def build_research_prior(ds: pd.DataFrame, X: Optional[pd.DataFrame], oos_l: pd.DataFrame) -> Dict:
    zc = [f"z_{k}" for k in C.FACTORS]
    d = ds.dropna(subset=["fwd_ret"])
    ic = daily_rank_ic(d, zc, "fwd_ret")
    ic_mean, ic_t = {}, {}
    for k in C.FACTORS:
        s = ic.get(f"z_{k}", pd.Series(dtype=float)).dropna()
        mm, se, tt, n = newey_west(s, 11) if len(s) else (0.0, 0, 0.0, 0)
        ic_mean[k], ic_t[k] = round(float(mm), 5), round(float(tt), 3)
    om = factor_corr(d)
    prior = {"generated_at": datetime.utcnow().strftime("%Y-%m-%d"), "horizon_months": C.HORIZON_MONTHS,
             "ic_mean": ic_mean, "ic_t_nw": ic_t, "omega": np.round(om, 4).tolist() if om is not None else None,
             "n_dates": int(len(ic)), "n_eff_dates": round(len(ic) / C.LABEL_HORIZON, 1), "regime_ic": {}}
    if X is not None and len(X) > 500:
        params = RM.fit_hmm(X)
        a = RM.filtered_probs(params, X)
        labs = pd.Series([params["labels"][int(i)] for i in a.argmax(axis=1)], index=X.index)
        d2 = d.copy()
        d2["regime_label"] = d2["tarih"].map(lambda t: labs.asof(t) if t >= labs.index[0] else None)
        for L, sub in d2.dropna(subset=["regime_label"]).groupby("regime_label"):
            icr = daily_rank_ic(sub, zc, "fwd_ret")
            if len(icr) < 24:
                continue
            prior["regime_ic"][L] = {"ic_mean": {k: round(float(icr[f"z_{k}"].mean()), 5) if icr[f"z_{k}"].notna().any() else 0.0
                                                 for k in C.FACTORS},
                                     "n_dates": int(len(icr)), "n_eff_dates": round(len(icr) / C.LABEL_HORIZON, 1)}
    if oos_l is not None and not oos_l.empty:
        e = add_excess(oos_l)
        if not e.empty:
            prior["calibration"] = {"buckets": bucket_table(e), "cutoffs": cutoff_stats(e), "market": market_stats(e),
                                    "source": "walk_forward_oos"}
    return prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", default="2010-01-01")
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    run(a.start_date, a.end_date, save=not a.no_save)


if __name__ == "__main__":
    main()
