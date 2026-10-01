"""US Small-Cap Real-Return Engine (port of BIST V3.10) — 100% stock portfolio, monthly cohorts,
calibrated confidence, value-trap guard, system health. Daily run after the New York close.

Every session:
  * execute yesterday's orders at today's OPEN, mark the portfolio to the CLOSE
    (adjusted change -> bonus issues are harmless), catastrophe-stop check,
  * NAV vs IWM vs US CPI bookkeeping, autonomy guard.
First session of every month (the monthly review):
  * price factors from adjusted yfinance history + TradingView fundamentals,
  * resolve 12-month labels of past monthly snapshots (nominal, REAL, vs IWM),
  * learning (champion/challenger) + calibration of expected REAL return,
  * V3.8: the top TRANCHE_N stocks form a new monthly cohort, cohorts are held TRANCHE_MONTHS,
    every pick carries a calibrated confidence -> orders for the next open.
`python main.py --self-test` runs everything on synthetic data in a temp folder.
"""
from __future__ import annotations

import os
import sys
import tempfile

if "--self-test" in sys.argv and "BOQ_DATA_DIR" not in os.environ:
    os.environ["BOQ_DATA_DIR"] = tempfile.mkdtemp(prefix="boq3_selftest_")

import argparse  # noqa: E402
from datetime import datetime  # noqa: E402
from html import escape  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import calendar_tr as cal  # noqa: E402
import config as C  # noqa: E402
import inflation as INF  # noqa: E402
import market_data as MD  # noqa: E402
import regime_model as RM  # noqa: E402
import benchmarks as BM  # noqa: E402
import telegram_report as TG  # noqa: E402
from autonomy_guard import evaluate_guard  # noqa: E402
from backtest_validator import enrich_lots, lot_metrics, nav_metrics  # noqa: E402
from calibration import calibrate  # noqa: E402
from data_integrity import validate_market_frame  # noqa: E402
from factors import price_factors_at, wide_from_history  # noqa: E402
from labels import forward_labels  # noqa: E402
from learner_engine import daily_rank_ic, live_composite_ic, newey_west, run_learning  # noqa: E402
from meta_engine import GOLD_TICKER, plan_tranche, score_universe  # noqa: E402
import confidence as CF  # noqa: E402
import health as HL  # noqa: E402
from state_manager import read_json  # noqa: E402
from portfolio import apply_day, new_portfolio, weights as pf_weights  # noqa: E402
from state_manager import (append_rows, load_monthly_snapshots, load_nav, load_research_prior,  # noqa: E402
                           load_state, load_trade_log, save_monthly_snapshots, save_state)

FUND_COLS = ["ticker", "market_cap", "sector", "roe", "pe", "pb", "ps", "net_income", "revenue",
             "rev_growth", "op_margin", "debt_to_equity", "div_yield"]
LABEL_COLS = ["fwd_1m", "fwd_3m", "fwd_ret", "real_ret", "xu_excess", "cpi_12m_pct",
              "hurdle_ret", "b_usd", "b_gold", "b_deposit", "beat_all"]


def send_telegram(message: str) -> bool:
    """Send (HTML). Logs the outcome; on an HTML parse error retries as plain text; splits long texts."""
    import re
    import requests
    token = (os.environ.get("TELEGRAM_TOKEN") or "").strip()
    chat_id = (os.environ.get("CHAT_ID") or "").strip()
    if not token or not chat_id:
        print("::warning::TELEGRAM_TOKEN / CHAT_ID secret'ı yok → mesaj yalnızca loga yazıldı.")
        print(message)
        return False
    chunks, buf = [], ""
    for line in message.split("\n"):
        if len(buf) + len(line) + 1 > 3900 and buf:
            chunks.append(buf)
            buf = ""
        buf += (("\n" if buf else "") + line)
    chunks.append(buf)
    ok_all = True
    for part in chunks:
        ok = False
        for mode in ("HTML", None):
            body = {"chat_id": chat_id, "text": part if mode else re.sub(r"<[^>]+>", "", part),
                    "disable_web_page_preview": True}
            if mode:
                body["parse_mode"] = mode
            try:
                r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json=body, timeout=20)
                if r.status_code == 200:
                    ok = True
                    break
                print(f"::warning::Telegram {r.status_code}: {r.text[:300]}")
            except Exception as exc:
                print(f"::warning::Telegram hatası: {exc}")
        ok_all &= ok
    print("📨 Telegram gönderildi." if ok_all else "::error::Telegram mesajı gönderilemedi (TOKEN / CHAT_ID kontrol edin).")
    return ok_all


def _fingerprint(df: pd.DataFrame) -> str:
    top = df.sort_values("value_traded", ascending=False).head(60)
    return f"{np.round(top['close'].to_numpy(float), 4).sum():.4f}|{np.round(top['volume'].to_numpy(float), 0).sum():.0f}"


def resolve_labels(snaps: pd.DataFrame, wide, cpi, index_close, bm=None, crate=None) -> pd.DataFrame:
    """Fill 1m/3m/12m labels of past monthly snapshots once the windows have elapsed."""
    if snaps.empty:
        return snaps
    s = snaps.copy()
    for c in LABEL_COLS:
        if c not in s.columns:
            s[c] = np.nan
    need = s[s["fwd_ret"].isna() | s["fwd_3m"].isna() | s["fwd_1m"].isna()]["tarih"].unique()
    if len(need) and wide is not None:
        lab = forward_labels(wide, list(need), cpi, index_close, bm, crate)
        if not lab.empty:
            lab = lab.rename(columns={c: f"{c}__new" for c in LABEL_COLS})
            s = s.merge(lab, on=["tarih", "ticker"], how="left")
            for c in LABEL_COLS:
                s[c] = s[c].fillna(s[f"{c}__new"])
            s = s.drop(columns=[f"{c}__new" for c in LABEL_COLS])
    # real return can resolve later than the nominal one (CPI publication lag)
    m = s["fwd_ret"].notna() & s["real_ret"].isna()
    if m.any() and cpi is not None and len(cpi):
        cr = INF.cpi_ratio_vec(cpi, s.loc[m, "tarih"], s.loc[m, "tarih"] + pd.DateOffset(months=12))
        s.loc[m, "real_ret"] = ((1 + s.loc[m, "fwd_ret"] / 100.0) / cr - 1.0) * 100.0
        s.loc[m, "cpi_12m_pct"] = (cr - 1.0) * 100.0
    # multi-benchmark hurdle can also resolve later (CPI lag)
    m2 = s["fwd_ret"].notna() & s["hurdle_ret"].isna()
    if m2.any():
        for d in s.loc[m2, "tarih"].unique():
            w = BM.window_returns(bm, cpi, crate, d, pd.Timestamp(d) + pd.DateOffset(months=12))
            if np.isfinite(w["hurdle"]):
                mm = m2 & (s["tarih"] == d)
                s.loc[mm, "hurdle_ret"] = w["hurdle"]
                s.loc[mm, "b_usd"], s.loc[mm, "b_gold"], s.loc[mm, "b_deposit"] = w["usd"], w["gold"], w["deposit"]
                s.loc[mm, "beat_all"] = (s.loc[mm, "fwd_ret"] > w["hurdle"]).astype(float)
    return s


def ic_stats(dataset: pd.DataFrame, target: str) -> dict:
    if dataset is None or dataset.empty or target not in dataset:
        return {"n_dates": 0}
    d = dataset.dropna(subset=[target, "composite"])
    ic = daily_rank_ic(d, ["composite"], target)
    if ic.empty:
        return {"n_dates": 0}
    s = ic["composite"]
    lag = 11 if target == "fwd_ret" else (2 if target == "fwd_3m" else 0)
    m, se, t, n = newey_west(s, lag)
    return {"n_dates": n, "ic_mean": round(m, 4), "t_nw": round(t, 2), "ic_recent": round(float(s.tail(6).mean()), 4)}


def gold_bar_today(bm, today):
    """Only used to SELL a legacy V3.5-3.7 gold sleeve (V3.8 never buys gold)."""
    if bm is None or "gold_try" not in bm or bm["gold_try"].notna().sum() < 2:
        return None
    g = bm["gold_try"].dropna()
    g = g[g.index <= pd.Timestamp(today)]
    if len(g) < 2:
        return None
    return pd.DataFrame({"open": [g.iloc[-1]], "close": [g.iloc[-1]], "chg_pct": [(g.iloc[-1] / g.iloc[-2] - 1) * 100]},
                        index=[GOLD_TICKER])


def monthly_review(state, research, snap, today, index_close, cpi, cpi_stats, guard, hist_fn, as_of=None,
                   bm=None, crate=None):
    """`as_of` = last COMPLETED session (used by intraday refresh runs: no partial bars)."""
    pf = state["portfolio"]
    intraday = as_of is not None
    today = pd.Timestamp(as_of) if intraday else today
    snaps = load_monthly_snapshots()
    universe = snap.sort_values("value_traded", ascending=False)["ticker"].head(C.SCAN_LIMIT).tolist()
    universe = sorted(set(universe) | set(pf["positions"].keys()))
    start = today - pd.Timedelta(days=430)
    unresolved = []
    if not snaps.empty:
        pend = snaps[snaps.get("fwd_ret").isna()] if "fwd_ret" in snaps else snaps
        pend = pend[pend["tarih"] >= today - pd.Timedelta(days=500)]
        if not pend.empty:
            start = min(start, pend["tarih"].min() - pd.Timedelta(days=10))
            unresolved = sorted(set(pend["ticker"]))
    hist = hist_fn(sorted(set(universe) | set(unresolved)), str(start.date()))
    hist = {t: g[g.index <= today] for t, g in hist.items()}
    hist = {t: g for t, g in hist.items() if len(g)}
    if len(hist) < C.MIN_CROSS_SECTION:
        return {"status": "HISTORY_UNAVAILABLE", "n_hist": len(hist)}, []
    wide = wide_from_history(hist)
    snaps = resolve_labels(snaps, wide, cpi, index_close, bm, crate)

    dataset = snaps.dropna(subset=["fwd_ret"]) if not snaps.empty and "fwd_ret" in snaps else pd.DataFrame()
    run_learning(state, dataset, research)
    calibrate(state, dataset, research)
    state["model"]["ic_live_12m"] = ic_stats(snaps, "fwd_ret")
    state["model"]["ic_live_3m"] = ic_stats(snaps, "fwd_3m")

    price_f = price_factors_at(wide, index_close, today)
    price_f = price_f[price_f["ticker"].isin(universe)]
    fund = snap[[c for c in FUND_COLS if c in snap.columns]].copy()
    frame, info = score_universe(price_f, fund, state, cpi_stats, state.get("sector_map"))
    if frame.empty:
        return {"status": "NO_FACTORS", **info}, []
    chg = None if intraday else snap.set_index("ticker")["change_pct"]
    state.pop("active_strategy", None)
    conf_prior = (research or {}).get(C.CONFIDENCE_FILE_KEY) if isinstance(research, dict) else None
    conf_model, conf_meta = CF.live_update(conf_prior, snaps)          # self-improving confidence
    state["confidence_model_meta"] = conf_meta
    state["confidence_live"] = HL.confidence_live_check(snaps)
    orders, summ = plan_tranche(pf, frame, state, today, today_change=chg, conf_model=conf_model)
    conf_all = CF.predict(frame, conf_model)
    frame = frame.assign(confidence=conf_all.to_numpy())
    keep = [o for o in pf["pending"] if o.get("reason") == "CATASTROPHE_STOP"]
    pf["pending"] = keep + [o for o in orders if o["ticker"] not in {k["ticker"] for k in keep}]
    pf["last_rebalance_month"] = today.strftime("%Y-%m")

    sel = set(summ.get("holds", [])) | set(summ.get("buys", []))
    cols = ["ticker", "sector", "close_adj", "med_value_traded", "vol_ann_pct", "beta", "composite", "composite_pct",
            "exp_real_12m", "exp_nominal_12m", "hurdle_12m", "exp_over_hurdle", "p_beat_cpi", "p_beat_all",
            "regime_label", "model_version", "fund_break", "confidence", "max_1m"] + \
        [f"f_{k}" for k in C.FACTORS] + [f"z_{k}" for k in C.FACTORS]
    rec = frame[[c for c in cols if c in frame.columns]].copy()
    rec["tarih"] = today
    rec["selected"] = rec["ticker"].isin(sel)
    for c in LABEL_COLS:
        rec[c] = np.nan
    base = snaps[snaps["tarih"] != today] if not snaps.empty else snaps
    save_monthly_snapshots(pd.concat([base, rec], ignore_index=True) if not base.empty else rec)
    top = frame[frame["ticker"].isin(summ.get("picks", []))].sort_values("composite", ascending=False)
    state["last_picks"] = {"month": summ.get("month"), "picks": [
        {"ticker": r["ticker"], "confidence": round(float(r["confidence"]), 4), "grade": summ["confidence_grade"].get(r["ticker"]),
         "split": summ["suggested_split"].get(r["ticker"]), "score_pct": round(float(r["composite_pct"]), 1),
         "sector": r.get("sector") if isinstance(r.get("sector"), str) else None,
         "exp_nominal_12m": None if pd.isna(r.get("exp_nominal_12m")) else round(float(r["exp_nominal_12m"]), 1)}
        for _, r in top.iterrows()]}
    return {"status": "OK", **info, **summ}, top.to_dict("records")


def run(force: bool = False, today=None, fetch=MD.fetch_snapshot, hist_fn=MD.download_history,
        cpi_fn=INF.load_cpi, regime_fn=RM.download_regime_series, cash_fn=INF.load_cash_rate,
        bench_fn=BM.download_benchmarks, refresh: bool = False, rescan: bool = False) -> dict:
    real_run = today is None
    today = pd.Timestamp(today) if today is not None else cal.today_tr()
    refresh = bool(refresh)
    if real_run and not force and not refresh:
        now_tr = pd.Timestamp.now(tz=C.MARKET_TZ)
        if now_tr.hour * 60 + now_tr.minute < C.CLOSE_READY_MIN or not cal.is_session(today):
            refresh = True
            print(f"ℹ️ Kapanış verisi yok ({now_tr:%d.%m %H:%M}) → YENİLEME modu: ABD TÜFE/T-bill/rejim güncellenir, "
                  "gerekirse aylık gözden geçirme son tamamlanan seansa göre yenilenir; işlem/NAV kaydı yapılmaz.")
    if not refresh and not cal.is_session(today) and not force:
        print(f"ℹ️ {today.date()} NYSE seansı değil.")
        return {"status": "NOT_SESSION"}
    as_of = None
    if refresh:
        d = today - pd.Timedelta(days=1)
        while not cal.is_session(d):
            d -= pd.Timedelta(days=1)
        as_of = d

    state = load_state()
    research = load_research_prior()
    raw, meta = fetch(state)
    if not raw.empty:
        raw["tarih"] = today
    snap, quality = validate_market_frame(raw)
    state["data_quality"] = {**quality, "fetch": meta}
    if snap.empty or not quality.get("ok"):
        state["last_run"] = {"date": str(today.date()), "status": f"DATA_BLOCKED:{quality.get('reason')}"}
        save_state(state)
        send_telegram(TG.blocked_report(today, str(quality.get("reason"))))
        return {"status": "DATA_BLOCKED"}
    fp = _fingerprint(snap)
    if refresh:
        fp = state.get("last_run", {}).get("fingerprint")      # a refresh never consumes the day
    elif not force and state.get("last_run", {}).get("fingerprint") == fp:
        print("ℹ️ Veri bir önceki çalışmayla aynı (tatil/donmuş veri); atlandı.")
        return {"status": "STALE"}
    if "sector" in snap:
        state.setdefault("sector_map", {}).update({t: s for t, s in zip(snap["ticker"], snap["sector"]) if isinstance(s, str) and s})

    # regime + index
    index_close = None
    try:
        rser = regime_fn()
        index_close = rser["idx"]
    except Exception as exc:
        rser = None
        print(f"⚠️ Rejim serisi alınamadı: {exc}")
    RM.update_regime(state, snap, today, series=rser, allow_download=False)

    # inflation
    cpi, cmeta = cpi_fn()
    if cmeta.get("status") not in ("OK", "STALE") and rser is not None and "fx" in rser:
        px = INF.proxy_from_fx(rser["fx"])
        if len(px) >= 24:
            cpi, cmeta = px, {**cmeta, "source": "USDTRY_PROXY", "status": "PROXY", "last_month": str(px.index[-1].date())}
    cpi_stats = INF.inflation_stats(cpi, proxy=cmeta.get("status") == "PROXY")
    state["inflation"] = {**cpi_stats, **cmeta}

    # portfolio: execute pending orders at today's open, mark to close
    if not state.get("portfolio"):
        state["portfolio"] = new_portfolio(today)
    pf = state["portfolio"]
    # A review taken while inflation was unknown had no real-return gate: cancel its
    # not-yet-executed buys and redo the review now that an inflation estimate exists.
    lr = state.get("last_rebalance", {}) or {}
    if lr.get("status") == "OK" and lr.get("expected_inflation_12m") is None and cpi_stats.get("expected_12m_pct") is not None:
        before = len(pf["pending"])
        pf["pending"] = [o for o in pf["pending"] if o["action"] == "SELL"]
        pf["last_rebalance_month"] = None
        print(f"ℹ️ Enflasyon kapısı olmadan verilmiş {before - len(pf['pending'])} emir iptal edildi; aylık gözden geçirme yenileniyor.")
    # Engine upgraded (or hurdle rule changed) after the review, or a manual rescan was asked:
    # cancel the not-yet-executed buys of that review and redo it with the current logic.
    stale_logic = (lr.get("status") == "OK" and pf.get("last_rebalance_month")
                   and (lr.get("engine_version") != C.ENGINE_VERSION or lr.get("hurdle_mode") != getattr(C, "HURDLE_MODE", "max"))
                   and any(o["action"] == "BUY" for o in pf["pending"]))
    if rescan or stale_logic:
        before = len(pf["pending"])
        pf["pending"] = [o for o in pf["pending"] if o["action"] == "SELL"]
        pf["last_rebalance_month"] = None
        print(f"ℹ️ {'Elle yeniden tarama' if rescan else 'Motor güncellendi (' + str(lr.get('engine_version')) + ' → ' + C.ENGINE_VERSION + ')'}: "
              f"{before - len(pf['pending'])} bekleyen alım iptal edildi, aylık gözden geçirme yeni kurallarla yenileniyor.")
    crate, cr_meta = cash_fn()
    cash_y = INF.cash_yield_at(crate, today)
    state["cash_rate"] = {**cr_meta, "net_yield_pct": round(cash_y, 2)}
    # benchmarks (USD, gold, Russell 2000) and the forward-looking multi-benchmark hurdle
    try:
        bm = bench_fn()
    except Exception as exc:
        print(f"⚠️ Kıyas serileri alınamadı ({exc}); USDTRY rejim serisinden kullanılıyor, altın yok.")
        bm = BM.assemble(rser["fx"], None, rser["idx"]) if rser is not None else None
    state["hurdles"] = BM.expected_hurdles(bm, cpi_stats, cash_y if len(crate) else None, as_of=as_of or today)
    bench = {"bm": bm, "crate": crate}
    events, lots = [], []
    wts = pf_weights(pf)
    nav_df = load_nav()
    if not refresh:
        bars = snap.set_index("ticker")[["open", "close", "change_pct"]].rename(columns={"change_pct": "chg_pct"})
        gbar = gold_bar_today(bm, today)
        if gbar is not None:
            bars = pd.concat([bars, gbar])
        events, lots = apply_day(pf, bars, today, cash_yield_pct=cash_y)
        if lots:
            append_rows(C.TRADE_LOG_FILE, lots)
        xu = float(index_close.iloc[-1]) if index_close is not None and len(index_close) else np.nan
        wts = pf_weights(pf)
        append_rows(C.NAV_FILE, [{"tarih": str(today.date()), "nav": round(pf["nav"], 6), "cash": round(pf["cash"], 6),
                                  "n_positions": len(pf["positions"]), "exposure": round(sum(wts.values()), 4),
                                  "xu100": xu}])
        nav_df = load_nav()
        guard = evaluate_guard(state, features=snap, data_quality=quality.get("score", 0.0), rows=quality.get("rows"),
                               live_ic=state.get("model", {}).get("ic_live_3m"), nav_df=nav_df)
    else:
        g0 = state.get("autonomy_guard", {}) or {}
        guard = {"mode": g0.get("mode", "WATCH"), "block_new_entries": bool(g0.get("block_new_entries", False))}

    review, top = None, []
    state["_no_inflation"] = cpi_stats.get("expected_12m_pct") is None
    review_month = (as_of or today).strftime("%Y-%m")
    if pf.get("last_rebalance_month") != review_month:
        review, top = monthly_review(state, research, snap, today, index_close, cpi, cpi_stats, guard, hist_fn, as_of=as_of,
                                     bm=bm, crate=crate)
        state["last_rebalance"] = {"date": str((as_of or today).date()), "mode": "REFRESH" if refresh else "EOD",
                                   "engine_version": C.ENGINE_VERSION, "hurdle_mode": getattr(C, "HURDLE_MODE", "max"),
                                   **{k: v for k, v in review.items() if k not in ("weights",)}}

    lots_all = enrich_lots(load_trade_log(), cpi, index_close, bench)
    state["performance"] = {"nav": nav_metrics(nav_df, cpi, bench), "lots": lot_metrics(lots_all),
                            "open_positions": {t: {"w": round(wts.get(t, 0), 4), "ret_pct": round((p["level"] - 1) * 100, 2),
                                                   "since": p["entry_date"]} for t, p in pf["positions"].items()}}
    state.pop("_no_inflation", None)
    if not refresh:
        state["last_eod_date"] = str(today.date())
    try:
        state["health"] = HL.evaluate(state, None, nav_df, read_json(C.BACKTEST_REPORT_FILE), today)
    except Exception as exc:
        state["health"] = {"overall": "UYARI", "checks": [], "error": str(exc)[:200]}
    state["last_run"] = {"date": str(today.date()), "status": "OK", "fingerprint": fp, "mode": "REFRESH" if refresh else "EOD",
                         "monthly_review": bool(review), "utc": datetime.utcnow().isoformat() + "Z"}
    save_state(state)

    if review and review.get("status") == "OK":
        send_telegram(TG.monthly_report(as_of or today, state, review, top))
    elif review:
        send_telegram(f"⚠️ <b>ABD Small-Cap</b>\nAylık gözden geçirme tamamlanamadı ({escape(str(review.get('status')))}). Bir sonraki çalışmada tekrar denenecek.")
        pf["last_rebalance_month"] = None
        save_state(state)
    elif events:
        nv = pd.to_numeric(nav_df["nav"], errors="coerce").dropna() if nav_df is not None and "nav" in nav_df else pd.Series(dtype=float)
        day_ret = float((nv.iloc[-1] / nv.iloc[-2] - 1) * 100) if len(nv) >= 2 else None
        send_telegram(TG.events_report(today, events, state, day_ret))
    else:
        send_telegram(TG.status_report(today, state, refresh))
    print(f"✅ {today.date()} [{'YENİLEME' if refresh else 'EOD'}] | NAV {pf['nav']:.4f} | pozisyon {len(pf['positions'])} | aylık={'evet' if review else 'hayır'}")
    return {"status": "OK", "state": state, "events": events, "review": review}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="TÜFE/nakit/rejim güncelle, işlem yapma")
    ap.add_argument("--rescan", action="store_true", help="bu ayın taramasını şimdi yeniden yap (bekleyen alımlar iptal)")
    a = ap.parse_args()
    if a.self_test:
        from selftest import run_self_test
        sys.exit(0 if run_self_test() else 1)
    run(force=a.force, refresh=a.refresh, rescan=a.rescan or os.environ.get("BOQ_RESCAN", "").lower() in ("1", "true", "evet"))


if __name__ == "__main__":
    main()
