"""Clean, minimal Telegram messages (HTML parse mode). One idea per line, no jargon."""
from __future__ import annotations

from html import escape
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

AYLAR = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
REASON_TR = {"RANK_EXIT": "skor düştü", "FUND_BREAK": "temel bozulma", "DRAWDOWN_CONFIRMED": "düşüş + zayıflayan tez",
             "CATASTROPHE_STOP": "kesin stop (−%50)", "NEW_ENTRY": "yeni giriş",
             "ROTATION": "eski altın payı kapatıldı", "ALLOCATION": "varlık dağılımı",
             "COHORT_EXPIRY": "6 aylık süresi doldu", "NOT_SELECTED": "yeni seçimde yok"}


REGIME_TR = {"RISK_ON": "Olumlu 🟢", "NEUTRAL": "Nötr ⚪", "RISK_OFF": "Riskli 🔴", "UNKNOWN": "Belirsiz"}
GUARD_TR = {"NORMAL": "Normal", "WATCH": "Temkinli", "RECOVERY": "Toparlanıyor", "SAFE": "Güvenli mod — alım yok"}
BENCH_TR = {"cpi": "ABD TÜFE", "usd": "Dolar", "gold": "Altın", "deposit": "Hazine bonosu", "xu100": "Russell 2000"}


def tarih(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.day} {AYLAR[d.month - 1]} {d.year}"


def pct(v, sign: bool = False, nd: int = 1) -> str:
    try:
        x = float(v)
        if not np.isfinite(x):
            return "—"
    except (TypeError, ValueError):
        return "—"
    s = f"{abs(x):.{nd}f}".replace(".", ",")
    if sign:
        return ("+" if x >= 0 else "−") + "%" + s
    return ("−" if x < 0 else "") + "%" + s


def hurdle_block(h: Dict) -> List[str]:
    if not isinstance(h, dict) or h.get("hurdle") is None:
        return ["🎯 <b>Hedef kartı hesaplanamadı</b> — enflasyon verisi alınamadı (hisse seçimi etkilenmez)."]
    parts = [f"{BENCH_TR[k]} {pct(h.get(k))}" for k in ("cpi", "usd", "gold", "deposit") if h.get(k) is not None]
    if h.get("mode") == "sum":
        edge = h.get("edge", 3.0)
        return [f"🎯 <b>12 aylık hedef: {pct(h['hurdle'])}</b>  <i>(hepsinin toplamı + %{edge:.0f})</i>",
                "   " + " + ".join(parts)]
    return [f"🎯 <b>12 aylık çıta: {pct(h['hurdle'])}</b>  <i>(en yüksek: {BENCH_TR.get(h.get('binding'), '?')})</i>",
            "   " + " · ".join(parts)]


def portfolio_line(state: Dict) -> List[str]:
    pf = state.get("portfolio") or {}
    nav = pf.get("nav", 1.0) or 1.0
    n = len(pf.get("positions", {}))
    cash = pf.get("cash", 0.0) / nav * 100 if nav else 0.0
    out = [f"💼 Portföy: <b>{n} hisse</b>" + (f" · nakit {pct(cash, nd=0)}" if cash >= 1 else "")]
    navm = (state.get("performance") or {}).get("nav") or {}
    if navm.get("days", 0) >= 2:
        b = navm.get("benchmarks_total_pct") or {}
        out.append(f"📈 Başlangıçtan: <b>{pct(navm.get('total_return_pct'), True)}</b> · reel {pct(navm.get('real_total_pct'), True)}"
                   f" · Russell 2000 {pct(b.get('xu100', navm.get('xu100_total_pct')), True)}")
    return out


def monthly_report(today, state: Dict, rev: Dict, buys: List[Dict]) -> str:
    reg, g = state.get("regime", {}), state.get("autonomy_guard", {})
    L = [f"🦅 <b>ABD Small-Cap</b> · Aylık Rapor", f"<i>{tarih(today)} · emirler bir sonraki açılışta</i>", ""]
    L += hurdle_block(state.get("hurdles") or {})
    L.append("")
    picks = rev.get("picks") or []
    conf, grade, split = rev.get("confidence", {}), rev.get("confidence_grade", {}), rev.get("suggested_split", {})
    held_before = set(rev.get("holds", []))
    if picks:
        L.append("🟢 <b>Bu ayın alımları</b>  <i>(güven · yeni paranın önerilen payı)</i>")
        for t in picks:
            again = "  <i>(portföyde, ağırlığı artar)</i>" if t in held_before else ""
            L.append(f"<b>{escape(str(t))}</b>  güven {pct(conf.get(t, 0) * 100, nd=0)} ({grade.get(t, '—')})"
                     f" · pay {pct(split.get(t, 0) * 100, nd=0)}{again}")
    else:
        L.append("✋ Bu ay alım kriterini karşılayan likit hisse bulunamadı.")
    if rev.get("sells"):
        L.append("")
        L.append("🔴 <b>SAT</b>")
        for t, why in rev.get("sell_reasons", {}).items():
            L.append(f"<b>{escape(str(t))}</b>  {REASON_TR.get(why, why)}")
    ext = rev.get("extended") or []
    if ext:
        L.append("")
        L.append("⏳ <b>Süresi uzatıldı</b>  " + " · ".join(escape(str(t)) for t in ext)
                 + "  <i>(zararda ama puanı hâlâ yüksek: zararı realize etmek yerine tutuluyor, en fazla 18 ay daha)</i>")
    keep = [t for t in rev.get("holds", []) if t not in picks and t not in ext]
    if keep:
        L.append("")
        L.append("⚪ <b>TUT</b>  " + " · ".join(escape(str(t)) for t in keep))
    exp = [t for t in rev.get("expiring_next", []) if t not in picks]
    if exp:
        L.append("")
        L.append("⏳ <b>Gelecek ay 6 ayı doluyor</b>  " + " · ".join(escape(str(t)) for t in exp)
                 + "  <i>(o ay yeniden seçilmezse satılacak; zarardaysa ve puanı yüksekse tutulmaya devam eder)</i>")
    L.append("")
    L += portfolio_line(state)
    L.append("ℹ️ <i>Güven: 12 ayda tipik bir ABD küçük hissesinden çok kazanma olasılığı (%50 = yazı-tura). "
             "Her hisse en az 6 ay tutulur.</i>")
    L.append(f"🧭 Piyasa: {REGIME_TR.get(reg.get('label'), reg.get('label'))}")
    L += health_lines(state)
    return "\n".join(L)


def events_report(today, events: List[Dict], state: Dict, day_ret: Optional[float]) -> str:
    buys = [e for e in events if e["type"] == "BUY"]
    sells = [e for e in events if e["type"].startswith("SELL_")]
    flags = [e for e in events if e["type"] == "DRAWDOWN_FLAG"]
    stops = [e for e in events if e["type"] == "HARD_STOP_QUEUED"]
    L = [f"🔔 <b>ABD Small-Cap</b> · {tarih(today)}", ""]
    if buys:
        L.append("🟢 Alındı: " + " · ".join(f"<b>{escape(e['ticker'])}</b> {pct(e.get('w', 0) * 100, nd=0)}" for e in buys))
    for e in sells:
        why = REASON_TR.get(e["type"][5:], e["type"][5:])
        L.append(f"🔴 Satıldı: <b>{escape(e['ticker'])}</b> {pct(e.get('ret'), True)} <i>({why})</i>")
    for e in stops:
        L.append(f"⛔ Kesin stop: <b>{escape(e['ticker'])}</b> {pct(e.get('dd_entry'), True)} — yarın açılışta satılacak")
    for e in flags:
        L.append(f"⚠️ Sert düşüş: <b>{escape(e['ticker'])}</b> {pct(e.get('dd_peak'), True)} zirveden (bilgi amaçlı; hisse 6 aylık süresi dolunca yeniden değerlendirilir)")
    if day_ret is not None:
        L.append("")
        L.append(f"💼 Portföy bugün {pct(day_ret, True, 2)}")
    hl = health_lines(state)
    if hl and (state.get("health") or {}).get("overall") != "OK":
        L += [""] + hl
    return "\n".join(L)


def blocked_report(today, reason: str) -> str:
    return f"🛑 <b>ABD Small-Cap</b> · {tarih(today)}\nVeri kalitesi yetersiz ({escape(reason)}). Bugün işlem yapılmadı."


def status_report(today, state: Dict, refresh: bool) -> str:
    """Short daily status — sent on every run where nothing else was sent (so a run is never silent)."""
    pf = state.get("portfolio") or {}
    reg, g = state.get("regime", {}) or {}, state.get("autonomy_guard", {}) or {}
    lr = state.get("last_rebalance") or {}
    L = [f"📋 <b>ABD Small-Cap</b> · Günlük Durum", f"<i>{tarih(today)}</i>", ""]
    if refresh:
        L.append("⏳ Seans kapanmadı (ya da bugün seans yok): kapanış verisi New York kapanışından sonraki çalışmada işlenir. "
                 "Bu çalıştırmada TÜFE, faiz, kıyaslar ve piyasa durumu güncellendi.")
        L.append("")
    L += hurdle_block(state.get("hurdles") or {})
    L.append("")
    pend = [o for o in pf.get("pending", []) if o.get("action") == "BUY"]
    psell = [o for o in pf.get("pending", []) if o.get("action") == "SELL"]
    if pend:
        L.append("🕘 <b>Bir sonraki açılışta alınacak</b>")
        L.append("   " + " · ".join(f"<b>{escape(str(o['ticker']))}</b> {pct(o.get('target_w', 0) * 100, nd=0)}"
                                  + (f" (güven {pct(o['entry_conf'] * 100, nd=0)})" if o.get("entry_conf") is not None else "")
                                  for o in pend))
    if psell:
        L.append("🕘 <b>Bir sonraki açılışta satılacak:</b> " + " · ".join(escape(str(o["ticker"])) for o in psell))
    pos = pf.get("positions", {}) or {}
    exp = (state.get("last_rebalance") or {}).get("expiring_next") or []
    exp = [t for t in exp if t in pos]
    if exp:
        L.append("⏳ Ay başında 6 ayı dolacak: " + " · ".join(f"<b>{escape(str(t))}</b>" for t in exp))
    if pos:
        best = sorted(pos.items(), key=lambda kv: -kv[1].get("level", 1))
        L.append("💼 <b>Portföy</b>  " + " · ".join(f"{escape(str(t))} {pct((p.get('level', 1) - 1) * 100, True)}" for t, p in best[:12]))
    elif not pend:
        L.append("💼 Portföy boş — ilk alımlar ayın ilk seansındaki taramadan sonra yapılır.")
    L += portfolio_line(state)[1:]
    if lr.get("date"):
        L.append(f"🔎 Son tarama: {tarih(lr['date'])} · {lr.get('n_scored', '—')} hisse puanlandı · sonraki: ayın ilk seansı")
    L.append(f"🧭 Piyasa: {REGIME_TR.get(reg.get('label'), reg.get('label'))}")
    L += health_lines(state)
    return "\n".join(L)


def health_lines(state: Dict, full: bool = False) -> List[str]:
    h = state.get("health") or {}
    if not h.get("checks"):
        return []
    em = {"OK": "🟢", "UYARI": "🟡", "KRİTİK": "🔴"}
    ov = h.get("overall", "OK")
    if not full:
        if ov == "OK":
            return ["🩺 Sistem sağlığı: 🟢 her şey normal"]
        names = h.get("red") if ov == "KRİTİK" else h.get("yellow")
        tail = " — <b>bu ayki kararlara körü körüne uymayın</b>" if ov == "KRİTİK" else ""
        return [f"🩺 Sistem sağlığı: {em[ov]} {escape(', '.join(names or []))}{tail}"]
    L = [f"🩺 <b>Sistem sağlığı: {em.get(ov, '⚪')} {ov}</b>"]
    for c in h["checks"]:
        L.append(f"{em.get(c['status'], '⚪')} {escape(c['name'], quote=False)}: {escape(str(c['msg']), quote=False)}")
    lv = h.get("live_vs_test") or {}
    if lv.get("status") == "OK" and lv.get("band_ret_pct"):
        b = lv["band_ret_pct"]
        L.append(f"📐 Canlı {str(lv['months']).replace('.', ',')} ay: portföy {pct(lv['live_ret_pct'], True)} · testte aynı süre için normal aralık "
                 f"{pct(b['p10'], True)} … {pct(b['p90'], True)} (ortanca {pct(b['p50'], True)})")
    w = h.get("win_rates") or {}
    if (w.get("closed_lots") or {}).get("live"):
        L.append(f"🎯 Kazanma oranı (kapanan): canlı {pct(w['hit_nominal_pct']['live'], nd=0)} · test {pct(w['hit_nominal_pct']['test'], nd=0)}"
                 f" | Russell 2000'ü geçen: canlı {pct(w['hit_beat_xu100_pct']['live'], nd=0)} · test {pct(w['hit_beat_xu100_pct']['test'], nd=0)}")
    cm = state.get("confidence_model_meta") or {}
    if cm:
        L.append(f"🧠 Güven modeli: canlı sonuç payı %{cm.get('live_weight', 0) * 100:.0f} ({cm.get('n_live', 0)} sonuçlanmış gözlem)")
    return L
