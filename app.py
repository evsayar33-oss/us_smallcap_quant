"""ABD Small-Cap V3.8 — %100 hisse portföyü, aylık dilimler, kalibre güven oranı (salt okunur pano)."""
import json
import math
import os

import pandas as pd
import streamlit as st

import config as C

st.set_page_config(page_title="ABD Small-Cap", layout="centered", page_icon="🦅", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ style
st.markdown("""
<style>
#MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 760px;}
.hdr {font-size: 1.55rem; font-weight: 700; letter-spacing: -.02em; margin-bottom: .1rem;}
.sub {opacity: .65; font-size: .85rem; margin-bottom: 1rem;}
.card {border: 1px solid rgba(128,128,128,.22); border-radius: 16px; padding: 14px 16px; margin: 8px 0;
       background: rgba(128,128,128,.06);}
.big {font-size: 2.1rem; font-weight: 700; line-height: 1.1; letter-spacing: -.02em;}
.lbl {font-size: .78rem; opacity: .65; text-transform: uppercase; letter-spacing: .04em;}
.chip {display: inline-block; padding: 3px 10px; margin: 4px 6px 0 0; border-radius: 999px; font-size: .82rem;
       border: 1px solid rgba(128,128,128,.3);}
.chip.on {border-color: #f5a524; background: rgba(245,165,36,.14); font-weight: 600;}
.kpi {text-align: left;}
.kpi .v, .v {font-size: 1.35rem; font-weight: 700;}
.pos {color: #17c964;} .neg {color: #f31260;} .mut {opacity: .6;}
.row {display: flex; justify-content: space-between; align-items: center; padding: 10px 0;
      border-bottom: 1px solid rgba(128,128,128,.15);}
.row:last-child {border-bottom: none;}
.tk {font-weight: 700; font-size: 1.02rem;}
.bar {height: 6px; border-radius: 3px; background: rgba(128,128,128,.18); margin-top: 5px; width: 120px;}
.bar > div {height: 6px; border-radius: 3px; background: #7c6cf2;}
.badge {display: inline-block; padding: 4px 12px; border-radius: 10px; font-weight: 600; font-size: .9rem;}
.b-in {background: rgba(23,201,100,.15); color: #17c964;} .b-buy {background: rgba(124,108,242,.18); color: #9d90ff;}
.b-watch {background: rgba(128,128,128,.15);} .b-no {background: rgba(243,18,96,.13); color: #f31260;}
.note {font-size: .88rem; line-height: 1.45;}
.cbar {height: 8px; border-radius: 4px; background: rgba(128,128,128,.18); width: 100%; margin-top: 6px;}
.cbar > div {height: 8px; border-radius: 4px;}
.g-hi {background: #17c964;} .g-mid {background: #f5a524;} .g-lo {background: #9aa0a6;}
.pick {padding: 12px 0; border-bottom: 1px solid rgba(128,128,128,.15);} .pick:last-child {border-bottom: none;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=120)
def _json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=120)
def _csv(path):
    return pd.read_csv(path, low_memory=False) if os.path.exists(path) else pd.DataFrame()


def num(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def pct(v, sign=False, nd=1):
    x = num(v)
    if x is None:
        return "—"
    s = f"{abs(x):.{nd}f}".replace(".", ",")
    return (("+" if x >= 0 else "−") if sign else ("−" if x < 0 else "")) + "%" + s


def cls(v):
    x = num(v)
    return "mut" if x is None else ("pos" if x >= 0 else "neg")


AYLAR = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
BENCH = {"cpi": "ABD TÜFE", "usd": "Dolar", "gold": "Altın", "deposit": "Hazine bonosu", "xu100": "Russell 2000"}
REGIME = {"RISK_ON": "Olumlu", "NEUTRAL": "Nötr", "RISK_OFF": "Riskli"}
GUARD = {"NORMAL": "Normal", "WATCH": "Temkinli", "RECOVERY": "Toparlanıyor", "SAFE": "Güvenli mod"}
FACTOR_TR = {
    "mom_12_1": "12 aylık momentum", "high_52w": "52 hafta zirvesine yakınlık", "trend_consistency": "istikrarlı yükseliş",
    "low_vol": "düşük oynaklık", "low_beta": "piyasadan bağımsızlık", "dd_resilience": "düşüşlere dayanıklılık",
    "liquidity": "likidite", "roe": "özsermaye kârlılığı", "earnings_yield": "kâra göre ucuzluk",
    "book_yield": "defter değerine göre ucuzluk", "sales_yield": "satışlara göre ucuzluk", "op_margin": "faaliyet marjı",
    "low_leverage": "düşük borç", "div_yield": "temettü", "real_growth": "enflasyon üstü büyüme",
}
REASON_TR = {"RANK_EXIT": "skor düştü", "FUND_BREAK": "temel bozulma", "DRAWDOWN_CONFIRMED": "düşüş + zayıflayan tez",
             "CATASTROPHE_STOP": "kesin stop", "NEW_ENTRY": "yeni giriş", "COHORT_EXPIRY": "6 aylık süresi doldu",
             "NOT_SELECTED": "yeni seçimde yok", "ROTATION": "eski altın payı kapatıldı"}


def dstr(s):
    try:
        d = pd.Timestamp(s)
        return f"{d.day} {AYLAR[d.month - 1]} {d.year}"
    except Exception:
        return "—"


state = _json(C.STATE_FILE)
report = _json(C.BACKTEST_REPORT_FILE)
nav = _csv(C.NAV_FILE)
snaps = _csv(C.MONTHLY_SNAPSHOT_FILE)

st.markdown('<div class="hdr">🦅 ABD Small-Cap</div>', unsafe_allow_html=True)
if not state:
    st.markdown('<div class="sub">Henüz veri yok. GitHub Actions → önce “Walk Forward Backtest”, sonra “Daily Run”.</div>',
                unsafe_allow_html=True)
    st.stop()

pf = state.get("portfolio") or {}
reg, g = state.get("regime", {}) or {}, state.get("autonomy_guard", {}) or {}
hz = state.get("hurdles", {}) or {}
perf = state.get("performance", {}) or {}
navm = perf.get("nav", {}) or {}
lr = state.get("last_run", {}) or {}
weights = (state.get("model", {}) or {}).get("champion_weights", {}) or {}
cal = state.get("calibration", {}) or {}
last = snaps[snaps["tarih"] == snaps["tarih"].max()] if not snaps.empty else pd.DataFrame()

HS = state.get("health") or {}
HEM = {"OK": "🟢", "UYARI": "🟡", "KRİTİK": "🔴"}
st.markdown(f'<div class="sub">Güncelleme {dstr(lr.get("date"))} · Piyasa {REGIME.get(reg.get("label"), "—")} · '
            f'Sistem sağlığı {HEM.get(HS.get("overall"), "⚪")} {HS.get("overall", "—")}</div>', unsafe_allow_html=True)
if HS.get("overall") == "KRİTİK":
    st.error("Sistem sağlığı KRİTİK: " + ", ".join(HS.get("red") or []) + " — ayrıntı için “Sağlık” sekmesine bakın; "
             "sorun çözülene kadar önerileri körü körüne uygulamayın.")

# ------------------------------------------------------------------ one actionable alert at most
inf = state.get("inflation", {}) or {}
if inf.get("expected_12m_pct") is None:
    st.warning("ABD TÜFE (FRED) verisi alınamadı → hedef kartı hesaplanamıyor (hisse seçimi etkilenmez).")
elif inf.get("status") == "PROXY":
    st.warning("ABD TÜFE verisi eski.")

# ------------------------------------------------------------------ hurdle card
if hz.get("hurdle") is not None and hz.get("mode") == "sum":
    keys = [k for k in ("cpi", "usd", "gold", "deposit") if hz.get(k) is not None]
    chips = '<span class="mut"> + </span>'.join(f'<span class="chip">{BENCH[k]} {pct(hz.get(k))}</span>' for k in keys)
    edge = hz.get("edge", C.MIN_EDGE_OVER_HURDLE_PCT)
    chips += f'<span class="mut"> + </span><span class="chip on">Fark %{edge:.0f}</span>'
    st.markdown(f'<div class="card"><div class="lbl">Portföyün 12 aylık hedefi</div>'
                f'<div class="big">{pct(hz["hurdle"])}</div>{chips}'
                f'<div class="note mut" style="margin-top:8px">Hedef: ABD TÜFE, altın ve hazine bonosunun '
                f'<b>toplamını</b> en az %{edge:.0f} farkla geçmek. Performans bu hedefe göre ölçülür; hisse seçimi her ay '
                f'en güçlü hisseleri alır.</div></div>',
                unsafe_allow_html=True)
elif hz.get("hurdle") is not None:
    chips = "".join(f'<span class="chip {"on" if k == hz.get("binding") else ""}">{BENCH[k]} {pct(hz.get(k))}</span>'
                    for k in ("cpi", "usd", "gold", "deposit") if hz.get(k) is not None)
    st.markdown(f'<div class="card"><div class="lbl">Hisselerin geçmesi gereken 12 aylık çıta</div>'
                f'<div class="big">{pct(hz["hurdle"])}</div>{chips}'
                f'<div class="note mut" style="margin-top:8px">Seçilen her hissenin 12 ayda TÜFE, dolar (+ABD enflasyonu), '
                f'altın ve mevduatın hepsini en az %{C.MIN_EDGE_OVER_HURDLE_PCT:.0f} farkla geçmesi beklenir.</div></div>',
                unsafe_allow_html=True)

# ------------------------------------------------------------------ KPIs
bt = (navm.get("benchmarks_total_pct") or {})
if navm.get("days", 0) >= 2:
    kp = "".join(f'<div style="flex:1;min-width:0"><div class="lbl">{lbl}</div><div class="kpi v {cls(v)}">{pct(v, True)}</div></div>'
                 for lbl, v in (("Portföy", navm.get("total_return_pct")), ("Reel", navm.get("real_total_pct")),
                                ("Russell 2000", bt.get("xu100", navm.get("xu100_total_pct")))))
    st.markdown(f'<div class="card"><div class="lbl" style="margin-bottom:6px">Başlangıçtan bu yana</div>'
                f'<div style="display:flex;gap:12px">{kp}</div></div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="card note mut">Portföy {dstr(pf.get("start_date"))} tarihinde başladı. '
                'Getiri kartları ilk işlem günlerinden sonra görünür.</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Portföy", "Hisse Ara", "Performans", "Sağlık"])

# ------------------------------------------------------------------ PORTFÖY
with tab1:
    lp = state.get("last_picks") or {}
    if lp.get("picks"):
        GR = {"Yüksek": "g-hi", "Orta": "g-mid", "Düşük": "g-lo"}
        rows = ""
        for x in lp["picks"]:
            cf = num(x.get("confidence")) or 0
            rows += (f'<div class="pick"><div style="display:flex;justify-content:space-between;align-items:baseline">'
                     f'<div><span class="tk">{x["ticker"]}</span> <span class="mut" style="font-size:.8rem">{x.get("sector") or ""}</span></div>'
                     f'<div><b>%{cf * 100:.0f}</b> <span class="mut" style="font-size:.8rem">güven · {x.get("grade") or ""}</span></div></div>'
                     f'<div class="cbar"><div class="{GR.get(x.get("grade"), "g-lo")}" style="width:{max(min((cf - 0.35) / 0.35, 1), 0.05) * 100:.0f}%"></div></div>'
                     f'<div class="mut" style="font-size:.8rem;margin-top:4px">önerilen pay (yeni para) {pct((num(x.get("split")) or 0) * 100, nd=0)}'
                     f' · skor {num(x.get("score_pct")) or 0:.0f}/100</div></div>')
        cm = (report.get("confidence_model") or {}) if report else {}
        calib = ""
        if cm.get("oos_top10_predicted") is not None:
            calib = (f' Geçmiş testte model aylık ilk 10 hisse için ortalama %{cm["oos_top10_predicted"] * 100:.0f} dedi, '
                     f'gerçekleşen %{cm["oos_top10_realised"] * 100:.0f} oldu.')
        st.markdown(f'<div class="card"><div class="lbl">Bu ayın alım önerileri · {lp.get("month", "")}</div>{rows}'
                    f'<div class="note mut" style="margin-top:8px"><b>Güven</b> = hissenin 12 ayda tipik bir ABD küçük hissesinden '
                    f'daha çok kazanma olasılığı (%50 = yazı-tura).{calib} Hisseler arasında para dağıtırken bunu kullanın; '
                    f'piyasanın genel yönü bütün hisseleri birlikte etkiler.</div></div>', unsafe_allow_html=True)
    pos = pf.get("positions", {}) or {}
    navv = pf.get("nav", 1.0) or 1.0
    exit_m = {}
    for c in pf.get("cohorts", []) or []:
        try:
            em = (pd.Period(c["month"], "M") + C.TRANCHE_MONTHS)
        except Exception:
            continue
        for t in c.get("tickers", []):
            exit_m[t] = max(exit_m.get(t, em), em)
    if pos:
        rows = ""
        for t, p in sorted(pos.items(), key=lambda kv: -kv[1]["value"]):
            w = p["value"] / navv * 100
            r = (p["level"] - 1) * 100
            flag = ' <span class="chip">⚠️ sert düşüş</span>' if p.get("dd_flag") else ""
            ec = num(p.get("entry_conf"))
            em = exit_m.get(t)
            soon = em is not None and em == (pd.Period(pd.Timestamp.now(), "M") + 1)
            flag += ' <span class="chip on">⏳ gelecek ay 6 ayı doluyor</span>' if soon else ""
            meta = f'{dstr(p["entry_date"])} alındı' + (f' · güven %{ec * 100:.0f}' if ec is not None else "") + \
                (f' · gözden geçirme {AYLAR[em.month - 1]} {em.year}' if em is not None else "")
            rows += (f'<div class="row"><div><span class="tk">{t}</span>{flag}<div class="mut" style="font-size:.8rem">'
                     f'{meta}</div></div><div style="text-align:right">'
                     f'<div class="{cls(r)}" style="font-weight:700">{pct(r, True)}</div>'
                     f'<div class="bar"><div style="width:{min(w / (C.MAX_NAME_W * 100) * 100, 100):.0f}%"></div></div>'
                     f'<div class="mut" style="font-size:.78rem">ağırlık {pct(w, nd=0)}</div></div></div>')
        st.markdown(f'<div class="card"><div class="lbl">{len(pos)} hisse · nakit {pct(pf.get("cash", 0) / navv * 100, nd=0)}</div>'
                    f'{rows}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card note mut">Henüz pozisyon yok. Aylık seçimdeki emirler bir sonraki seans açılışında gerçekleşir.</div>',
                    unsafe_allow_html=True)
    pend = pf.get("pending", []) or []
    if pend:
        rows = ""
        for o in pend:
            if o["action"] == "BUY":
                ec = num(o.get("entry_conf"))
                extra = f'güven %{ec * 100:.0f}' if ec is not None else ""
                rows += (f'<div class="row"><div><span class="tk">🟢 {o["ticker"]}</span><div class="mut" style="font-size:.8rem">'
                         f'{extra}</div></div>'
                         f'<div style="font-weight:600">{pct(o.get("target_w", 0) * 100, nd=0)}</div></div>')
            elif o["action"] == "SELL":
                rows += (f'<div class="row"><div><span class="tk">🔴 {o["ticker"]}</span></div>'
                         f'<div class="mut">{REASON_TR.get(o.get("reason"), o.get("reason"))}</div></div>')
        if rows:
            st.markdown(f'<div class="card"><div class="lbl">Bir sonraki açılışta</div>{rows}</div>', unsafe_allow_html=True)
    st.caption(f"Kural: her ay en güçlü {C.TRANCHE_N} hisse alınır (aynı sektörden en fazla {C.TRANCHE_SECTOR_CAP}; "
               f"ucuz ama hâlâ düşmekte olan 'değer tuzağı' hisseler alınmaz), "
               f"her aylık dilim {C.TRANCHE_MONTHS} ay tutulur; portföy tamamen hisse, tek hisse en fazla %{C.MAX_NAME_W * 100:.0f}.")

# ------------------------------------------------------------------ HİSSE ARA
with tab2:
    if last.empty:
        st.info("Henüz aylık tarama yok.")
    else:
        q = st.text_input("Hisse kodu", placeholder="örn. THYAO").strip().upper()
        all_t = sorted(set(last["ticker"].astype(str)) | set((pf.get("positions") or {}).keys()))
        if q:
            hits = [t for t in all_t if q in t]
            if not hits:
                st.markdown(f'<div class="card note">“{q}” bu ayın taramasında yok (likidite ya da fiyat geçmişi yetersiz).</div>',
                            unsafe_allow_html=True)
            else:
                t = q if q in hits else (hits[0] if len(hits) == 1 else st.selectbox("Eşleşenler", hits))
                row = last[last["ticker"] == t]
                in_pf = t in (pf.get("positions") or {})
                pending_buy = any(o["ticker"] == t and o["action"] == "BUY" for o in (pf.get("pending") or []))
                if row.empty:
                    st.markdown(f'<div class="card"><span class="tk">{t}</span> — bu ay puanlanmadı.</div>', unsafe_allow_html=True)
                else:
                    r = row.iloc[0]
                    sc = num(r.get("composite_pct")) or 0
                    picked = t in {x["ticker"] for x in (state.get("last_picks") or {}).get("picks", [])}
                    if in_pf:
                        badge, txt = "b-in", "✅ Portföyde"
                    elif pending_buy or picked:
                        badge, txt = "b-buy", "🟢 Alım listesinde"
                    elif sc >= 90:
                        badge, txt = "b-watch", "⚪ İzlemede"
                    else:
                        badge, txt = "b-no", "🔴 Şu an uygun değil"
                    why = []
                    if not in_pf and not pending_buy and not picked:
                        rank = int((last["composite"] > r["composite"]).sum()) + 1 if "composite" in last else None
                        if rank:
                            why.append(f"Bu ay {len(last)} hisse içinde {rank}. sırada; her ay yalnızca ilk {C.TRANCHE_N} hisse alınır.")
                        if (num(r.get("med_value_traded")) or 0) < C.MIN_MEDIAN_VALUE_TRADED_TL:
                            why.append("İşlem hacmi (likidite) alım eşiğinin altında.")
                    contrib = sorted(((k, weights.get(k, 0) * (num(r.get(f"z_{k}")) or 0)) for k in C.FACTORS),
                                     key=lambda x: x[1], reverse=True)
                    good = [FACTOR_TR[k] for k, v in contrib if v > 0.02][:3]
                    bad = [FACTOR_TR[k] for k, v in contrib[::-1] if v < -0.02][:3]
                    cf = num(r.get("confidence"))
                    html = (f'<div class="card"><div style="display:flex;justify-content:space-between;align-items:center">'
                            f'<span class="big" style="font-size:1.6rem">{t}</span><span class="badge {badge}">{txt}</span></div>'
                            f'<div class="mut" style="font-size:.82rem;margin-top:2px">{r.get("sector") if isinstance(r.get("sector"), str) else ""}</div>'
                            f'<div style="display:flex;gap:18px;margin-top:12px;flex-wrap:wrap">'
                            f'<div><div class="lbl">Skor</div><div class="kpi v">{sc:.0f}<span class="mut" style="font-size:.9rem">/100</span></div></div>'
                            f'<div><div class="lbl">Beklenen 12A</div><div class="kpi v">{pct(r.get("exp_nominal_12m"))}</div></div>'
                            f'<div><div class="lbl">Güven</div><div class="kpi v">{pct(cf * 100, nd=0) if cf is not None else "—"}</div></div>'
                            f'</div>')
                    if why:
                        html += '<div class="note" style="margin-top:12px">' + "<br>".join("• " + w for w in why) + "</div>"
                    if good:
                        html += f'<div class="note" style="margin-top:10px"><span class="pos">▲ Güçlü:</span> {", ".join(good)}</div>'
                    if bad:
                        html += f'<div class="note"><span class="neg">▼ Zayıf:</span> {", ".join(bad)}</div>'
                    st.markdown(html + "</div>", unsafe_allow_html=True)
                    h = snaps[snaps["ticker"] == t].sort_values("tarih")
                    if len(h) > 1:
                        st.caption("Aylık skor geçmişi")
                        st.line_chart(h.set_index("tarih")["composite_pct"], height=160)

# ------------------------------------------------------------------ PERFORMANS
with tab3:
    if len(nav) >= 2:
        n = nav.copy()
        n["tarih"] = pd.to_datetime(n["tarih"])
        n = n.set_index("tarih")
        ch = pd.DataFrame({"Portföy": n["nav"] / n["nav"].iloc[0]})
        if "xu100" in n and n["xu100"].notna().any():
            x = n["xu100"].ffill().bfill()
            ch["Russell 2000"] = x / x.iloc[0]
        st.line_chart(ch, height=220)
        if bt:
            rows = f'<div class="row"><span class="tk">Portföy</span><span class="{cls(navm.get("total_return_pct"))}" style="font-weight:700">{pct(navm.get("total_return_pct"), True)}</span></div>'
            for k in ("cpi", "usd", "gold", "deposit", "xu100"):
                if bt.get(k) is not None:
                    rows += f'<div class="row"><span>{BENCH[k]}</span><span class="mut">{pct(bt.get(k), True)}</span></div>'
            st.markdown(f'<div class="card"><div class="lbl">Başlangıçtan bu yana</div>{rows}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card note mut">Canlı performans grafiği en az iki işlem gününden sonra görünür.</div>',
                    unsafe_allow_html=True)
    if report:
        p = report.get("portfolio", {}) or {}
        rb = p.get("rolling12m_beat") or {}
        summ = report.get("hurdle_mode") == "sum"
        lines = [f'Test dönemi {p.get("start", "")[:4]}–{p.get("end", "")[:4]} · yıllık <b>{pct(p.get("cagr_pct"))}</b> '
                 f'(Russell 2000 {pct(p.get("xu100_cagr_pct"))}) · maks. düşüş {pct(p.get("max_drawdown_pct"))}']
        if rb:
            lines.append("12 aylık dönemlerde geçme oranı: " +
                         " · ".join(f'{BENCH.get(k, {"all": "Toplam hedef" if summ else "Hepsi", "each": "Hepsi tek tek"}.get(k, k))} <b>{pct(v, nd=0)}</b>'
                                    for k, v in rb.items()))
            if summ and p.get("rolling12m_median_gap_pp") is not None:
                lines.append(f'Tipik 12 ayda toplam hedefe uzaklık: <b>{pct(p.get("rolling12m_median_gap_pp"), True)}</b> puan')
        if report.get("universe_downloaded"):
            lines.append(f'Test evreni: <b>{report.get("universe_downloaded")}</b> hisse · her ay o tarihteki likidite eşiğiyle')
        elif p.get("rolling12m_beat_cpi_pct") is not None:
            lines.append(f'12 aylık dönemlerin <b>{pct(p.get("rolling12m_beat_cpi_pct"), nd=0)}</b>’inde TÜFE’yi geçti.')
        st.markdown('<div class="card"><div class="lbl">Geçmiş test (gerçek veri, dışarıda bırakılmış dönemler)</div>'
                    f'<div class="note">{"<br>".join(lines)}</div></div>', unsafe_allow_html=True)
        ru = report.get("rules") or {}
        if ru.get("engine"):
            st.markdown(f'<div class="card"><div class="lbl">Test edilen kural (canlı sistemle aynı kod)</div><div class="note">'
                        f'Her ay en güçlü <b>{ru.get("picks_per_month")}</b> hisse · her dilim <b>{ru.get("cohort_months")} ay</b> · '
                        f'portföyde ortalama <b>{str(report.get("avg_names_held", "—")).replace(".", ",")}</b> hisse · tek hisse en fazla '
                        f'%{(ru.get("max_name_weight") or 0) * 100:.0f} · %100 hisse. Kural sabittir, her yıl değişmez. '
                        f'Not: birkaç ayar bu geçmiş veriyle araştırılarak seçildi ve evrende batmış şirketler yok; '
                        f'bu yüzden gerçekçi beklenti test sonucunun yılda ~10 puan altıdır.</div></div>', unsafe_allow_html=True)
        cm = report.get("confidence_model") or {}
        if cm.get("oos_reliability"):
            rows = "".join(f'<div class="row"><span>Model %{x["predicted"] * 100:.0f} dedi</span>'
                           f'<span><b>gerçekleşen %{x["realised"] * 100:.0f}</b> <span class="mut">({x["n"]:,} gözlem)</span></span></div>'.replace(",", ".")
                           for x in cm["oos_reliability"])
            st.markdown(f'<div class="card"><div class="lbl">Güven oranı ne kadar doğru? (dışarıda bırakılmış yıllar)</div>{rows}'
                        f'<div class="note mut" style="margin-top:6px">Tahmin edilen ile gerçekleşen birbirine yakınsa güven oranına '
                        f'güvenilebilir. Yazı-turaya göre iyileşme: %{str(round(cm.get("oos_brier_skill_pct", 0) or 0, 1)).replace(".", ",")} (hisse seçiminde '
                        f'gerçekçi düzey).</div></div>', unsafe_allow_html=True)
        py = report.get("per_year") or {}
        if py:
            tbl = []
            for y, v in py.items():
                tbl.append({"Yıl": str(y), "Portföy": pct(v.get("nominal_pct"), True), "TÜFE": pct(v.get("cpi_pct")),
                            "Dolar": pct(v.get("usd_pct")), "Altın": pct(v.get("gold_pct")), "Mevduat": pct(v.get("deposit_pct")),
                            "Russell 2000": pct(v.get("xu100_pct"), True),
                            **({"Hedef (toplam)": pct(v.get("hurdle_pct")),
                                "Tek tek": {True: "✅", False: "❌"}.get(v.get("beat_each"), "—")} if summ else {}),
                            ("Hedefi geçti" if summ else "Hepsini geçti"): {True: "✅", False: "❌"}.get(v.get("beat_all"), "—")})
            st.dataframe(pd.DataFrame(tbl), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ SAĞLIK
with tab4:
    if not HS.get("checks"):
        st.markdown('<div class="card note mut">Sağlık raporu ilk günlük çalışmadan sonra oluşur.</div>', unsafe_allow_html=True)
    else:
        ov = HS.get("overall")
        txt = {"OK": "Her şey normal. Veriler güncel, hesaplamalar doğrulandı, canlı sonuçlar testle uyumlu.",
               "UYARI": "Dikkat edilmesi gereken bir şey var; sistem çalışıyor ama aşağıdaki sarı maddeleri izleyin.",
               "KRİTİK": "Sistemde ciddi bir sorun var. Kırmızı madde çözülene kadar önerileri uygulamadan önce kontrol edin."}.get(ov, "")
        st.markdown(f'<div class="card"><div class="lbl">Sistem sağlığı · {dstr(HS.get("date"))}</div>'
                    f'<div class="big">{HEM.get(ov, "⚪")} {ov}</div><div class="note mut" style="margin-top:6px">{txt}</div></div>',
                    unsafe_allow_html=True)
        rows = "".join(f'<div class="row"><div><b>{HEM.get(c["status"], "⚪")} {c["name"]}</b>'
                       f'<div class="mut" style="font-size:.82rem">{c["msg"]}</div></div></div>' for c in HS["checks"])
        st.markdown(f'<div class="card"><div class="lbl">Kontroller</div>{rows}</div>', unsafe_allow_html=True)

        lv = HS.get("live_vs_test") or {}
        if lv.get("status") == "OK" and lv.get("band_ret_pct"):
            b = lv["band_ret_pct"]
            pos_ = max(0.0, min(100.0, lv.get("percentile", 50)))
            bar = (f'<div class="cbar" style="position:relative;height:10px;background:linear-gradient(90deg,#f31260 0%,#f5a524 10%,'
                   f'#17c964 25%,#17c964 100%)"><div style="position:absolute;left:calc({pos_:.0f}% - 5px);top:-4px;width:10px;height:18px;'
                   f'border-radius:3px;background:#fff;border:2px solid #333"></div></div>')
            ex = (f'<br>Russell 2000\'e göre fark: canlı {pct(lv.get("live_excess_pp"), True)} · testte normal aralık '
                  f'{pct(lv["band_excess_pp"]["p10"], True)} … {pct(lv["band_excess_pp"]["p90"], True)}') if lv.get("band_excess_pp") else ""
            st.markdown(f'<div class="card"><div class="lbl">Canlı sonuç vs test (canlı backtest)</div>'
                        f'<div class="note">Portföy <b>{str(lv["months"]).replace(".", ",")}</b> aydır canlı: <b>{pct(lv["live_ret_pct"], True)}</b>. '
                        f'Testte aynı süreli dönemlerin %80\'i {pct(b["p10"], True)} ile {pct(b["p90"], True)} arasındaydı '
                        f'(ortanca {pct(b["p50"], True)}).{ex}</div>{bar}'
                        f'<div class="note mut" style="margin-top:6px">İşaret testteki dönemlerin %{lv.get("percentile", 0):.0f}\'inden '
                        f'iyi olduğunuzu gösterir. Kırmızı bölgeye (%5 altı) düşerse sistem testteki gibi davranmıyor demektir.</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="card note mut"><b>Canlı sonuç vs test:</b> portföy 1 aydan eskileşince, canlı getiri testte '
                        'aynı süreli bütün dönemlerle karşılaştırılıp "normal aralıkta mı" diye burada gösterilir.</div>',
                        unsafe_allow_html=True)

        w = HS.get("win_rates") or {}
        lab = {"closed_lots": "Kapanan pozisyon", "hit_nominal_pct": "Kârla kapanan %", "hit_beat_xu100_pct": "Russell 2000'ü geçen %",
               "hit_beat_cpi_pct": "TÜFE'yi geçen %", "hit_beat_all_pct": "Toplam hedefi geçen %",
               "avg_nominal_pct": "Ortalama getiri %", "avg_months_held": "Ortalama tutma (ay)"}
        tbl = [{"Ölçü": lab[k], "Canlı": ("—" if v.get("live") is None else str(v.get("live")).replace(".", ",")),
                "Test": ("—" if v.get("test") is None else str(v.get("test")).replace(".", ","))} for k, v in w.items() if k in lab]
        if tbl:
            st.caption("Kazanma oranları — canlı ve test")
            st.dataframe(pd.DataFrame(tbl), hide_index=True, use_container_width=True)

        cm = state.get("confidence_model_meta") or {}
        cl = state.get("confidence_live") or {}
        ic = (state.get("model") or {}).get("ic_live_12m") or {}
        gen = (report or {}).get("generated_at", "")[:10]
        nxt = (pd.Timestamp.now().normalize().replace(day=1) + pd.DateOffset(months=1)).replace(day=2)
        lines = [f'Model her ayın 2\'sinde tüm borsa verisiyle yeniden eğitilir · son: {dstr(gen) if gen else "—"} · sonraki: {dstr(nxt)}',
                 f'Faktör ağırlıkları canlı sonuçlarla güncellenir · canlı sinyal ölçümü: {ic.get("n_dates", 0)} ay'
                 + (f' (IC {str(ic.get("ic_mean")).replace(".", ",")})' if ic.get("n_dates") else ""),
                 f'Güven modeli: canlı sonuçların payı <b>%{(cm.get("live_weight") or 0) * 100:.0f}</b> '
                 f'({cm.get("n_live", 0)} sonuçlanmış gözlem; {2000} gözlemden sonra canlı veri devreye girer)']
        for hz_, lbl_ in (("3m", "3 ay sonra (erken gösterge)"), ("12m", "12 ay sonra")):
            c = cl.get(hz_) or {}
            if c.get("picks_n"):
                lines.append(f'Öneriler {lbl_}: tahmin %{c["picks_pred"] * 100:.0f} → gerçekleşen <b>%{c["picks_real"] * 100:.0f}</b> '
                             f'({c["picks_n"]} öneri)')
        st.markdown(f'<div class="card"><div class="lbl">Kendini geliştirme</div><div class="note">{"<br>".join(lines)}</div></div>',
                    unsafe_allow_html=True)

with st.expander("Teknik detay"):
    m = state.get("model", {}) or {}
    st.json({"model": m.get("status"), "sürüm": m.get("champion_version"), "kalibrasyon": cal.get("status"),
             "dilimler": pf.get("cohorts"), "hedef": hz, "TÜFE": inf, "nakit": state.get("cash_rate"),
             "rejim olasılıkları": reg.get("probs"), "sağlık": HS.get("checks"),
             "canlı IC": {"12A": m.get("ic_live_12m"), "3A": m.get("ic_live_3m")}, "kapanan pozisyonlar": perf.get("lots")})
