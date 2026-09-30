"""Haftalık sade özet (salt okunur). Canlı sonuçları motorun state dosyasından okur."""
from __future__ import annotations

import config as C
import telegram_report as TG
from main import send_telegram
from state_manager import load_state


def audit() -> str:
    st = load_state()
    perf = st.get("performance", {}) or {}
    nav, lots = perf.get("nav", {}) or {}, perf.get("lots", {}) or {}
    L = [f"🗓 <b>ABD Small-Cap</b> · Haftalık Özet", ""]
    L += TG.hurdle_block(st.get("hurdles") or {})
    L.append("")
    L += TG.portfolio_line(st)
    rb = nav.get("rolling12m_beat") or {}
    if rb:
        allk = "Toplam hedef" if getattr(C, "HURDLE_MODE", "max") == "sum" else "Hepsi"
        L.append("🎯 12 aylık dönemlerde geçme: " + " · ".join(
            f"{TG.BENCH_TR.get(k, {'all': allk, 'each': 'Hepsi tek tek'}.get(k, k))} {TG.pct(v, nd=0)}" for k, v in rb.items()))
    if lots.get("closed_lots"):
        L.append(f"📦 Kapanan {lots['closed_lots']} pozisyon · hedefi geçen {TG.pct(lots.get('hit_beat_all_pct'), nd=0)}"
                 f" · ort. {lots.get('avg_months_held')} ay")
    pos = (st.get("portfolio") or {}).get("positions", {}) or {}
    if pos:
        best = sorted(pos.items(), key=lambda kv: -kv[1]["level"])[:3]
        L.append("🏆 En iyiler: " + " · ".join(f"{t} {TG.pct((p['level'] - 1) * 100, True)}" for t, p in best))
        flagged = [t for t, p in pos.items() if p.get("dd_flag")]
        if flagged:
            L.append("⚠️ Sert düşüş (bilgi): " + " · ".join(flagged))
    L.append(f"🧭 Piyasa: {TG.REGIME_TR.get((st.get('regime') or {}).get('label'), '—')}")
    hl = TG.health_lines(st, full=True)
    if hl:
        L += [""] + hl
    return "\n".join(L)


if __name__ == "__main__":
    send_telegram(audit())
