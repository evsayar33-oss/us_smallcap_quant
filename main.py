import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings

from state_manager import load_ai_state, load_lifecycle_signals, LIFECYCLE_LOG_FILE
from longterm_auditor import audit_and_calibrate

warnings.filterwarnings('ignore')

GECMIS_DOSYA = "gecmis_veri.csv"

# =============================================================================
# 1. TRADINGVIEW ABD PİYASA, SEKTÖR VE ESAS FAALİYET KÂRI VERİSİ
# =============================================================================

def get_us_smallcap_data():
    url = "https://scanner.tradingview.com/america/scan"
    payload = {
        "filter": [
            {"left": "type", "operation": "equal", "right": "stock"},
            {"left": "exchange", "operation": "in_range", "right": ["AMEX", "NASDAQ", "NYSE"]},
            {"left": "Value.Traded", "operation": "greater", "right": 2000000},
            {"left": "market_cap_basic", "operation": "in_range", "right": [250000000, 6000000000]} # $250M - $6B
        ],
        "columns": [
            "name", "close", "open", "high", "low", "volume", "change", "Value.Traded",
            "price_52_week_high",
            "price_52_week_low",
            "market_cap_basic",
            "return_on_equity_fq",
            "price_earnings_ttm",
            "price_book_fq",
            "Perf.Y",
            "relative_volume_10d_calc",
            "Perf.1M",
            "Perf.W",
            "sector",
            "industry",
            "operating_margin" # 🛡️ ESAS FAALİYET MARJI KALKANI
        ],
        "sort": {"sortBy": "Value.Traded", "sortOrder": "desc"},
        "range": [0, 500]
    }
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=15)
        data = res.json()
        rows = []
        for item in data.get("data", []):
            d = item["d"]
            close_p = float(d[1]) if d[1] is not None else 0.0
            high_p = float(d[3]) if d[3] is not None else close_p
            low_p = float(d[4]) if d[4] is not None else close_p
            
            rows.append({
                "ticker": d[0],
                "close": close_p,
                "open": float(d[2]) if d[2] is not None else close_p,
                "high": high_p,
                "low": low_p,
                "volume": float(d[5]) if d[5] is not None else 0.0,
                "change_%": float(d[6]) if d[6] is not None else 0.0,
                "value_traded": float(d[7]) if d[7] is not None else 0.0,
                "high_52w": float(d[8]) if d[8] is not None else close_p * 1.5,
                "low_52w": float(d[9]) if d[9] is not None else close_p * 0.7,
                "market_cap": float(d[10]) if d[10] is not None else 1000000000.0,
                "roe": float(d[11]) if d[11] is not None else 12.0,
                "pe": float(d[12]) if d[12] is not None else 15.0,
                "pb": float(d[13]) if d[13] is not None else 2.0,
                "perf_y": float(d[14]) if d[14] is not None else 0.0,
                "rvol": float(d[15]) if len(d) > 15 and d[15] is not None else 1.0,
                "perf_1m": float(d[16]) if len(d) > 16 and d[16] is not None else 0.0,
                "perf_w": float(d[17]) if len(d) > 17 and d[17] is not None else 0.0,
                "sector": str(d[18]) if len(d) > 18 and d[18] is not None else "Genel",
                "industry": str(d[19]) if len(d) > 19 and d[19] is not None else "Genel",
                "oper_margin": float(d[20]) if len(d) > 20 and d[20] is not None else 10.0,
                "tarih": pd.Timestamp.now().normalize()
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"⚠️ US Piyasa Verisi Hatası: {e}")
        return pd.DataFrame()

# =============================================================================
# 2. PİYASA DEĞERİ KADEMELERİ (USD) VE DİNAMİK EŞİK FONKSİYONU
# =============================================================================

def get_us_tier_thresholds(mcap, thresholds):
    """
    ABD piyasasında piyasa değerine ($250M - $1B Micro, $1B - $3B Small, $3B - $6B SMID)
    göre optimize edilen dinamik eşikleri döner.
    """
    tiers = thresholds.get("market_cap_tiers", {})
    if not tiers:
        return {
            "tier_name": "CORE_US",
            "min_roe": thresholds.get("min_roe", 12.0),
            "min_oper_margin": thresholds.get("min_oper_margin", 5.0),
            "min_dist_from_52w_low": 3.0,
            "max_dist_from_52w_low": 25.0,
            "ideal_pe_max": 18.0,
            "acceptable_pe_max": 30.0,
            "stop_loss_pct": thresholds.get("macro_stop_loss_pct", -12.0),
            "target_cup_min": 45.0,
            "max_patience_days": 90
        }

    if mcap < 1_000_000_000:
        cfg = tiers.get("micro_cap", {})
        tier_name = "US_MICRO_CAP"
    elif mcap < 3_000_000_000:
        cfg = tiers.get("small_cap", {})
        tier_name = "US_SMALL_CAP"
    else:
        cfg = tiers.get("mid_cap", {})
        tier_name = "US_SMID_CAP"

    return {
        "tier_name": tier_name,
        "min_roe": cfg.get("min_roe", 10.0),
        "min_oper_margin": cfg.get("min_oper_margin", 5.0),
        "min_dist_from_52w_low": cfg.get("min_dist_from_52w_low", 2.5),
        "max_dist_from_52w_low": cfg.get("max_dist_from_52w_low", 30.0),
        "ideal_pe_max": cfg.get("ideal_pe_max", 20.0),
        "acceptable_pe_max": cfg.get("acceptable_pe_max", 32.0),
        "stop_loss_pct": cfg.get("stop_loss_pct", -14.0),
        "target_cup_min": cfg.get("target_cup_min", 40.0),
        "max_patience_days": cfg.get("max_patience_days", 75)
    }

# =============================================================================
# 3. REJİM VE KADEME DUYARLI RUSSELL 2000 QUANT MOTORU
# =============================================================================

def calculate_us_quant_scores(df, df_gecmis, state):
    if df.empty:
        return df

    thresholds = state.get("thresholds", {})

    # 1. WALL STREET PİYASA REJİMİ TESPİTİ
    market_perf_median = float(df['perf_1m'].median())
    if market_perf_median >= 0.0:
        market_regime = "BOĞA / GENİŞLEME"
        # Boğada momentum ödüllendirilir
        weights = {"macro_base": 0.35, "growth_quality": 0.25, "volume_flow": 0.20, "ignition": 0.20}
    else:
        market_regime = "AYI / DURGUNLUK"
        # Ayıda sahte kırılımlara karşı DEFANS MODU (Kârlılık %40 yapılır)
        weights = {"macro_base": 0.35, "growth_quality": 0.40, "volume_flow": 0.20, "ignition": 0.05}

    state["market_regime"] = market_regime
    scored_data = []

    for idx, row in df.iterrows():
        item = row.to_dict()
        close = float(item.get('close', 0.0))
        high = float(item.get('high', close))
        low = float(item.get('low', close))
        change = float(item.get('change_%', 0.0))
        rvol = float(item.get('rvol', 1.0))
        
        high_52w = float(item.get('high_52w', close * 1.5))
        low_52w = float(item.get('low_52w', close * 0.7))
        mcap = float(item.get('market_cap', 1000000000.0))
        roe = float(item.get('roe', 12.0))
        pe = float(item.get('pe', 15.0))
        pb = float(item.get('pb', 2.0))
        perf_y = float(item.get('perf_y', 0.0))
        perf_1m = float(item.get('perf_1m', 0.0))
        perf_w = float(item.get('perf_w', 0.0))
        sector = item.get('sector', '')
        industry = item.get('industry', '')
        oper_margin = float(item.get('oper_margin', 10.0))

        # Piyasa Değeri Katmanına Göre Dinamik Eşikleri Al
        tier_cfg = get_us_tier_thresholds(mcap, thresholds)
        t_min_roe = tier_cfg["min_roe"]
        t_min_margin = tier_cfg["min_oper_margin"]
        t_min_dist = tier_cfg["min_dist_from_52w_low"]
        t_max_dist = tier_cfg["max_dist_from_52w_low"]
        t_ideal_pe = tier_cfg["ideal_pe_max"]
        t_accept_pe = tier_cfg["acceptable_pe_max"]
        t_stop_pct = tier_cfg["stop_loss_pct"]
        t_target_cup = tier_cfg["target_cup_min"]

        dist_from_52w_low = ((close - low_52w) / (low_52w + 1e-9)) * 100.0 if low_52w > 0 else 0.0
        target_cup = round(high_52w, 2)
        target_bagger = round(close * 2.50, 2)
        stop_price = round(min(low_52w * 0.96, close * (1.0 + (t_stop_pct / 100.0))), 2)
        potansiyel_cup = round(((target_cup - close) / close) * 100.0, 1)

        # 🛡️ 1. ESAS FAALİYET KÂRI KALKANI (KADEMELİ)
        is_fake_profit = (oper_margin < t_min_margin)

        # 🛡️ 2. BİYOTEKNOLOJİ / FDA KUMAR KALKANI
        is_binary_biotech = False
        if "biotechnology" in industry.lower() or "pharmaceuticals" in industry.lower():
            if pe <= 0 or pe > 35.0 or roe < 20.0 or oper_margin < 10.0:
                is_binary_biotech = True

        # 🛡️ 3. RÖLATİF DÜŞEN BIÇAK
        rel_perf_1m = perf_1m - market_perf_median
        is_falling_knife = False
        if rel_perf_1m < -14.0:
            is_falling_knife = True
        elif close <= low_52w * 1.004:
            is_falling_knife = True

        # 4. Zombi ve Aşırı Prim Filtresi
        is_zombie = (roe < t_min_roe) or (pb <= 0.0) or is_fake_profit
        is_overextended = (perf_y > 150.0) or (dist_from_52w_low > (t_max_dist * 1.5))

        # 1. Kademeli Makro Taban Skoru
        score_base = 20.0
        if t_min_dist <= dist_from_52w_low <= t_max_dist:
            score_base = 90.0
            if potansiyel_cup >= t_target_cup:
                score_base = 100.0
        elif dist_from_52w_low <= (t_max_dist * 1.25):
            score_base = 65.0

        # 2. Kademeli Kalite Skoru
        score_quality = 30.0
        if roe >= (t_min_roe * 1.8) and oper_margin >= (t_min_margin * 1.8): score_quality += 45.0
        elif roe >= (t_min_roe * 1.3) and oper_margin >= (t_min_margin * 1.3): score_quality += 30.0
        elif roe >= t_min_roe and oper_margin >= t_min_margin: score_quality += 15.0

        if 0 < pe <= t_ideal_pe: score_quality += 25.0
        elif 0 < pe <= t_accept_pe: score_quality += 10.0
        score_quality = min(max(score_quality, 5.0), 100.0)

        # 3. Kapanış Gücü & Hacim Akışı
        range_span = high - low
        clv = ((close - low) - (high - close)) / range_span if range_span > 0 else 0.0
        score_flow = round(min(max((max(clv, 0.0) * 70.0) + (min(rvol, 3.0) * 10.0), 10.0), 98.0), 1)

        # 4. Hacimli Ateşleme
        score_ignition = round(min(max((rvol * 35.0) + (max(change, 0.0) * 5.0), 10.0), 100.0), 1)

        item['tier'] = tier_cfg['tier_name']
        item['dist_from_52w_low'] = round(dist_from_52w_low, 1)
        item['stop_price'] = stop_price
        item['target_cup'] = target_cup
        item['target_bagger'] = target_bagger
        item['potansiyel_cup'] = potansiyel_cup
        item['mcap_milyon'] = round(mcap / 1000000.0, 1)
        item['score_base'] = score_base
        item['score_quality'] = score_quality
        item['score_flow'] = score_flow
        item['score_ignition'] = score_ignition
        item['is_disqualified'] = is_zombie or is_overextended or is_falling_knife or is_binary_biotech or is_fake_profit
        item['is_biotech'] = is_binary_biotech
        item['is_knife'] = is_falling_knife
        item['is_fake'] = is_fake_profit
        scored_data.append(item)

    res_df = pd.DataFrame(scored_data)
    if res_df.empty:
        return res_df

    res_df['pct_base'] = res_df['score_base'].rank(pct=True) * 100.0
    res_df['pct_qual'] = res_df['score_quality'].rank(pct=True) * 100.0
    res_df['pct_flow'] = res_df['score_flow'].rank(pct=True) * 100.0
    res_df['pct_ign'] = res_df['score_ignition'].rank(pct=True) * 100.0

    w_b = weights["macro_base"]
    w_q = weights["growth_quality"]
    w_f = weights["volume_flow"]
    w_i = weights["ignition"]

    raw_score = np.round(
        res_df['pct_base'] * w_b +
        res_df['pct_qual'] * w_q +
        res_df['pct_flow'] * w_f +
        res_df['pct_ign'] * w_i,
        1
    )

    res_df['quant_score'] = np.where(
        (res_df['change_%'] > 0.0) & (~res_df['is_disqualified']),
        raw_score,
        0.0
    )

    conditions = [
        res_df['is_fake'],
        res_df['is_biotech'],
        res_df['is_knife'],
        res_df['is_disqualified'],
        (res_df['quant_score'] >= 65.0) & (res_df['potansiyel_cup'] >= 40.0),
        (res_df['quant_score'] >= 50.0)
    ]
    choices = [
        "⚠️ SAHTE KÂR (FAALİYET KÂRI YETERSİZ)",
        "⚠️ BİYOTEK TUZAĞI (FDA RİSKİ)",
        "🪤 DÜŞEN BIÇAK (RÖLATİF ÇÖKÜŞ)",
        "⚠️ ELENDİ (ZOMBİ VEYA PRİMLİ)",
        "🦅 US KULUÇKA LİDERİ (MULTI-BAGGER)",
        "⚡ TABAN BİRİKTİRME (TAKİP)"
    ]
    res_df['regime'] = np.select(conditions, choices, default="NÖTR")

    drop_cols = ['pct_base', 'pct_qual', 'pct_flow', 'pct_ign', 'is_disqualified', 'is_biotech', 'is_knife', 'is_fake']
    res_df = res_df.drop(columns=[col for col in drop_cols if col in res_df.columns])

    res_df['score_diff'] = 0.0
    if not df_gecmis.empty and 'quant_score' in df_gecmis.columns:
        son_tarih = df_gecmis['tarih'].max()
        df_son = df_gecmis[df_gecmis['tarih'] == son_tarih]
        eski_map = dict(zip(df_son['ticker'], df_son['quant_score']))
        res_df['score_diff'] = np.round(res_df['quant_score'] - res_df['ticker'].map(eski_map).fillna(res_df['quant_score']), 1)

    return res_df.sort_values(by='quant_score', ascending=False).reset_index(drop=True)

# =============================================================================
# 4. YAŞAM DÖNGÜSÜ GÜNLÜĞÜ
# =============================================================================

def log_lifecycle_signals(df_scored, state):
    try:
        leaders = df_scored[df_scored['regime'].str.contains("US KULUÇKA LİDERİ")].head(6)
        if leaders.empty:
            return

        today = pd.Timestamp.now().normalize()
        history_df = load_lifecycle_signals()

        new_entries = []
        for _, row in leaders.iterrows():
            new_entries.append({
                "tarih": today,
                "ticker": row["ticker"],
                "entry_price": float(row["close"]),
                "stop_price": float(row["stop_price"]),
                "target_cup": float(row["target_cup"]),
                "target_bagger": float(row["target_bagger"]),
                "last_seen_price": float(row["close"]),
                "quant_score": float(row["quant_score"]),
                "regime": row["regime"],
                "score_base": float(row.get("score_base", 0)),
                "score_quality": float(row.get("score_quality", 0)),
                "score_flow": float(row.get("score_flow", 0)),
                "score_ignition": float(row.get("score_ignition", 0)),
                "ret_30d": np.nan,
                "ret_90d": np.nan,
                "ret_180d": np.nan,
                "max_drawdown": 0.0,
                "peak_gain": 0.0,
                "outcome": "INCUBATING"
            })

        df_new = pd.DataFrame(new_entries)
        if not history_df.empty:
            existing_tickers = set(history_df[history_df["outcome"].isin(["INCUBATING", "PENDING"])]["ticker"].tolist())
            df_to_add = df_new[~df_new["ticker"].isin(existing_tickers)]
            if not df_to_add.empty:
                combined = pd.concat([history_df, df_to_add], ignore_index=True)
                combined.to_csv(LIFECYCLE_LOG_FILE, index=False)
        else:
            df_new.to_csv(LIFECYCLE_LOG_FILE, index=False)
    except Exception as e:
        print(f"⚠️ US Sinyal günlüğü hatası: {e}")

# =============================================================================
# 5. TELEGRAM VE RAPORLAMA SİSTEMİ
# =============================================================================

def send_telegram(message):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("CHAT_ID")
    if not token or not chat_id:
        print("ℹ️ US Telegram bilgileri eksik, terminale yazdırılıyor.")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"⚠️ US Telegram hatası: {e}")
        return False

def format_telegram_report(df_scored, state, exit_alerts):
    regime = state.get("market_regime", "BOĞA / GENİŞLEME")
    audit = state.get("audit_summary", {})
    
    tarih_str = datetime.now().strftime("%d.%m.%Y")
    regime_icon = "🟢" if "BOĞA" in regime else "🔴"
    
    msg = f"🦅 <b>WALL STREET MULTI-BAGGER TERMINAL</b> | <code>{tarih_str}</code>\n"
    msg += f"───────────────────────\n"
    msg += f"🧭 Wall Street Rejimi: {regime_icon} <b>{regime}</b>\n"
    msg += f"🧠 Model Durumu: <b>{audit.get('status', 'Optimizasyon Tamamlandı')}</b>\n"
    msg += f"🏆 6 Aylık Win Rate: <b>%{audit.get('win_rate_6m', 0.0):.1f}</b>\n"
    msg += f"───────────────────────\n\n"

    if exit_alerts:
        msg += "🚨 <b>DİNAMİK RİSK VE ÇIKIŞ UYARILARI (USD)</b>\n"
        for alert in exit_alerts:
            msg += f"• <b>#{alert['ticker']}</b>: {alert['msg']}\n"
        msg += "\n"

    leaders = df_scored[df_scored['regime'].str.contains("US KULUÇKA LİDERİ")].head(5)
    if not leaders.empty:
        msg += "💎 <b>GÜNÜN US KULUÇKA LİDERLERİ (Multi-Bagger Adayları)</b>\n"
        msg += "<i>(Piyasa Değeri Kademesi, Taban & Esas Faaliyet Kâr Teyitli)</i>\n\n"
        
        for idx, row in leaders.iterrows():
            s_diff = row.get('score_diff', 0.0)
            fark_str = f"+{s_diff:.1f}" if s_diff > 0 else f"{s_diff:.1f}"
            tier_label = row.get('tier', 'US_SMALL_CAP')
            
            msg += f"⭐ <b>#{row['ticker']}</b> [{tier_label}] ── <b>Skor: {row['quant_score']:.1f}</b> <i>({fark_str})</i>\n"
            msg += f"💵 Fiyat: <b>${row['close']:.2f}</b> (PD: <b>${row['mcap_milyon']:.0f}M</b> | {row.get('sector', 'N/A')})\n"
            msg += f"📊 ROE: <b>%{row.get('roe', 0):.1f}</b> | F/K: <b>{row.get('pe', 0):.1f}</b> | Faaliyet Marjı: <b>%{row.get('oper_margin', 0):.1f}</b>\n"
            msg += f"🎯 1. Çanak Hedefi: <b>${row['target_cup']:.2f}</b> (Potansiyel: <b>+%{row['potansiyel_cup']:.1f}</b>)\n"
            msg += f"🚀 2. Multi-Bagger: <b>${row['target_bagger']:.2f}</b> (+%150)\n"
            msg += f"🛡️ Taban Stop: <b>${row['stop_price']:.2f}</b> | 52H Dip Farkı: <b>%{row['dist_from_52w_low']:.1f}</b>\n"
            msg += f"───────────────────────\n"
    else:
        msg += "ℹ️ Bugün tüm Wall Street kuluçka ve kalite filtrelerini geçen yeni hisse bulunamadı (USD Nakit Koruma).\n"

    msg += "\n<i>Not: Yatırım tavsiyesi değildir. Russell 2000 Quant Kuluçka Modeli çıktısıdır.</i>"
    return msg

# =============================================================================
# 6. ANA YÜRÜTÜCÜ
# =============================================================================

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🦅 Wall Street Quant Kuluçka Motoru Başlatılıyor...")

    state = load_ai_state()
    df_gecmis = pd.DataFrame()
    if os.path.exists(GECMIS_DOSYA):
        try:
            df_gecmis = pd.read_csv(GECMIS_DOSYA)
            if 'tarih' in df_gecmis.columns:
                df_gecmis['tarih'] = pd.to_datetime(df_gecmis['tarih'])
        except Exception:
            pass

    # 1. Denetçi ve Öğrenme Döngüsünü Çalıştır
    state, exit_alerts = audit_and_calibrate()

    # 2. ABD Piyasa Verilerini Çek
    df_current = get_us_smallcap_data()
    if df_current.empty:
        print("⚠️ Güncel US verisi çekilemedi, işlem sonlandırılıyor.")
        return

    # 3. Kademeli US Quant Puanlarını Hesapla
    df_scored = calculate_us_quant_scores(df_current, df_gecmis, state)

    # 4. Sinyalleri Kaydet
    log_lifecycle_signals(df_scored, state)

    # 5. Geçmiş Veriyi Güncelle
    if not df_gecmis.empty:
        df_yeni = pd.concat([df_gecmis, df_scored], ignore_index=True)
    else:
        df_yeni = df_scored
    df_yeni.to_csv(GECMIS_DOSYA, index=False)

    # 6. Telegram Raporu Gönder
    report = format_telegram_report(df_scored, state, exit_alerts)
    send_telegram(report)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ US Quant Güncellemesi Başarıyla Tamamlandı!")

if __name__ == "__main__":
    main()
