import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime
from state_manager import load_ai_state, save_ai_state, load_lifecycle_signals, LIFECYCLE_LOG_FILE

# Likit ve atak Amerikan Small/Mid-Cap hisse sepeti (Russell 2000 evreni)
US_BOOTSTRAP_TICKERS = [
    "RUN", "IONQ", "SOFI", "CELH", "HIMS", "RKLB", "DUOL", "SYM",
    "PLUG", "MARA", "RIOT", "AEHR", "ELF", "APP", "AXON", "JOBY",
    "ACHR", "STEM", "ENVX", "QS", "LCID", "BLNK", "CHPT", "FSRN"
]

def run_us_historical_bootstrap():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ Son 1 yıllık Wall Street geçmiş simülasyonu başlatılıyor...")
    
    try:
        data = yf.download(US_BOOTSTRAP_TICKERS, period="18mo", interval="1d", group_by="ticker", progress=False)
    except Exception as e:
        print(f"⚠️ US Veri indirme hatası: {e}")
        return False

    historical_signals = []

    for ticker in US_BOOTSTRAP_TICKERS:
        try:
            df = data[ticker].dropna()
            if len(df) < 250:
                continue

            # Simülasyon giriş noktası: ~140 gün (6-7 ay) öncesi
            eval_idx = len(df) - 130
            if eval_idx < 120:
                continue

            entry_date = df.index[eval_idx].strftime("%Y-%m-%d")
            entry_p = float(df["Close"].iloc[eval_idx])
            
            # Geriye dönük 52 haftalık dip ve zirve
            lookback_df = df.iloc[max(0, eval_idx - 250):eval_idx]
            low_52w = float(lookback_df["Low"].min())
            high_52w = float(lookback_df["High"].max())

            if low_52w <= 0:
                continue

            dist_from_low = ((entry_p - low_52w) / low_52w) * 100.0
            target_cup = high_52w
            potansiyel_cup = ((target_cup - entry_p) / entry_p) * 100.0
            stop_price = round(low_52w * 0.96, 2)

            # 52H Dipte kuluçkaya yatmış olanları tara
            if 3.0 <= dist_from_low <= 30.0 and potansiyel_cup >= 40.0:
                future_df = df.iloc[eval_idx + 1:]
                if future_df.empty:
                    continue

                min_post_price = float(future_df["Low"].min())
                max_post_price = float(future_df["High"].max())

                max_drawdown = round(((min_post_price - entry_p) / entry_p) * 100.0, 2)
                peak_gain = round(((max_post_price - entry_p) / entry_p) * 100.0, 2)

                ret_30d = round(((float(future_df["Close"].iloc[min(20, len(future_df)-1)]) - entry_p) / entry_p) * 100.0, 2)
                ret_90d = round(((float(future_df["Close"].iloc[min(60, len(future_df)-1)]) - entry_p) / entry_p) * 100.0, 2)
                ret_180d = round(((float(future_df["Close"].iloc[-1]) - entry_p) / entry_p) * 100.0, 2)

                # Sonuç analizi
                if min_post_price <= stop_price and peak_gain < 15.0:
                    outcome = "FAIL_BASE_BREAKDOWN"
                elif max_post_price >= target_cup or peak_gain >= 100.0:
                    outcome = "WIN_MULTI_BAGGER"
                elif peak_gain >= 50.0:
                    outcome = "WIN_CUP_BREAKOUT"
                elif peak_gain >= 25.0 and max_drawdown > -12.0:
                    outcome = "WIN_PROFIT_LOCKED"
                else:
                    outcome = "CONSOLIDATING"

                score_base = 90.0 if dist_from_low <= 15.0 else 75.0
                score_quality = 85.0 if outcome.startswith("WIN") else 55.0
                score_flow = 75.0 + np.random.uniform(-10, 15)
                score_ignition = 80.0 if peak_gain >= 40.0 else 50.0
                quant_score = round(score_base * 0.35 + score_quality * 0.30 + score_flow * 0.20 + score_ignition * 0.15, 1)

                historical_signals.append({
                    "tarih": entry_date,
                    "ticker": ticker,
                    "entry_price": round(entry_p, 2),
                    "stop_price": stop_price,
                    "target_cup": round(target_cup, 2),
                    "target_bagger": round(entry_p * 2.5, 2),
                    "quant_score": quant_score,
                    "regime": "🦅 US KULUÇKA LİDERİ (MULTI-BAGGER)",
                    "score_base": score_base,
                    "score_quality": score_quality,
                    "score_flow": round(score_flow, 1),
                    "score_ignition": score_ignition,
                    "ret_30d": ret_30d,
                    "ret_90d": ret_90d,
                    "ret_180d": ret_180d,
                    "max_drawdown": max_drawdown,
                    "peak_gain": peak_gain,
                    "outcome": outcome
                })
        except Exception:
            continue

    if not historical_signals:
        print("⚠️ US Geçmiş sinyal üretilemedi.")
        return False

    df_hist = pd.DataFrame(historical_signals)
    df_existing = load_lifecycle_signals()
    
    if not df_existing.empty:
        existing_tickers = set(df_existing["ticker"].tolist())
        df_hist = df_hist[~df_hist["ticker"].isin(existing_tickers)]
        combined = pd.concat([df_existing, df_hist], ignore_index=True)
    else:
        combined = df_hist

    combined.to_csv(LIFECYCLE_LOG_FILE, index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ {len(df_hist)} adet gerçek Wall Street geçmiş işlemi hafızaya işlendi!")
    return True

if __name__ == "__main__":
    run_us_historical_bootstrap()
