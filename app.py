import streamlit as st
import pandas as pd
import numpy as np
import os
import json

st.set_page_config(
    page_title="Wall Street Multi-Bagger Terminal (USD)",
    layout="wide",
    page_icon="🦅"
)

# =============================================================================
# VERİ YÜKLEME FONKSİYONLARI
# =============================================================================

@st.cache_data(ttl=60)
def load_historical_data():
    if os.path.exists("gecmis_veri.csv"):
        try:
            df = pd.read_csv("gecmis_veri.csv")
            if 'tarih' in df.columns:
                df['tarih'] = pd.to_datetime(df['tarih'])
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_ai_state():
    if os.path.exists("us_ai_state.json"):
        try:
            with open("us_ai_state.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

@st.cache_data(ttl=60)
def load_lifecycle_signals():
    if os.path.exists("signals_lifecycle.csv"):
        try:
            df = pd.read_csv("signals_lifecycle.csv")
            df['tarih'] = pd.to_datetime(df['tarih'])
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

df_gecmis = load_historical_data()
ai_state = load_ai_state()
df_lifecycle = load_lifecycle_signals()

# =============================================================================
# ÜST BİLGİ VE METRİK KARTLARI
# =============================================================================

st.title("🦅 Wall Street Multi-Bagger & Russell 2000 Terminal")
st.markdown("*ABD piyasalarında (NYSE & NASDAQ) 52 haftalık dipte kuluçkaya yatmış; **kârlı ve %100 - %250 USD potansiyeli taşıyan Small-Cap** hisseleri bulan Quant Motoru.*")

col1, col2, col3, col4 = st.columns(4)
weights = ai_state.get("weights", {"macro_base": 0.35, "growth_quality": 0.30, "volume_flow": 0.20, "ignition": 0.15})
audit = ai_state.get("audit_summary", {})

with col1:
    st.metric("🤖 US AI Durumu", "Aktif", audit.get("status", "Kuluçka Takibinde")[:22] + "...")
with col2:
    st.metric("🏆 6 Aylık Win Rate", f"%{audit.get('win_rate_6m', 0.0):.1f}", f"Denetlenen: {audit.get('total_signals_audited', 0)}")
with col3:
    st.metric("🎯 Getiri Hedefi", "%100 - %250 USD", "Saf Dolar Bazlı")
with col4:
    last_date = audit.get("last_audit_date", "-")
    st.metric("🗓️ Son Model Güncellemesi", str(last_date))

st.divider()

# =============================================================================
# YAN PANEL (SIDEBAR) - US HİSSE SORGULAMA
# =============================================================================

st.sidebar.header("🔍 US Hisse Sorgu (Ticker)")
search_ticker = st.sidebar.text_input("Hisse Kodu (Örn: SOFI, IONQ):").upper().strip()

if not df_gecmis.empty:
    son_tarih = df_gecmis['tarih'].max()
    df_latest = df_gecmis[df_gecmis['tarih'] == son_tarih].copy()
    
    if search_ticker:
        h_data = df_latest[df_latest['ticker'] == search_ticker]
        if not h_data.empty:
            score = float(h_data['quant_score'].iloc[0])
            regime = h_data['regime'].iloc[0]
            d_52w = float(h_data.get('dist_from_52w_low', 0.0).iloc[0])
            mcap = float(h_data.get('mcap_milyon', 0.0).iloc[0])
            target_1 = float(h_data.get('target_cup', 0.0).iloc[0])
            target_2 = float(h_data.get('target_bagger', 0.0).iloc[0])
            roe = float(h_data.get('roe', 0.0).iloc[0])
            pe = float(h_data.get('pe', 0.0).iloc[0])
            price = float(h_data.get('close', 0.0).iloc[0])

            st.sidebar.metric(f"{search_ticker} Kuluçka Skoru", f"{score:.1f}")
            st.sidebar.write(f"**Fiyat:** ${price:.2f}")
            st.sidebar.write(f"**Durum:** {regime}")
            st.sidebar.write(f"**Piyasa Değeri:** ${mcap:.0f}M")
            st.sidebar.write(f"**52H Dip Mesafesi:** %{d_52w:+.1f}")
            st.sidebar.write(f"**ROE:** %{roe:.1f} | **F/K:** {pe:.1f}")
            st.sidebar.write(f"🎯 **1. Çanak Hedefi:** ${target_1:.2f}")
            st.sidebar.write(f"🚀 **2. Multi-Bagger:** ${target_2:.2f}")

            trend = df_gecmis[df_gecmis['ticker'] == search_ticker][['tarih', 'quant_score']].sort_values('tarih')
            if not trend.empty:
                trend.set_index('tarih', inplace=True)
                st.sidebar.line_chart(trend['quant_score'])
        else:
            st.sidebar.warning("Hisse son US taramasında bulunamadı.")

# =============================================================================
# SEKMELER
# =============================================================================

tab_leads, tab_ai, tab_risks = st.tabs([
    "💎 US Kuluçka Liderleri",
    "🧠 Quant AI & Portföy Takip Defteri",
    "🏢 Elenen Hisseler (Zombiler & Devler)"
])

# -----------------------------------------------------------------------------
# 1. SEKME: KULUÇKA LİDERLERİ
# -----------------------------------------------------------------------------
with tab_leads:
    st.subheader("💎 52 Haftalık Dipte Kuluçkaya Yatan US Şirketleri")
    st.markdown("*Piyasa değeri $250M - $6B arası, ROE'si pozitif ve 1 yıllık dip desteğinde hacimlenen atak Wall Street şirketleri.*")
    
    if not df_gecmis.empty:
        leaders = df_latest[
            df_latest['regime'].str.contains("US KULUÇKA LİDERİ", na=False)
        ].sort_values(by='quant_score', ascending=False).head(20)

        col_map = {
            'ticker': 'Hisse',
            'quant_score': 'Quant Skoru',
            'close': 'Fiyat ($)',
            'mcap_milyon': 'Piyasa Değeri ($M)',
            'dist_from_52w_low': '52H Dip %',
            'target_cup': '1. Çanak Hedefi ($)',
            'potansiyel_cup': 'Çanak Prim %',
            'target_bagger': '2. Bagger Hedefi ($)',
            'stop_price': 'Dinamik Stop ($)',
            'roe': 'ROE %',
            'pe': 'F/K'
        }
        
        display_cols = [c for c in col_map.keys() if c in leaders.columns]
        
        if not leaders.empty:
            st.dataframe(
                leaders[display_cols].rename(columns=col_map),
                column_config={
                    "Quant Skoru": st.column_config.ProgressColumn("Quant Skoru", min_value=0, max_value=100, format="%.1f"),
                    "Fiyat ($)": st.column_config.NumberColumn("Fiyat ($)", format="$%.2f"),
                    "52H Dip %": st.column_config.NumberColumn("52H Dip %", format="%+0.1f%%"),
                    "Çanak Prim %": st.column_config.NumberColumn("Çanak Prim %", format="%+0.0f%%"),
                    "1. Çanak Hedefi ($)": st.column_config.NumberColumn("1. Çanak Hedefi ($)", format="$%.2f"),
                    "2. Bagger Hedefi ($)": st.column_config.NumberColumn("2. Bagger Hedefi ($)", format="$%.2f"),
                    "Dinamik Stop ($)": st.column_config.NumberColumn("Dinamik Stop ($)", format="$%.2f"),
                    "ROE %": st.column_config.NumberColumn("ROE %", format="%%%0.1f"),
                    "F/K": st.column_config.NumberColumn("F/K", format="%.1f")
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Bugün US kuluçka şartlarını sağlayan taze hisse bulunamadı.")
    else:
        st.info("Henüz taranmış US verisi bulunmuyor. Lütfen workflow'u çalıştırın.")

# -----------------------------------------------------------------------------
# 2. SEKME: MODEL FAKTÖRLERİ & YAŞAM DÖNGÜSÜ DEFTERİ
# -----------------------------------------------------------------------------
with tab_ai:
    st.subheader("🧠 Model Dinamik Ağırlıkları (Wall Street Optimizasyonu)")
    
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        st.write(f"**52H Taban Geometrisi:** %{int(weights.get('macro_base', 0.35)*100)}")
        st.progress(float(weights.get('macro_base', 0.35)))
    with col_w2:
        st.write(f"**Büyüme & Kârlılık (ROE):** %{int(weights.get('growth_quality', 0.30)*100)}")
        st.progress(float(weights.get('growth_quality', 0.30)))
    with col_w3:
        st.write(f"**Hacim Akışı & CLV:** %{int(weights.get('volume_flow', 0.20)*100)}")
        st.progress(float(weights.get('volume_flow', 0.20)))
    with col_w4:
        st.write(f"**Hacimli Ateşleme:** %{int(weights.get('ignition', 0.15)*100)}")
        st.progress(float(weights.get('ignition', 0.15)))

    st.write("")
    st.subheader("📋 Sinyal Yaşam Döngüsü & Açık Pozisyon Koruma Defteri")
    
    if not df_lifecycle.empty:
        recent_lifecycle = df_lifecycle.sort_values(by='tarih', ascending=False).head(30)
        life_map = {
            'tarih': 'Sinyal Tarihi',
            'ticker': 'Hisse',
            'entry_price': 'Giriş ($)',
            'stop_price': 'İzleyen Stop ($)',
            'target_cup': 'Çanak Hedefi ($)',
            'max_drawdown': 'Max Çekilme %',
            'peak_gain': 'Tepe Kâr %',
            'outcome': 'Kuluçka Durumu'
        }
        l_cols = [c for c in life_map.keys() if c in recent_lifecycle.columns]
        
        st.dataframe(
            recent_lifecycle[l_cols].rename(columns=life_map),
            column_config={
                "Sinyal Tarihi": st.column_config.DateColumn("Sinyal Tarihi", format="YYYY-MM-DD"),
                "Giriş ($)": st.column_config.NumberColumn("Giriş ($)", format="$%.2f"),
                "İzleyen Stop ($)": st.column_config.NumberColumn("İzleyen Stop ($)", format="$%.2f"),
                "Çanak Hedefi ($)": st.column_config.NumberColumn("Çanak Hedefi ($)", format="$%.2f"),
                "Max Çekilme %": st.column_config.NumberColumn("Max Çekilme %", format="%0.1f%%"),
                "Tepe Kâr %": st.column_config.NumberColumn("Tepe Kâr %", format="%+0.1f%%")
            },
            use_container_width=True,
            hide_index=True
        )

# -----------------------------------------------------------------------------
# 3. SEKME: ELENEN HİSSELER
# -----------------------------------------------------------------------------
with tab_risks:
    st.subheader("🏢 Elenen Hisseler (Zombiler & Aşırı Primliler)")
    st.markdown("*Sermayesini yakan kârsız zombi şirketler ve 52 haftalık dipten aşırı uzaklaşmış riskli hisseler.*")
    
    if not df_gecmis.empty:
        traps = df_latest[
            df_latest['regime'].str.contains("ELENDİ", na=False)
        ].head(25)

        if not traps.empty:
            r_cols = ['ticker', 'regime', 'mcap_milyon', 'roe', 'close']
            r_cols = [c for c in r_cols if c in traps.columns]
            st.dataframe(
                traps[r_cols].rename(columns={
                    'ticker': 'Hisse',
                    'regime': 'Elenme Sebebi',
                    'mcap_milyon': 'Piyasa Değeri ($M)',
                    'roe': 'ROE %',
                    'close': 'Fiyat ($)'
                }),
                column_config={
                    "Piyasa Değeri ($M)": st.column_config.NumberColumn("Piyasa Değeri ($M)", format="$%0.0fM"),
                    "Fiyat ($)": st.column_config.NumberColumn("Fiyat ($)", format="$%.2f")
                },
                use_container_width=True,
                hide_index=True
            )
