import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==============================================================================
# 1. KONFIGURASI HALAMAN & CSS PREMIUM
# ==============================================================================
st.set_page_config(
    page_title="EWS Cuaca Citeko Pro",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom untuk Tampilan Vertikal yang Rapi & Profesional
st.markdown("""
<style>
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 30px;
        padding-bottom: 20px;
        border-bottom: 2px solid #eee;
    }
    .status-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .danger-theme {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
        color: white;
    }
    .safe-theme {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .mitigation-card {
        background-color: #f8f9fa;
        border-left: 5px solid #1E3A8A;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #34495e;
        margin-top: 30px;
        margin-bottom: 15px;
        border-left: 5px solid #1E3A8A;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD ASSETS (UPDATED: XGBOOST ONLY)
# ==============================================================================
@st.cache_resource
def load_assets():
    base_dir = 'deployment_files'
    try:
        # Load Config
        with open(os.path.join(base_dir, 'model_config.json'), 'r') as f:
            config = json.load(f)
        
        # Load XGBoost Model (Pemenang)
        # Note: Scaler dan LSTM dihapus karena di Notebook terakhir kita hanya save XGBoost
        xgb_model = joblib.load(os.path.join(base_dir, 'best_model_xgboost.pkl'))
        
        return config, xgb_model
    except Exception as e:
        st.error(f"❌ System Error: {e}. Pastikan folder 'deployment_files' lengkap.")
        st.stop()

config, xgb_model = load_assets()

# DATA HISTORIS CITEKO (HARDCODED STATS UNTUK KONTEKS)
HISTORICAL_MEAN = {
    'RR': 8.5,    
    'RH': 82.0,   
    'TAVG': 24.5  
}

# ==============================================================================
# 3. HEADER
# ==============================================================================
st.markdown('<div class="main-title">SISTEM PERINGATAN DINI CUACA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Prediksi Risiko Hujan Ekstrem (H+1) & Rekomendasi Mitigasi Cerdas</div>', unsafe_allow_html=True)

# ==============================================================================
# 4. SIDEBAR (INPUT & DOWNLOAD CENTER)
# ==============================================================================
with st.sidebar:
    st.header("🎛️ Input Data HARI INI")
    st.info("Masukkan data observasi cuaca hari ini untuk memprediksi kondisi **BESOK**.")
    
    # Init Session State
    if 'rr_val' not in st.session_state: st.session_state['rr_val'] = 0.0
    if 'rh_val' not in st.session_state: st.session_state['rh_val'] = 80.0
    if 'tavg_val' not in st.session_state: st.session_state['tavg_val'] = 24.0

    # Tombol Simulasi
    st.markdown("**Simulasi Cepat:**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("☀️ Hari Cerah"):
            st.session_state.update({'rr_val': np.random.uniform(0, 5), 'rh_val': np.random.uniform(60, 80), 'tavg_val': np.random.uniform(25, 29)})
    with c2:
        if st.button("⛈️ Hari Hujan"):
            st.session_state.update({'rr_val': np.random.uniform(40, 100), 'rh_val': np.random.uniform(92, 99), 'tavg_val': np.random.uniform(20, 23)})

    # Input Manual
    st.markdown("---")
    rr_val = st.number_input("Curah Hujan Hari Ini (mm)", 0.0, 500.0, key='rr_val')
    rh_val = st.number_input("Kelembaban Hari Ini (%)", 0.0, 100.0, key='rh_val')
    tavg_val = st.number_input("Suhu Rata-rata Hari Ini (°C)", 10.0, 40.0, key='tavg_val')
    
    # Tambahan input agar sesuai fitur model (SS dan FF)
    # Kita buat hidden/default calculation jika user tidak mau ribet, 
    # atau tampilkan jika ingin detail. Disini kita pakai estimasi sederhana.
    ss_est = 6.0 if rr_val < 5 else (0.0 if rr_val > 20 else 2.0)
    ff_est = 2.0 if rr_val < 20 else (5.0 if rr_val > 50 else 3.0)
    
    analyze_btn = st.button("🔍 PREDIKSI & ANALISIS", type="primary", use_container_width=True)

    # FOOTER SIDEBAR
    st.markdown("---")
    st.caption(f"Versi Sistem: v5.0 (Final)\nModel: XGBoost Optuna\nLast Update: {datetime.now().strftime('%d-%m-%Y')}")

# ==============================================================================
# 5. LOGIKA PREDIKSI & GENERATOR LAPORAN
# ==============================================================================
prob_xgb = 0.0
is_danger = False
# Mengambil threshold dari config (default 0.35 jika tidak ada)
threshold = config.get('threshold', 0.35) 
prediction_text = ""

if analyze_btn:
    try:
        # --- A. FEATURE ENGINEERING (SINKRONISASI DENGAN NOTEBOOK) ---
        # Kita menggunakan data input "Hari Ini" sebagai basis.
        # Strategi: Persistence (Nilai Lag 1, 2, 3 dianggap sama dengan Hari Ini)
        # agar user tidak perlu input data historis.
        
        # 1. Base Variables
        current_data = {
            'RR': rr_val,
            'TAVG': tavg_val,
            'RH_AVG': rh_val,
            'SS': ss_est,
            'FF_AVG': ff_est
        }
        
        features_dict = {}
        
        # 2. Masukkan Variabel Utama (Non-Target) ke Features Dict
        # Sesuai notebook: X = df.drop(['RR', 'Target']) -> Jadi TAVG, RH, SS, FF masuk
        features_dict['TAVG'] = tavg_val
        features_dict['RH_AVG'] = rh_val
        features_dict['SS'] = ss_est
        features_dict['FF_AVG'] = ff_est
        
        # 3. Generate Lag Features (1, 2, 3)
        cols_to_lag = ['RR', 'TAVG', 'RH_AVG', 'SS', 'FF_AVG']
        for col in cols_to_lag:
            val = current_data[col]
            for i in [1, 2, 3]:
                # Persistence assumption: Lag_i = Current Value
                features_dict[f'{col}_Lag{i}'] = val
        
        # 4. Generate Rolling Features
        # RR_Roll3_Mean dari RR_Lag1, Lag2, Lag3
        rr_lags = [features_dict['RR_Lag1'], features_dict['RR_Lag2'], features_dict['RR_Lag3']]
        features_dict['RR_Roll3_Mean'] = np.mean(rr_lags)
        features_dict['RR_Roll3_Max'] = np.max(rr_lags)
        
        # 5. DataFrame Creation & Ordering
        df_input = pd.DataFrame([features_dict])
        
        # PENTING: Reorder kolom sesuai urutan training XGBoost
        required_feats = config['feature_names']
        
        # Safety check: isi 0 jika ada kolom kurang (tapi logic diatas harusnya sudah cover)
        for f in required_feats:
            if f not in df_input.columns:
                df_input[f] = 0.0
                
        df_input = df_input[required_feats]
        
        # --- B. PREDICTION (XGBOOST) ---
        # XGBoost tidak butuh Scaling manual jika dilatih dengan data raw/robust scaler internal
        # Di notebook terakhir, kita langsung fit ke X_train_smote
        
        # Ambil probabilitas kelas 1 (Bahaya)
        prob_xgb = xgb_model.predict_proba(df_input)[0][1]
        is_danger = prob_xgb >= threshold

    except Exception as e:
        st.error(f"Terjadi kesalahan pada model: {e}")
        st.stop()

# ==============================================================================
# 6. VISUALISASI UTAMA
# ==============================================================================

if analyze_btn:
    # --- BAGIAN 1: STATUS UTAMA ---
    col_res1, col_res2 = st.columns([1.5, 1])
    
    with col_res1:
        st.markdown('<div class="section-header">📡 Status Prediksi (H+1)</div>', unsafe_allow_html=True)
        if is_danger:
            prediction_text = "BAHAYA / SIAGA"
            st.markdown(f"""
            <div class="status-box danger-theme">
                <h1 style="margin:0; font-size:3.5rem">BAHAYA 🚨</h1>
                <h3>Potensi Hujan Ekstrem (>20mm) Besok Hari</h3>
                <p style="font-size:1.2rem; margin-top:10px">Sistem mendeteksi pola cuaca yang berisiko tinggi memicu bencana hidrometeorologi.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            prediction_text = "AMAN / NORMAL"
            st.markdown(f"""
            <div class="status-box safe-theme">
                <h1 style="margin:0; font-size:3.5rem">AMAN ✅</h1>
                <h3>Cuaca Diprediksi Kondusif Besok Hari</h3>
                <p style="font-size:1.2rem; margin-top:10px">Tidak terdeteksi gangguan cuaca signifikan. Aktivitas normal dapat dilakukan.</p>
            </div>
            """, unsafe_allow_html=True)
            
    with col_res2:
        st.markdown('<div class="section-header">🌡️ Tingkat Risiko</div>', unsafe_allow_html=True)
        gauge_val = prob_xgb * 100
        fig_g = go.Figure(go.Indicator(
            mode = "gauge+number", value = gauge_val,
            title = {'text': "Probabilitas (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#2c3e50"},
                'steps': [{'range': [0, threshold*100], 'color': "#d1f2eb"}, {'range': [threshold*100, 100], 'color': "#fadbd8"}],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold*100}
            }
        ))
        fig_g.update_layout(height=250, margin=dict(l=20,r=20,t=10,b=20))
        st.plotly_chart(fig_g, use_container_width=True)

    # --- BAGIAN BARU: KONTEKS HISTORIS ---
    st.markdown('<div class="section-header">📊 Konteks Data Input (Anomaly Detection)</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    delta_rr = rr_val - HISTORICAL_MEAN['RR']
    delta_rh = rh_val - HISTORICAL_MEAN['RH']
    
    c1.metric("Curah Hujan Input", f"{rr_val:.1f} mm", f"{delta_rr:.1f} mm vs Rata-rata", delta_color="inverse")
    c2.metric("Kelembaban Input", f"{rh_val:.1f} %", f"{delta_rh:.1f} % vs Rata-rata", delta_color="inverse")
    c3.metric("Status Anomali", "Terdeteksi" if is_danger else "Normal", "High Risk" if is_danger else "Low Risk")
    
    st.info("💡 **Insight:** Metrik di atas membandingkan input Anda dengan rata-rata historis Citeko.")

    # --- BAGIAN 2: ESTIMASI TREN (FORECAST 24 JAM) ---
    st.markdown('<div class="section-header">📉 Estimasi Tren Cuaca 24 Jam Kedepan</div>', unsafe_allow_html=True)
    
    hours = [f"{i:02d}:00" for i in range(24)]
    hourly_risk = []
    base_risk = prob_xgb
    for i in range(24):
        factor = 1.0
        if 13 <= i <= 17: factor = 1.3 
        elif 0 <= i <= 6: factor = 0.6 
        val = min(base_risk * factor * np.random.uniform(0.9, 1.1), 0.99)
        hourly_risk.append(val * 100)
    
    df_forecast = pd.DataFrame({"Jam": hours, "Risiko (%)": hourly_risk})
    fig_line = px.area(df_forecast, x="Jam", y="Risiko (%)", markers=True)
    color_line = '#c0392b' if is_danger else '#27ae60'
    fig_line.update_traces(line_color=color_line, fillcolor=color_line.replace(')', ', 0.3)').replace('rgb', 'rgba'))
    fig_line.add_hline(y=threshold*100, line_dash="dash", line_color="red", annotation_text="Batas Waspada")
    fig_line.update_layout(yaxis_range=[0, 100], height=350, margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig_line, use_container_width=True)

    # --- BAGIAN BARU: REKOMENDASI MITIGASI ---
    st.markdown('<div class="section-header">🛡️ Rekomendasi Mitigasi (SOP)</div>', unsafe_allow_html=True)
    
    col_mit1, col_mit2 = st.columns(2)
    
    with col_mit1:
        st.markdown("#### 📋 Langkah Taktis (Masyarakat)")
        if is_danger:
            st.markdown("""
            - 🔴 **Hindari bantaran sungai:** Potensi banjir bandang meningkat.
            - 🔴 **Periksa saluran air:** Pastikan drainase tidak tersumbat sampah.
            - 🔴 **Waspada longsor:** Hindari area tebing curam di kawasan Puncak.
            - 🔴 **Siapkan Tas Siaga Bencana:** Dokumen penting, obat, dan senter.
            """)
        else:
            st.markdown("""
            - 🟢 **Aktivitas Normal:** Aman untuk beraktivitas di luar ruangan.
            - 🟢 **Pemeliharaan:** Lakukan pembersihan selokan secara rutin.
            - 🟢 **Hemat Air:** Manfaatkan kondisi cerah untuk menampung air bersih.
            """)
            
    with col_mit2:
        st.markdown("#### 🏛️ Rekomendasi Instansi (BPBD/Pemda)")
        if is_danger:
            st.markdown("""
            - 📢 **Broadcast Peringatan:** Kirim notifikasi SMS blast ke warga Citeko/Puncak.
            - 🚜 **Siagakan Alat Berat:** Fokus di titik rawan longsor jalur Puncak.
            - 🚑 **Tim Reaksi Cepat:** Standby di posko bencana kecamatan.
            """)
        else:
            st.markdown("""
            - 📊 **Monitoring Rutin:** Terus pantau data telemetri AWS.
            - 🛠️ **Maintenance Alat:** Cek sensor curah hujan dan anemometer.
            """)

    # --- BAGIAN 3: EXPLAINABLE AI (XAI) - XGBOOST ---
    st.markdown('<div class="section-header">🧠 Analisis Penyebab (XAI)</div>', unsafe_allow_html=True)
    
    col_xai1, col_xai2 = st.columns(2)
    with col_xai1:
        st.subheader("Faktor Dominan (Feature Importance)")
        importance = xgb_model.feature_importances_
        feats = config['feature_names']
        df_imp = pd.DataFrame({'Fitur': feats, 'Bobot': importance}).sort_values('Bobot', ascending=True).tail(8)
        fig_bar = px.bar(df_imp, x='Bobot', y='Fitur', orientation='h', text_auto='.3f', 
                          color='Bobot', color_continuous_scale='Blues')
        fig_bar.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_xai2:
        st.subheader("Profil Input (Radar Chart)")
        # Normalisasi sederhana untuk visualisasi radar
        vals = [
            min(rr_val/100, 1), 
            min(rh_val/100, 1), 
            1-(min((tavg_val-15)/20, 1)), # Semakin dingin semakin ke luar (1)
            min(ss_est/12, 1), 
            min(ff_est/10, 1)
        ]
        cats = ['Hujan', 'Kelembaban', 'Suhu Dingin', 'Sinar Matahari', 'Angin']
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill='toself', 
            line_color='#e74c3c' if is_danger else '#2ecc71'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=350, showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- FITUR BARU: GENERATE REPORT ---
    st.markdown("---")
    
    report_text = f"""
    LAPORAN PERINGATAN DINI CUACA CITEKO
    ====================================
    Tanggal Generate : {datetime.now().strftime("%d-%m-%Y %H:%M")}
    Lokasi           : Stasiun Klimatologi Citeko, Bogor
    
    DATA INPUT:
    - Curah Hujan Hari Ini : {rr_val} mm
    - Kelembaban Rata-rata : {rh_val} %
    - Suhu Rata-rata       : {tavg_val} C
    
    HASIL PREDIKSI (H+1):
    - Status           : {prediction_text}
    - Probabilitas     : {prob_xgb:.2%}
    - Threshold Model  : {threshold:.2f}
    
    REKOMENDASI:
    {'[SIAGA] Aktifkan protokol bencana.' if is_danger else '[NORMAL] Lakukan monitoring rutin.'}
    
    Dibuat oleh Sistem EWS berbasis XGBoost.
    """
    
    st.download_button(
        label="📄 UNDUH LAPORAN (TXT)",
        data=report_text,
        file_name=f"Laporan_EWS_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        help="Unduh hasil prediksi ini untuk keperluan laporan administrasi."
    )

else:
    # Tampilan Awal
    st.warning("👈 Silakan masukkan data cuaca hari ini di Sidebar, lalu klik tombol **PREDIKSI & ANALISIS**.")
    st.markdown("""
    <div style="text-align: center; color: #95a5a6; padding: 50px;">
        <h2>Sistem Siap Digunakan</h2>
        <p>Gunakan tombol 'Simulasi Cepat' di sidebar untuk mencoba skenario ekstrem.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 0.8rem; color: gray;'>Developed for Thesis Research | Universitas Universitas Nasional</div>", unsafe_allow_html=True)