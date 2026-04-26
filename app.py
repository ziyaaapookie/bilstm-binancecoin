import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from datetime import timedelta
import warnings

warnings.filterwarnings('ignore')

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="BNB Prediction Dashboard",
    layout="wide",
    page_icon="🪙"
)

# =============================
# CUSTOM CSS
# =============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .main {
        background-color: #0B0E11;
        color: #EAECEF;
    }

    .stMetric {
        background-color: #1E2329;
        padding: 12px 15px;
        border-radius: 10px;
        border: 1px solid #2B3139;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    }

    h1, h2, h3 {
        color: #F0B90B;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_resources():
    model = load_model("bilstm_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_resources()

# =============================
# HEADER
# =============================
bnb_logo = "https://cdn.pixabay.com/photo/2021/04/30/16/47/binance-logo-6219389_640.png"

st.markdown(f"""
<div style="text-align: center;">
    <img src="{bnb_logo}" width="120" style="border-radius:100px;">
    <h1>Binance Coin (BNB) Price Prediction With Bi-LSTM</h1>
    <p>Data real-time dari Yahoo Finance, model dilatih offline.</p>
</div>
""", unsafe_allow_html=True)

# =============================
# FETCH DATA
# =============================
df = yf.download("BNB-USD", period="90d", interval="1d", progress=False, auto_adjust=True)

if df.empty:
    st.error("⚠️ Gagal mengambil data.")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

close_prices = df['Close'].values

# =============================
# PREPARE DATA
# =============================
time_step = 60
data_scaled = scaler.transform(close_prices.reshape(-1, 1))
last_60 = data_scaled[-time_step:]

# =============================
# PREDIKSI
# =============================

# Prediksi hari ini
X_today = data_scaled[-(time_step + 1):-1].reshape(1, time_step, 1)
pred_today = scaler.inverse_transform(model.predict(X_today, verbose=0))[0][0]

# Prediksi besok
X_tomorrow = last_60.reshape(1, time_step, 1)
pred_tomorrow = scaler.inverse_transform(model.predict(X_tomorrow, verbose=0))[0][0]

# Harga aktual
today_price = close_prices[-1]
yesterday_price = close_prices[-2]
day_before_yesterday_price = close_prices[-3]

# =============================
# DELTA YANG BENAR (FIX)
# =============================
delta_today = pred_today - today_price
delta_pct_today = (delta_today / today_price) * 100

delta_tomorrow = pred_tomorrow - today_price
delta_pct_tomorrow = (delta_tomorrow / today_price) * 100

# =============================
# METRICS
# =============================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📅 2 Hari Lalu", f"${day_before_yesterday_price:,.2f}")

with col2:
    st.metric("📅 Kemarin", f"${yesterday_price:,.2f}")

with col3:
    st.metric(
        "🎯 Prediksi Hari Ini",
        f"${pred_today:,.2f}",
        delta=f"{delta_today:+,.2f} ({delta_pct_today:+.2f}%)"
    )

with col4:
    st.metric("💰 Harga Realtime", f"${today_price:,.2f}")

with col5:
    st.metric(
        "🔮 Prediksi Besok",
        f"${pred_tomorrow:,.2f}",
        delta=f"{delta_tomorrow:+,.2f} ({delta_pct_tomorrow:+.2f}%)"
    )

# =============================
# CHART 7 HARI
# =============================
st.divider()
st.subheader("📈 Tren Harga 7 Hari Terakhir")

last_7 = df.tail(7)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=last_7.index,
    y=last_7['Close'],
    mode='lines+markers',
    line=dict(color='#F0B90B', width=3)
))

fig.update_layout(template='plotly_dark')
st.plotly_chart(fig, use_container_width=True)

# =============================
# FUTURE PREDICTION
# =============================
def predict_future(days):
    temp = list(last_60.flatten())
    preds = []

    for _ in range(days):
        x = np.array(temp[-time_step:]).reshape(1, time_step, 1)
        yhat = model.predict(x, verbose=0)
        temp.append(yhat[0][0])
        preds.append(yhat[0][0])

    preds = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds).flatten()

st.divider()
st.subheader("🚀 Prediksi Jangka Pendek")

days = st.selectbox("Pilih Hari", [3, 7, 14], index=1)

if st.button("Generate Prediksi"):
    future = predict_future(days)
    dates = [df.index[-1] + timedelta(days=i+1) for i in range(days)]

    future_df = pd.DataFrame({
        "Tanggal": dates,
        "Prediksi": future
    })

    st.dataframe(
        future_df.style.format({"Prediksi": "${:,.2f}"}),
        use_container_width=True,
        hide_index=True
    )

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=future_df['Tanggal'],
        y=future_df['Prediksi'],
        mode='lines+markers',
        line=dict(color='#00C853', dash='dot')
    ))

    fig2.add_trace(go.Scatter(
        x=[df.index[-1]],
        y=[today_price],
        mode='markers',
        marker=dict(size=10, color='#F0B90B')
    ))

    fig2.update_layout(template='plotly_dark')
    st.plotly_chart(fig2, use_container_width=True)

# =============================
# FOOTER
# =============================
st.divider()
st.caption("⚠️ Bukan saran finansial. Gunakan dengan bijak.")