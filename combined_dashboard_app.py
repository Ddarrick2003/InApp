import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.callbacks import EarlyStopping
from utils.preprocessing import preprocess_data
from model.lstm_model import create_sequences, build_lstm_model
from model.garch_model import forecast_garch_var

# =================== Indicator Functions ===================

def calculate_returns(df):
    df['Returns'] = df['Close'].pct_change()
    return df

def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    df['MACD_Signal'] = signal_line
    df['MACD_Histogram'] = histogram
    return df

# =================== Streamlit Setup ===================

st.set_page_config(page_title="📊 Financial Dashboard (Indicators + LSTM + GARCH)", layout="wide")
st.title("📉 Combined Financial App: Indicators + LSTM Forecast + GARCH Risk")

uploaded_file = st.file_uploader("📥 Upload your OHLCV CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("🧾 Original Uploaded Data", df.head())

        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(df.columns):
            st.error(f"❌ Your file must include: {required_cols}")
            st.stop()

        # Sidebar settings for indicators
        st.sidebar.header("📐 Indicator Parameters")
        rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
        fast = st.sidebar.slider("MACD Fast EMA", 5, 20, 12)
        slow = st.sidebar.slider("MACD Slow EMA", 10, 50, 26)
        signal = st.sidebar.slider("MACD Signal Line", 5, 20, 9)

        # Indicator calculation
        df = calculate_returns(df)
        df = calculate_rsi(df, rsi_period)
        df = calculate_macd(df, fast, slow, signal)
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df['Log_Volume'] = np.log(df['Volume'].clip(lower=1))
        df['MACD'] = df['MACD_Signal']

        st.success(f"✅ Data Ready: {df.shape[0]} rows")

        tab1, tab2, tab3 = st.tabs(["📊 Indicators", "📈 LSTM Forecast", "📉 GARCH Risk"])

        with tab1:
            st.subheader("📈 RSI and MACD")

            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(fig_rsi, use_container_width=True)

            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='MACD Signal'))
            fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram'))
            st.plotly_chart(fig_macd, use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV with Indicators", csv, "output_indicators.csv", "text/csv")

        with tab2:
            st.subheader("LSTM Forecast")
            try:
                features = ['Open', 'High', 'Low', 'Close', 'Log_Volume', 'RSI', 'MACD', 'Returns']
                df_lstm, _, _ = preprocess_data(df)
                X, y = create_sequences(df_lstm[features], target_col='Close')
                if len(X) < 60:
                    st.warning("⚠️ Not enough data for LSTM.")
                else:
                    split = int(len(X) * 0.8)
                    X_train, X_test = X[:split], X[split:]
                    y_train, y_test = y[:split], y[split:]

                    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
                    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test),
                              callbacks=[EarlyStopping(patience=3)], verbose=0)
                    preds = model.predict(X_test).flatten()
                    st.line_chart({"Actual": y_test[:100], "Predicted": preds[:100]})
            except Exception as e:
                st.error(f"LSTM Error: {e}")

        with tab3:
            st.subheader("GARCH Forecast")
            try:
                vol_forecast, var_1d = forecast_garch_var(df)
                st.metric(label="1-Day VaR (95%)", value=f"{var_1d:.2f}%")
                st.line_chart(vol_forecast.values)
                st.markdown("### 📖 Risk Interpretation")
                st.info(f"""
                - **Volatility Forecast** shows expected future variance.
                - Spikes = higher market uncertainty.
                - **1-Day VaR (95%) = {abs(var_1d):.2f}%**, i.e., 95% confidence loss will not exceed this in a day.
                """)
            except Exception as e:
                st.error(f"GARCH Error: {e}")

    except Exception as e:
        st.error(f"❌ Processing Error: {e}")
else:
    st.info("👆 Upload a CSV with OHLCV to get started.")
