import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ===== Indicator Functions =====

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

# ===== Streamlit UI =====

st.set_page_config("📉 RSI, MACD, Returns Generator", layout="wide")
st.title("📊 Financial Indicator Generator (RSI, MACD, Returns)")

uploaded_file = st.file_uploader("📥 Upload your OHLCV CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(df.columns):
            st.error(f"❌ Your file must include at least: {required_cols}")
            st.stop()

        # Sidebar config
        st.sidebar.header("📐 Indicator Parameters")
        rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
        fast = st.sidebar.slider("MACD Fast EMA", 5, 20, 12)
        slow = st.sidebar.slider("MACD Slow EMA", 10, 50, 26)
        signal = st.sidebar.slider("MACD Signal Line", 5, 20, 9)

        # Compute indicators
        df = calculate_returns(df)
        df = calculate_rsi(df, rsi_period)
        df = calculate_macd(df, fast, slow, signal)

        # Ensure compatibility with LSTM-GARCH app
        df['Log_Volume'] = np.log(df['Volume'].replace(0, np.nan)).fillna(0)
        df['MACD'] = df['MACD_Signal']

        # Display data
        st.subheader("🧾 Data Preview")
        st.dataframe(df.tail(10))

        # RSI Chart
        st.subheader("📈 RSI Line Chart")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(height=300)
        st.plotly_chart(fig_rsi, use_container_width=True)

        # MACD Chart
        st.subheader("📉 MACD Signal + Histogram")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='MACD Signal', line=dict(color='orange')))
        fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram'))
        fig_macd.update_layout(height=300)
        st.plotly_chart(fig_macd, use_container_width=True)

        # Download CSV
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV with Indicators", csv_data, file_name="indicators_output.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("👆 Upload a CSV with OHLCV columns to get started.")
