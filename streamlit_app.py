import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD
import plotly.graph_objs as go

# === STREAMLIT SETUP ===
st.set_page_config(layout="wide")
st.title("📊 Offline Trading Scanner (CSV Upload)")

# === FUNCTION: Load historical CSV ===
def fetch_data_from_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # ✅ Clean up column names
    df.columns = [col.strip().capitalize() for col in df.columns]

    # ✅ Standardize datetime column
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'Datetime'}, inplace=True)
    elif 'Timestamp' in df.columns:
        df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    return df


# === FUNCTION: Add Indicators ===
def add_indicators(df):
    close_series = pd.Series(df["Close"].values.flatten(), index=df.index)

    bb = BollingerBands(close=close_series, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband().values.ravel()
    df["bb_lower"] = bb.bollinger_lband().values.ravel()
    df["bb_mid"]   = bb.bollinger_mavg().values.ravel()

    rsi = RSIIndicator(close=close_series)
    df["rsi"] = rsi.rsi().values.ravel()

    macd = MACD(close=close_series)
    df["macd"] = macd.macd().values.ravel()
    df["macd_signal"] = macd.macd_signal().values.ravel()

    return df

# === FUNCTION: Predict Next Close Price ===
def predict_price(df):
    df.columns = [str(col).strip().lower() for col in df.columns]

    for col in df.columns:
        if "time" in col or "date" in col:
            df.rename(columns={col: "timestamp"}, inplace=True)
        if "close" in col and col != "close":
            df.rename(columns={col: "close"}, inplace=True)

    df = df.dropna(subset=["timestamp", "close"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["timestamp"].view(np.int64) // 10**9

    X = df["timestamp"].values.reshape(-1, 1)
    y = df["close"].values

    model = LinearRegression()
    model.fit(X, y)

    next_time = np.array([[X[-1][0] + 900]])
    predicted_price = model.predict(next_time)[0]

    return round(predicted_price, 2)

# === FUNCTION: Strategy Logic ===
def check_strategy(df):
    if df.shape[0] < 2 or "bb_upper" not in df.columns:
        return False
    c24 = df.iloc[-2]
    f1 = c24["Open"] > c24["bb_upper"]
    f2 = c24["Close"] > c24["bb_upper"]
    f3 = c24["Close"] > c24["Open"]
    return f1 and f2 and f3

# === FUNCTION: Plot Candlestick Chart ===
def plot_chart(df, symbol):
    fig = go.Figure(data=[go.Candlestick(x=df['Datetime'],
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'],
                                         name="Candlesticks")])
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['bb_upper'], line=dict(color='orange'), name="BB Upper"))
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['bb_lower'], line=dict(color='blue'), name="BB Lower"))

    fig.update_layout(title=symbol, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# === APP INTERFACE ===
uploaded_files = st.file_uploader("📁 Upload one or more historical CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        st.header(f"📈 {file.name}")
        try:
            df = fetch_data_from_csv(file)
            df = add_indicators(df)
            predicted_price = predict_price(df)
            strategy_passed = check_strategy(df)

            col1, col2 = st.columns([2, 1])
            with col1:
                plot_chart(df, file.name)

            with col2:
                st.metric("Predicted Close Price (next 15 min)", f"₹{predicted_price}")
                st.metric("RSI", round(df['rsi'].iloc[-1], 2))
                st.metric("MACD", round(df['macd'].iloc[-1], 2))
                st.metric("Signal Line", round(df['macd_signal'].iloc[-1], 2))

                if strategy_passed:
                    st.success("✅ Strategy Passed!")
                else:
                    st.warning("❌ Strategy Not Passed.")

            st.markdown("---")

        except Exception as e:
            st.error(f"💥 Error in {file.name}: {e}")
else:
    st.info("📤 Upload one or more CSV files to begin.")
