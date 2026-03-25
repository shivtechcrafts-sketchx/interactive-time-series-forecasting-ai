import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

st.set_page_config(page_title="Time Series Predictor", layout="centered")

st.title("📈 Time Series Predictor (Interactive)")

# Load model
model = load_model("model/model.keras")

# Upload CSV
file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:

    # Read data
    data = pd.read_csv(file)
    data.columns = data.columns.str.strip()

    st.subheader("📊 Data Preview")
    st.write(data.head())

    # Safe column selection
    if "value" in data.columns:
        col = "value"
    else:
        col = data.columns[-1]

    values = data[col].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    window = 3

    # Last sequence
    last_values = values_scaled[-window:]
    last_values = last_values.reshape(1, window, 1)

    # Prediction
    prediction = model.predict(last_values)
    pred_value = scaler.inverse_transform(prediction)

    st.success(f"🔮 Next Value Prediction: {pred_value[0][0]:.2f}")

    # =========================
    # 📊 GRAPH 1: Line Plot (Actual vs Prediction)
    # =========================
    st.subheader("📊 Actual vs Prediction (Interactive Line)")

    extended = list(values.flatten()) + [pred_value[0][0]]

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        y=values.flatten(),
        mode='lines+markers',
        name='Actual'
    ))

    fig1.add_trace(go.Scatter(
        y=extended,
        mode='lines',
        name='Prediction',
        line=dict(dash='dash')
    ))

    fig1.update_layout(title="Actual vs Predicted Value")

    st.plotly_chart(fig1, use_container_width=True)

    # =========================
    # 📉 GRAPH 2: Area Plot (Moving Average)
    # =========================
    st.subheader("📉 Moving Average Trend (Interactive Area)")

    data["moving_avg"] = data[col].rolling(window=3).mean()

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        y=data[col],
        fill='tozeroy',
        name='Actual'
    ))

    fig2.add_trace(go.Scatter(
        y=data["moving_avg"],
        mode='lines',
        name='Moving Avg'
    ))

    fig2.update_layout(title="Trend Visualization")

    st.plotly_chart(fig2, use_container_width=True)

    # =========================
    # 📊 GRAPH 3: Bar Plot (Comparison)
    # =========================
    st.subheader("📊 Prediction Comparison (Bar Chart)")

    labels = ["Last Actual", "Predicted"]
    values_bar = [values[-1][0], pred_value[0][0]]

    fig3 = go.Figure()

    fig3.add_trace(go.Bar(
        x=labels,
        y=values_bar
    ))

    fig3.update_layout(title="Actual vs Predicted Value")

    st.plotly_chart(fig3, use_container_width=True)

    # =========================
    # 💡 Insights
    # =========================
    st.subheader("💡 Insights")

    if values[-1] < pred_value[0][0]:
        st.success("📈 Trend is increasing")
    else:
        st.warning("📉 Trend is decreasing or stable")