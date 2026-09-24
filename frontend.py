"""
frontend.py  –  Streamlit UI for House Price Prediction
-------------------------------------------------------
Run:  streamlit run frontend.py
Expects the FastAPI backend running at http://localhost:8000
"""

import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; }
    .main-header p  { margin: 0.4rem 0 0 0; opacity: 0.85; font-size: 1rem; }
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .metric-card h2 { margin: 0; color: #1e3a5f; font-size: 1.6rem; }
    .metric-card p  { margin: 0.2rem 0 0 0; color: #57606a; font-size: 0.85rem; }
    .price-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 14px;
        padding: 2.5rem;
        text-align: center;
        color: white;
    }
    .price-box .label { font-size: 1rem; opacity: 0.85; margin-bottom: 0.3rem; }
    .price-box .price { font-size: 3rem; font-weight: 800; letter-spacing: -1px; }
    .price-box .model { font-size: 0.8rem; opacity: 0.7; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Helper: fetch model info (cached) ─────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_model_info():
    try:
        r = requests.get(f"{API_BASE}/model-info", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def call_predict(payload: dict):
    r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏠 House Price Predictor</h1>
    <p>Enter house details in the sidebar and click <strong>Predict Price</strong> to get an instant ML-powered estimate.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar – input form ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏡 House Details")
    st.markdown("---")

    square_footage = st.slider(
        "Square Footage", min_value=500, max_value=5000,
        value=2000, step=50, help="Total interior living area in sq ft"
    )
    num_bedrooms = st.selectbox(
        "Bedrooms", options=list(range(1, 6)), index=2
    )
    num_bathrooms = st.selectbox(
        "Bathrooms", options=list(range(1, 6)), index=1
    )
    year_built = st.slider(
        "Year Built", min_value=1900, max_value=2024,
        value=2000, step=1
    )
    lot_size = st.slider(
        "Lot Size (acres)", min_value=0.1, max_value=10.0,
        value=2.0, step=0.1
    )
    garage_size = st.selectbox(
        "Garage Capacity (cars)", options=list(range(0, 6)), index=1
    )
    neighborhood_quality = st.slider(
        "Neighborhood Quality (1–10)", min_value=1, max_value=10,
        value=5, step=1, help="1 = lowest, 10 = highest"
    )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Price", use_container_width=True, type="primary")

# ── Main content ──────────────────────────────────────────────────────────────
model_info = fetch_model_info()

tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Model Metrics", "🔍 Data Explorer"])

# ━━ Tab 1: Prediction ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    if model_info is None:
        st.error("⚠️ Cannot reach the backend API at **http://localhost:8000**. "
                 "Please start it with:\n```\nuvicorn app:app --reload\n```")
    else:
        if predict_btn:
            payload = {
                "Square_Footage":       square_footage,
                "Num_Bedrooms":         num_bedrooms,
                "Num_Bathrooms":        num_bathrooms,
                "Year_Built":           year_built,
                "Lot_Size":             lot_size,
                "Garage_Size":          garage_size,
                "Neighborhood_Quality": neighborhood_quality,
            }
            with st.spinner("Predicting..."):
                try:
                    result = call_predict(payload)
                    price = result["predicted_price"]

                    # ── Price display ──────────────────────────────────────────
                    st.markdown(f"""
                    <div class="price-box">
                        <div class="label">Estimated House Price</div>
                        <div class="price">${price:,.0f}</div>
                        <div class="model">Model: {result['model_used']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("### 📋 Input Summary")
                    col1, col2 = st.columns(2)
                    labels = {
                        "Square Footage": f"{square_footage:,} sq ft",
                        "Bedrooms": num_bedrooms,
                        "Bathrooms": num_bathrooms,
                        "Year Built": year_built,
                        "Lot Size": f"{lot_size} acres",
                        "Garage": f"{garage_size} car(s)",
                        "Neighborhood Quality": f"{neighborhood_quality}/10",
                    }
                    items = list(labels.items())
                    for i, (k, v) in enumerate(items[:4]):
                        col1.metric(k, v)
                    for i, (k, v) in enumerate(items[4:]):
                        col2.metric(k, v)

                    # ── Gauge chart ────────────────────────────────────────────
                    stats = model_info["feature_stats"]
                    price_min = 100_000
                    price_max = 1_200_000

                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=price,
                        number={"prefix": "$", "valueformat": ",.0f"},
                        delta={"reference": 618861, "prefix": "$", "valueformat": ",.0f"},
                        title={"text": "Predicted vs Dataset Average", "font": {"size": 14}},
                        gauge={
                            "axis": {"range": [price_min, price_max], "tickformat": "$,.0f"},
                            "bar":  {"color": "#2d6a9f"},
                            "steps": [
                                {"range": [price_min, 400_000], "color": "#d1ecf1"},
                                {"range": [400_000, 750_000],  "color": "#bee3f8"},
                                {"range": [750_000, price_max], "color": "#90cdf4"},
                            ],
                            "threshold": {
                                "line": {"color": "#e53e3e", "width": 3},
                                "thickness": 0.75,
                                "value": 618861,
                            },
                        }
                    ))
                    fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                except requests.exceptions.ConnectionError:
                    st.error("Backend not reachable. Start it with `uvicorn app:app --reload`.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.info("👈 Adjust house details in the sidebar and click **Predict Price**.")

            # ── Feature reference cards ──────────────────────────────────────
            if model_info:
                st.markdown("### 📐 Dataset Feature Ranges")
                stats = model_info["feature_stats"]
                feature_labels = {
                    "Square_Footage":       "Square Footage",
                    "Num_Bedrooms":         "Bedrooms",
                    "Num_Bathrooms":        "Bathrooms",
                    "Year_Built":           "Year Built",
                    "Lot_Size":             "Lot Size (acres)",
                    "Garage_Size":          "Garage Size",
                    "Neighborhood_Quality": "Neighborhood Quality",
                }
                cols = st.columns(4)
                for i, (feat, label) in enumerate(feature_labels.items()):
                    s = stats[feat]
                    with cols[i % 4]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h2>{s['mean']:.1f}</h2>
                            <p><b>{label}</b><br>
                            min {s['min']:.1f} · max {s['max']:.1f}</p>
                        </div>
                        """, unsafe_allow_html=True)

# ━━ Tab 2: Model Metrics ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    if model_info is None:
        st.error("Backend unavailable.")
    else:
        st.markdown("### 🏆 Model Comparison")
        all_metrics = model_info.get("all_model_metrics", {})
        if all_metrics:
            df_metrics = pd.DataFrame(all_metrics).T.reset_index()
            df_metrics.columns = ["Model", "MAE", "RMSE", "R²"]
            df_metrics = df_metrics.sort_values("R²", ascending=False)

            best_model_name = model_info["model_name"]
            df_metrics["Best"] = df_metrics["Model"] == best_model_name

            # Table
            st.dataframe(
                df_metrics[["Model", "MAE", "RMSE", "R²"]].style.format({
                    "MAE":  "${:,.0f}",
                    "RMSE": "${:,.0f}",
                    "R²":   "{:.4f}",
                }).highlight_max(subset=["R²"], color="#d4edda")
                  .highlight_min(subset=["MAE", "RMSE"], color="#d4edda"),
                use_container_width=True,
            )

            col1, col2 = st.columns(2)

            # R² bar chart
            with col1:
                fig_r2 = px.bar(
                    df_metrics, x="Model", y="R²",
                    color="R²", color_continuous_scale="Blues",
                    title="R² Score by Model (higher = better)",
                    text_auto=".4f",
                )
                fig_r2.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_r2, use_container_width=True)

            # MAE / RMSE grouped bar
            with col2:
                fig_err = go.Figure()
                fig_err.add_trace(go.Bar(
                    name="MAE",  x=df_metrics["Model"], y=df_metrics["MAE"],
                    marker_color="#2d6a9f"
                ))
                fig_err.add_trace(go.Bar(
                    name="RMSE", x=df_metrics["Model"], y=df_metrics["RMSE"],
                    marker_color="#90cdf4"
                ))
                fig_err.update_layout(
                    barmode="group", title="MAE vs RMSE by Model (lower = better)",
                    height=350, yaxis_tickprefix="$", yaxis_tickformat=","
                )
                st.plotly_chart(fig_err, use_container_width=True)

        m = model_info["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE",  f"${m['MAE']:,.0f}",  help="Mean Absolute Error")
        c2.metric("RMSE", f"${m['RMSE']:,.0f}", help="Root Mean Squared Error")
        c3.metric("R²",   f"{m['R2']:.4f}",     help="Coefficient of Determination")

# ━━ Tab 3: Data Explorer ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("### 🔍 Explore the Training Dataset")
    try:
        df = pd.read_csv("house_price_regression_dataset.csv")

        col1, col2 = st.columns([2, 1])
        with col1:
            # Scatter: Square Footage vs Price
            color_by = st.selectbox(
                "Color points by:",
                ["Neighborhood_Quality", "Num_Bedrooms", "Num_Bathrooms", "Garage_Size"]
            )
            fig_scatter = px.scatter(
                df, x="Square_Footage", y="House_Price",
                color=color_by, size="Lot_Size",
                hover_data=["Year_Built", "Num_Bedrooms", "Num_Bathrooms"],
                color_continuous_scale="Viridis",
                labels={"House_Price": "Price ($)", "Square_Footage": "Square Footage"},
                title=f"Square Footage vs House Price (colored by {color_by})",
                opacity=0.7,
            )
            fig_scatter.update_layout(height=420)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            # Price distribution
            fig_hist = px.histogram(
                df, x="House_Price", nbins=40,
                title="Price Distribution",
                labels={"House_Price": "Price ($)"},
                color_discrete_sequence=["#2d6a9f"],
            )
            fig_hist.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

        # Correlation heatmap
        st.markdown("#### Correlation Matrix")
        corr = df.corr(numeric_only=True)
        fig_heat = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Heatmap",
        )
        fig_heat.update_layout(height=420)
        st.plotly_chart(fig_heat, use_container_width=True)

        # Raw data
        with st.expander("📄 Raw Data (first 100 rows)"):
            st.dataframe(df.head(100), use_container_width=True)

    except FileNotFoundError:
        st.error("Dataset CSV not found. Make sure `house_price_regression_dataset.csv` is in the project root.")
