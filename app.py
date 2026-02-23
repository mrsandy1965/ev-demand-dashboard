"""
app.py
Streamlit Dashboard: EV Charging Demand Forecasting & Load Intelligence
Cloud-optimized: processes only the selected zone, not the full dataset.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from models import load_model, saved_model_exists
from forecasting import recursive_forecast
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ──────────────────── Startup Debug ────────────────────
print("WORKING DIR:", os.getcwd())
print("FILES:", os.listdir())

# ──────────────────── Page Config ────────────────────
st.set_page_config(
    page_title="EV Charging Demand Forecasting",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1DB954;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1DB954;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 0.3rem;
    }
    .best-badge {
        background: #1DB954;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────── Data Path Config ────────────────────
DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "20220901-20230228_zone-cleaned-aggregated",
    "charge_1hour",
)
VOLUME_CSV = os.path.join(DATA_DIR, "volume.csv")
FEATURE_COLS = ["hour", "day_of_week", "month", "lag_1", "lag_24"]


# ──────────────────── Cached: Load Wide CSV ────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_wide_csv(csv_source=None):
    """Load the wide-format CSV once. Returns DataFrame + zone list."""
    if csv_source is None:
        df = pd.read_csv(VOLUME_CSV)
    elif isinstance(csv_source, str):
        df = pd.read_csv(csv_source)
    else:
        df = pd.read_csv(csv_source)

    # First column is 'time', rest are zone columns
    time_col = df.columns[0]
    df.rename(columns={time_col: "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"])
    zone_cols = [c for c in df.columns if c != "time"]
    print(f"[LOAD] {len(df)} rows, {len(zone_cols)} zones")
    return df, zone_cols


# ──────────────────── Per-Zone Processing ────────────────────
@st.cache_data(show_spinner="Processing zone data...")
def prepare_zone_data(wide_df, zone_id):
    """Extract and engineer features for a single zone from wide-format data."""
    zone_df = wide_df[["time", zone_id]].copy()
    zone_df.rename(columns={zone_id: "demand"}, inplace=True)
    zone_df["zone_id"] = zone_id
    zone_df = zone_df.sort_values("time").reset_index(drop=True)

    # Temporal features
    zone_df["hour"] = zone_df["time"].dt.hour
    zone_df["day_of_week"] = zone_df["time"].dt.dayofweek
    zone_df["month"] = zone_df["time"].dt.month

    # Lag features
    zone_df["lag_1"] = zone_df["demand"].shift(1)
    zone_df["lag_24"] = zone_df["demand"].shift(24)
    zone_df.dropna(subset=["lag_1", "lag_24"], inplace=True)

    # Time-based 80/20 split
    unique_times = zone_df["time"].sort_values().unique()
    cutoff_idx = int(len(unique_times) * 0.8)
    cutoff_time = unique_times[cutoff_idx]
    train_df = zone_df[zone_df["time"] < cutoff_time].copy()
    test_df = zone_df[zone_df["time"] >= cutoff_time].copy()

    return zone_df, train_df, test_df


@st.cache_data(show_spinner="Computing predictions...")
def predict_zone(_model, train_df, test_df, zone_id, all_zone_ids):
    """Scale features, encode zone, predict on train/test for a single zone."""
    # Fit scaler on train
    scaler = StandardScaler()
    num_train = scaler.fit_transform(train_df[FEATURE_COLS])
    num_test = scaler.transform(test_df[FEATURE_COLS])

    # Fit OHE on ALL zones so feature vector matches saved model (275 columns)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(pd.DataFrame({"zone_id": all_zone_ids}))
    zone_enc_train = ohe.transform(train_df[["zone_id"]])
    zone_enc_test = ohe.transform(test_df[["zone_id"]])

    X_train = np.hstack([num_train, zone_enc_train])
    X_test = np.hstack([num_test, zone_enc_test])
    y_test = test_df["demand"].values

    preds = _model.predict(X_test)

    return preds, y_test, scaler, ohe, X_train, X_test


@st.cache_data(show_spinner="Computing top zones...")
def compute_top_zones(wide_df, zone_cols, n=5):
    """Compute top N zones by average demand from wide-format data (cheap)."""
    avg_demand = wide_df[zone_cols].mean().sort_values(ascending=False)
    peak_demand = wide_df[zone_cols].max()
    top_ids = avg_demand.head(n).index.tolist()
    top_df = pd.DataFrame({
        "Zone ID": top_ids,
        "Avg Demand": [round(avg_demand[z], 2) for z in top_ids],
        "Peak Demand": [round(peak_demand[z], 2) for z in top_ids],
    })
    return top_df, top_ids


@st.cache_data(show_spinner="Detecting peak hours...")
def compute_peak_hours(wide_df, zone_id):
    """Detect peak hours for a single zone: demand > μ + σ."""
    series = wide_df[["time", zone_id]].copy()
    series.rename(columns={zone_id: "demand"}, inplace=True)
    mu = series["demand"].mean()
    sigma = series["demand"].std()
    threshold = mu + sigma
    peaks = series[series["demand"] > threshold].copy()
    peaks["Zone ID"] = zone_id
    peaks["Threshold"] = round(threshold, 2)
    peaks["Hour"] = peaks["time"].dt.hour
    return peaks[["Zone ID", "time", "demand", "Hour", "Threshold"]]


# ──────────────────── Sidebar: Upload ────────────────────
with st.sidebar:
    st.markdown("### 📁 Dataset")
    uploaded_file = st.file_uploader(
        "Upload EV Charging Dataset", type=["csv"], key="csv_uploader"
    )

# Determine data source
if uploaded_file is not None:
    try:
        peek = pd.read_csv(uploaded_file, nrows=3)
        uploaded_file.seek(0)
        if len(peek.columns) < 3:
            st.error("Invalid dataset format. Please upload a valid EV charging dataset.")
            st.stop()
    except Exception:
        st.error("Invalid dataset format. Please upload a valid EV charging dataset.")
        st.stop()
    using_uploaded = True
else:
    using_uploaded = False

# Load data (cached — fast on re-runs)
wide_df, zone_cols = load_wide_csv(uploaded_file if using_uploaded else None)


# ──────────────────── Load Saved Model ────────────────────
try:
    if "best_model" not in st.session_state:
        st.session_state.best_model = None

    if st.session_state.best_model is None and saved_model_exists():
        print("[STARTUP] Loading saved model from disk...")
        st.session_state.best_model = load_model()
        print("[STARTUP] Model loaded ✅")

    best_model = st.session_state.best_model

    if best_model is None:
        st.error("Model file not found in saved_models/")
        st.stop()

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ──────────────────── Compute Top Zones ────────────────────
top_zones_df, top_zone_ids = compute_top_zones(wide_df, zone_cols)


# ──────────────────── Header ────────────────────
st.markdown('<div class="main-header">⚡ EV Charging Demand Forecasting</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Hourly zone-level demand prediction &middot; Pre-trained Random Forest</div>',
    unsafe_allow_html=True,
)


# ──────────────────── Sidebar: Controls ────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Controls")
    st.success("✅ Model loaded from disk")
    st.markdown("---")
    if using_uploaded:
        st.caption("📄 Dataset: **Uploaded Dataset**")
    else:
        st.caption("📄 Dataset: **Default Dataset**")
    st.caption("Best model: **Random Forest**")

    st.markdown("---")
    st.markdown("### 🎯 Zone Selection")
    selected_zone = st.selectbox(
        "Select zone to analyze:",
        options=top_zone_ids + [z for z in zone_cols if z not in top_zone_ids],
        index=0,
        key="zone_select",
    )


# ──────────────────── Process Selected Zone ────────────────────
zone_df, train_df, test_df = prepare_zone_data(wide_df, selected_zone)
preds, y_test, scaler, ohe, X_train, X_test = predict_zone(
    best_model, train_df, test_df, selected_zone, zone_cols
)

# Build test_preds DataFrame
test_preds = test_df[["time", "zone_id", "demand"]].copy()
test_preds["Predicted"] = preds
test_preds["Residual"] = test_preds["demand"] - test_preds["Predicted"]

# Compute MAE and RMSE for this zone
mae = np.mean(np.abs(y_test - preds))
rmse = np.sqrt(np.mean((y_test - preds) ** 2))


# ──────────────────── 1. Data Summary ────────────────────
st.markdown("---")
st.subheader("📊 Data Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{len(zone_cols)}</div>
            <div class="metric-label">Total Zones</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{len(wide_df):,}</div>
            <div class="metric-label">Hourly Records</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{len(train_df):,}</div>
            <div class="metric-label">Training Samples</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{len(test_df):,}</div>
            <div class="metric-label">Test Samples</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    f"**Time Range:** {wide_df['time'].min().strftime('%Y-%m-%d %H:%M')} → "
    f"{wide_df['time'].max().strftime('%Y-%m-%d %H:%M')}  ·  **Zone:** {selected_zone}"
)


# ──────────────────── 2. Model Performance ────────────────────
st.markdown("---")
st.subheader(f"🏆 Model Performance — Zone {selected_zone}")

m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.5rem;">{mae:.2f}</div>
            <div class="metric-label">MAE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m_col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.5rem;">{rmse:.2f}</div>
            <div class="metric-label">RMSE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m_col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.4rem;">Random Forest</div>
            <div class="metric-label">Pre-trained Model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────── 3. Forecast Visualization ────────────────────
st.markdown("---")
st.subheader(f"📈 Forecast Visualization — Zone {selected_zone}")

train_display = train_df.tail(500)
cutoff_time = test_df["time"].iloc[0] if len(test_df) > 0 else train_df["time"].iloc[-1]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=train_display["time"], y=train_display["demand"],
    mode="lines", name="Train (Actual)",
    line=dict(color="#888888", width=1.5), opacity=0.6,
))

fig.add_trace(go.Scatter(
    x=test_preds["time"], y=test_preds["demand"],
    mode="lines", name="Test (Actual)",
    line=dict(color="#1DB954", width=2),
))

fig.add_trace(go.Scatter(
    x=test_preds["time"], y=test_preds["Predicted"],
    mode="lines", name="RF Predicted",
    line=dict(color="#4ECDC4", width=1.5, dash="dash"),
))

fig.add_shape(
    type="line", x0=cutoff_time, x1=cutoff_time, y0=0, y1=1,
    xref="x", yref="paper",
    line=dict(color="#FFD93D", width=2, dash="dash"),
)
fig.add_annotation(
    x=cutoff_time, y=1, yref="paper",
    text="Train / Test Split", showarrow=False,
    xanchor="left", font=dict(color="#FFD93D"),
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Time", yaxis_title="Demand (kWh)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=480, margin=dict(l=40, r=20, t=50, b=40),
)

st.plotly_chart(fig, use_container_width=True)


# ──────────────────── 4. 7-Day Forward Forecast ────────────────────
st.markdown("---")
st.subheader(f"🔮 7-Day Forward Forecast — Zone {selected_zone}")


@st.cache_data(show_spinner="Running recursive forecast...")
def _cached_forecast(_model, _zone_data, zone_id, _scaler, _ohe):
    return recursive_forecast(
        model=_model, last_known_data=_zone_data, zone_id=zone_id,
        scaler=_scaler, ohe=_ohe, feature_cols=FEATURE_COLS, horizon=168,
    )


forecast_df = _cached_forecast(best_model, zone_df, selected_zone, scaler, ohe)

fig_fc = go.Figure()

hist_tail = zone_df.tail(168)
fig_fc.add_trace(go.Scatter(
    x=hist_tail["time"], y=hist_tail["demand"],
    mode="lines", name="Historical",
    line=dict(color="#1DB954", width=2),
))

fig_fc.add_trace(go.Scatter(
    x=forecast_df["time"], y=forecast_df["demand_forecast"],
    mode="lines", name="7-Day Forecast",
    line=dict(color="#FF6B6B", width=2.5),
))

forecast_start = forecast_df["time"].iloc[0]
fig_fc.add_shape(
    type="line", x0=forecast_start, x1=forecast_start, y0=0, y1=1,
    xref="x", yref="paper",
    line=dict(color="#FFD93D", width=2, dash="dash"),
)
fig_fc.add_annotation(
    x=forecast_start, y=1, yref="paper",
    text="Forecast Start", showarrow=False,
    xanchor="left", font=dict(color="#FFD93D"),
)

fig_fc.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Time", yaxis_title="Demand (kWh)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=450, margin=dict(l=40, r=20, t=50, b=40),
)

st.plotly_chart(fig_fc, use_container_width=True)
st.caption("Forecast model: **Random Forest** · Recursive method (lag_1 & lag_24 from predictions)")


# ──────────────────── 5. Residual Analysis ────────────────────
st.markdown("---")
st.subheader(f"📉 Residual Analysis — Zone {selected_zone}")

res_col1, res_col2 = st.columns(2)

with res_col1:
    fig_hist = px.histogram(
        test_preds, x="Residual", nbins=60,
        color_discrete_sequence=["#4ECDC4"], title="Residual Distribution",
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="#FFD93D", line_width=1.5)
    fig_hist.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with res_col2:
    sample_n = min(2000, len(test_preds))
    fig_scatter = px.scatter(
        test_preds.sample(sample_n, random_state=42),
        x="time", y="Residual", color_discrete_sequence=["#FF6B6B"],
        title="Residuals vs Time", opacity=0.4,
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="#FFD93D", line_width=1.5)
    fig_scatter.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

mean_residual = test_preds["Residual"].mean()
std_residual = test_preds["Residual"].std()

mr_col1, mr_col2, mr_col3 = st.columns(3)
with mr_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.5rem;">{mean_residual:.4f}</div>
            <div class="metric-label">Mean Residual</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with mr_col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.5rem;">{std_residual:.4f}</div>
            <div class="metric-label">Std Residual</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with mr_col3:
    bias_label = "Overestimating" if mean_residual < -0.01 else "Underestimating" if mean_residual > 0.01 else "Well Calibrated"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.3rem;">{bias_label}</div>
            <div class="metric-label">Model Bias</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────── 6. Top 5 Zones Table ────────────────────
st.markdown("---")
st.subheader("🏙️ Top 5 Zones by Average Demand")

st.dataframe(
    top_zones_df.style.background_gradient(cmap="Greens", subset=["Avg Demand"]),
    use_container_width=True,
)


# ──────────────────── 7. Peak Hour Insights ────────────────────
st.markdown("---")
st.subheader(f"⏰ Peak Hour Insights — Zone {selected_zone}")
st.caption("Peak = Demand > (Mean + Std) for selected zone")

peak_df = compute_peak_hours(wide_df, selected_zone)
if peak_df.empty:
    st.info("No peak hours detected for this zone.")
else:
    st.dataframe(peak_df.head(200), use_container_width=True, hide_index=True)
    st.caption(f"Showing {min(200, len(peak_df))} of {len(peak_df)} peak records")


# ──────────────────── Footer ────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; font-size:0.85rem;'>"
    "EV Charging Demand Forecasting Dashboard &middot; Built with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
