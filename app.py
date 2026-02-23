"""
app.py
Streamlit Dashboard: EV Charging Demand Forecasting & Load Intelligence
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from preprocessing import (
    load_data,
    wide_to_long,
    engineer_features,
    train_test_split_time,
)
from models import (
    train_linear_regression,
    train_random_forest,
    save_model,
    load_model,
    saved_model_exists,
)
from evaluation import evaluate_model, compare_models
from peak_analysis import top_zones, detect_peak_hours
from forecasting import recursive_forecast
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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


# ──────────────────── Cached Data Pipeline ────────────────────
@st.cache_data(show_spinner="Loading and processing data...")
def load_and_prepare_data(csv_source=None):
    """Load data, engineer features, split, and prepare.
    csv_source: file path (str) or uploaded file bytes. Default: VOLUME_CSV.
    """
    print("\n" + "="*60)
    print("[PIPELINE] Starting data pipeline...")
    print("="*60)
    if csv_source is None:
        raw_df = load_data(VOLUME_CSV)
    elif isinstance(csv_source, str):
        raw_df = load_data(csv_source)
    else:
        # Uploaded file (BytesIO / UploadedFile)
        print("[PIPELINE] Reading uploaded CSV...")
        raw_df = pd.read_csv(csv_source)
        print(f"[PIPELINE] Uploaded: {len(raw_df)} rows, {len(raw_df.columns)} columns")
    long_df = wide_to_long(raw_df)
    feat_df = engineer_features(long_df)
    train_df, test_df = train_test_split_time(feat_df)

    feature_cols = ["hour", "day_of_week", "month", "lag_1", "lag_24"]

    # Fit transformers
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    zone_train = ohe.fit_transform(train_df[["zone_id"]])
    zone_test = ohe.transform(test_df[["zone_id"]])
    zone_feature_names = ohe.get_feature_names_out(["zone_id"]).tolist()

    scaler = StandardScaler()
    num_train = scaler.fit_transform(train_df[feature_cols])
    num_test = scaler.transform(test_df[feature_cols])

    X_train = np.hstack([num_train, zone_train])
    X_test = np.hstack([num_test, zone_test])
    y_train = train_df["demand"].values
    y_test = test_df["demand"].values
    feature_names = feature_cols + zone_feature_names

    # Top zones & peak hours (from full data)
    top_zones_df = top_zones(feat_df)
    top_zone_ids = top_zones_df["Zone ID"].tolist()
    peak_hours_df = detect_peak_hours(feat_df, top_zone_ids)

    # Summary stats
    summary = {
        "total_zones": feat_df["zone_id"].nunique(),
        "total_observations": len(feat_df),
        "time_start": feat_df["time"].min(),
        "time_end": feat_df["time"].max(),
        "train_size": len(train_df),
        "test_size": len(test_df),
    }

    print(f"[PIPELINE] Data ready: {summary['total_zones']} zones, "
          f"{summary['total_observations']:,} observations")
    print(f"[PIPELINE] Train: {summary['train_size']:,} | Test: {summary['test_size']:,}")
    print("="*60 + "\n")
    return (
        feat_df, train_df, test_df,
        X_train, X_test, y_train, y_test,
        feature_names, ohe, scaler,
        top_zones_df, top_zone_ids, peak_hours_df,
        summary,
    )


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train both models, evaluate, save best."""
    lr_model = train_linear_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    lr_results = evaluate_model(lr_model, X_test, y_test)
    rf_results = evaluate_model(rf_model, X_test, y_test)

    comparison_df = compare_models({
        "Linear Regression": lr_results,
        "Random Forest": rf_results,
    })

    # Determine and save best model
    best_name = comparison_df.loc[comparison_df["Best"] == "✅", "Model"].values[0]
    best_model = rf_model if best_name == "Random Forest" else lr_model
    save_model(best_model)

    return lr_model, rf_model, lr_results, rf_results, comparison_df, best_name


# ──────────────────── Sidebar: Upload ────────────────────
with st.sidebar:
    st.markdown("### 📁 Dataset")
    uploaded_file = st.file_uploader(
        "Upload EV Charging Dataset", type=["csv"], key="csv_uploader"
    )

# Determine data source and detect dataset changes
if uploaded_file is not None:
    # Validate: must have a time-like first column + at least 2 zone columns
    try:
        peek_df = pd.read_csv(uploaded_file, nrows=5)
        uploaded_file.seek(0)  # reset for re-read
        if len(peek_df.columns) < 3:
            st.error("Invalid dataset format. Please upload a valid EV charging dataset.")
            st.stop()
        print(f"[UPLOAD] Validated CSV: {len(peek_df.columns)} columns")
    except Exception:
        st.error("Invalid dataset format. Please upload a valid EV charging dataset.")
        st.stop()

    csv_key = f"uploaded_{uploaded_file.name}_{uploaded_file.size}"
    using_uploaded = True
else:
    csv_key = "default"
    using_uploaded = False

# If dataset changed, reset models so they retrain on the new data
if "current_csv_key" not in st.session_state:
    st.session_state.current_csv_key = csv_key  # Initialize to current — no false trigger on first load

if st.session_state.current_csv_key != csv_key:
    st.session_state.current_csv_key = csv_key
    st.session_state.models_trained = False
    st.session_state.best_model_loaded_from_disk = False
    print(f"[STARTUP] Dataset changed → resetting models (key={csv_key})")

# Load data
(
    feat_df, train_df, test_df,
    X_train, X_test, y_train, y_test,
    feature_names, ohe, scaler,
    top_zones_df, top_zone_ids, peak_hours_df,
    summary,
) = load_and_prepare_data(uploaded_file if using_uploaded else None)


# ──────────────────── Model Persistence Logic ────────────────────
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False
    st.session_state.lr_model = None
    st.session_state.rf_model = None
    st.session_state.lr_results = None
    st.session_state.rf_results = None
    st.session_state.comparison_df = None
    st.session_state.best_name = None
    st.session_state.best_model = None
    st.session_state.best_model_loaded_from_disk = False


def _do_training():
    """Run training and store results in session state."""
    print("\n" + "="*60)
    print("[TRAINING] Starting model training...")
    print("="*60)
    (
        st.session_state.lr_model,
        st.session_state.rf_model,
        st.session_state.lr_results,
        st.session_state.rf_results,
        st.session_state.comparison_df,
        st.session_state.best_name,
    ) = train_and_evaluate(X_train, X_test, y_train, y_test)
    # Set best_model from freshly trained models
    best_n = st.session_state.best_name
    st.session_state.best_model = (
        st.session_state.rf_model if best_n == "Random Forest"
        else st.session_state.lr_model
    )
    st.session_state.models_trained = True
    print(f"[TRAINING] Best model: {st.session_state.best_name}")
    print("[TRAINING] Training complete!")
    print("="*60 + "\n")


# On first load: load saved model OR train from scratch
if not st.session_state.models_trained:
    if not using_uploaded and saved_model_exists():
        print("[STARTUP] Saved model found — loading from disk...")
        saved = load_model()
        if saved is not None:
            st.session_state.best_model = saved
            st.session_state.best_model_loaded_from_disk = True
            st.session_state.models_trained = True
            print("[STARTUP] Model loaded — skipping retraining ✅")
        else:
            _do_training()
    else:
        if using_uploaded:
            print("[STARTUP] Uploaded dataset — training from scratch")
        else:
            print("[STARTUP] No saved model — training from scratch")
        _do_training()

# Convenience aliases — safe access via .get()
lr_model = st.session_state.get("lr_model")
rf_model = st.session_state.get("rf_model")
lr_results = st.session_state.get("lr_results")
rf_results = st.session_state.get("rf_results")
comparison_df = st.session_state.get("comparison_df")
best_name = st.session_state.get("best_name")
best_model = st.session_state.get("best_model")

# Determine best predictions key (for residuals)
best_preds_key = "RF_Predicted" if best_name == "Random Forest" else "LR_Predicted"

# Feature importance — only if rf_model was trained
feat_imp_df = None
if rf_model is not None:
    feat_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)

# Test predictions — only if results exist
test_preds = None
if lr_results is not None and rf_results is not None:
    test_preds = test_df[["time", "zone_id", "demand"]].copy()
    test_preds["LR_Predicted"] = lr_results["predictions"]
    test_preds["RF_Predicted"] = rf_results["predictions"]
    test_preds["Residual"] = test_preds["demand"] - test_preds[best_preds_key]


# ──────────────────── Header ────────────────────
st.markdown('<div class="main-header">⚡ EV Charging Demand Forecasting</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Hourly zone-level demand prediction &middot; Linear Regression vs Random Forest</div>',
    unsafe_allow_html=True,
)


# ──────────────────── Sidebar: Controls ────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Controls")
    if st.session_state.best_model_loaded_from_disk:
        st.success("✅ Model loaded from disk (no retraining)")
    elif saved_model_exists():
        st.success("✅ Saved model available")
    else:
        st.info("No saved model — trained from scratch")

    if st.button("🔄 Retrain Model", use_container_width=True):
        _do_training()
        st.session_state.best_model_loaded_from_disk = False
        st.rerun()

    st.markdown("---")
    # Dataset source indicator
    if using_uploaded:
        st.caption("📄 Dataset: **Uploaded Dataset**")
    else:
        st.caption("📄 Dataset: **Default Dataset**")
    if best_name:
        st.caption(f"Best model: **{best_name}**")
    else:
        st.caption("Model loaded from disk")


# ──────────────────── 1. Data Summary ────────────────────
st.markdown("---")
st.subheader("📊 Data Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{summary["total_zones"]}</div>
            <div class="metric-label">Total Zones</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{summary["total_observations"]:,}</div>
            <div class="metric-label">Total Observations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{summary["train_size"]:,}</div>
            <div class="metric-label">Training Samples</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{summary["test_size"]:,}</div>
            <div class="metric-label">Test Samples</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    f"**Time Range:** {summary['time_start'].strftime('%Y-%m-%d %H:%M')} → "
    f"{summary['time_end'].strftime('%Y-%m-%d %H:%M')}"
)


# ──────────────────── 2. Model Comparison ────────────────────
st.markdown("---")
st.subheader("🏆 Model Comparison")

if comparison_df is not None:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.dataframe(
            comparison_df.style.apply(
                lambda row: [
                    "background-color: rgba(29,185,84,0.15); font-weight: bold"
                    if row["Best"] == "✅"
                    else ""
                    for _ in row
                ],
                axis=1,
            ),
            use_container_width=True,
            hide_index=True,
        )

    with col_right:
        display_name = best_name or "Loaded Model"
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top: 0.5rem;">
                <div class="metric-label">Best Model</div>
                <div class="metric-value" style="font-size: 1.4rem;">{display_name}</div>
                <div class="metric-label">Based on lowest RMSE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("ℹ️ Model loaded from disk — click **Retrain Model** in the sidebar to see comparison metrics.")


# ──────────────────── 3. Forecast Visualization with Train/Test Split ────────────────────
st.markdown("---")
st.subheader("📈 Forecast Visualization — Top 5 Zones")

selected_zone = st.selectbox(
    "Select a zone to visualize:",
    options=top_zone_ids,
    index=0,
)

# --- Build combined chart: Train + Test + Predictions ---
train_zone = train_df[train_df["zone_id"] == selected_zone].sort_values("time")

if test_preds is not None:
    test_zone = test_preds[test_preds["zone_id"] == selected_zone].sort_values("time")
else:
    test_zone = test_df[test_df["zone_id"] == selected_zone].sort_values("time")

# Train/test cutoff
cutoff_time = test_zone["time"].iloc[0] if len(test_zone) > 0 else train_zone["time"].iloc[-1]

fig = go.Figure()

# Train actual (show last 500 points for readability)
train_display = train_zone.tail(500)
fig.add_trace(go.Scatter(
    x=train_display["time"],
    y=train_display["demand"],
    mode="lines",
    name="Train (Actual)",
    line=dict(color="#888888", width=1.5),
    opacity=0.6,
))

# Test actual
fig.add_trace(go.Scatter(
    x=test_zone["time"],
    y=test_zone["demand"],
    mode="lines",
    name="Test (Actual)",
    line=dict(color="#1DB954", width=2),
))

# LR predictions
if test_preds is not None and "LR_Predicted" in test_zone.columns:
    fig.add_trace(go.Scatter(
        x=test_zone["time"],
        y=test_zone["LR_Predicted"],
        mode="lines",
        name="Linear Regression",
        line=dict(color="#FF6B6B", width=1.5, dash="dot"),
    ))

    # RF predictions
    fig.add_trace(go.Scatter(
        x=test_zone["time"],
        y=test_zone["RF_Predicted"],
        mode="lines",
        name="Random Forest",
        line=dict(color="#4ECDC4", width=1.5, dash="dash"),
    ))

# Vertical line at train/test split
fig.add_shape(
    type="line",
    x0=cutoff_time, x1=cutoff_time,
    y0=0, y1=1,
    xref="x", yref="paper",
    line=dict(color="#FFD93D", width=2, dash="dash"),
)
fig.add_annotation(
    x=cutoff_time, y=1, yref="paper",
    text="Train / Test Split",
    showarrow=False, xanchor="left",
    font=dict(color="#FFD93D"),
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Time",
    yaxis_title="Demand (kWh)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=480,
    margin=dict(l=40, r=20, t=50, b=40),
)

st.plotly_chart(fig, use_container_width=True)


# ──────────────────── 4. 7-Day Forward Forecast ────────────────────
st.markdown("---")
st.subheader("🔮 7-Day Forward Forecast (168 Hours)")

forecast_zone = st.selectbox(
    "Select a zone to forecast:",
    options=top_zone_ids,
    index=0,
    key="forecast_zone_select",
)

# Get last known data for this zone
zone_full_data = feat_df[feat_df["zone_id"] == forecast_zone].sort_values("time")


@st.cache_data(show_spinner="Running recursive forecast...")
def _cached_forecast(_model, _zone_data, zone_id, _scaler, _ohe):
    """Cached wrapper around recursive_forecast."""
    return recursive_forecast(
        model=_model,
        last_known_data=_zone_data,
        zone_id=zone_id,
        scaler=_scaler,
        ohe=_ohe,
        feature_cols=["hour", "day_of_week", "month", "lag_1", "lag_24"],
        horizon=168,
    )


forecast_df = _cached_forecast(best_model, zone_full_data, forecast_zone, scaler, ohe)

# Build forecast plot
fig_fc = go.Figure()

# Historical tail (last 168 hours for context)
hist_tail = zone_full_data.tail(168)

fig_fc.add_trace(go.Scatter(
    x=hist_tail["time"],
    y=hist_tail["demand"],
    mode="lines",
    name="Historical",
    line=dict(color="#1DB954", width=2),
))

# Test predictions for this zone (if available)
if test_preds is not None:
    test_zone_fc = test_preds[test_preds["zone_id"] == forecast_zone].sort_values("time")
    if len(test_zone_fc) > 0:
        fig_fc.add_trace(go.Scatter(
            x=test_zone_fc["time"],
            y=test_zone_fc[best_preds_key],
            mode="lines",
            name="Test Predictions",
            line=dict(color="#4ECDC4", width=1.5, dash="dash"),
        ))

# Future forecast
fig_fc.add_trace(go.Scatter(
    x=forecast_df["time"],
    y=forecast_df["demand_forecast"],
    mode="lines",
    name="7-Day Forecast",
    line=dict(color="#FF6B6B", width=2.5),
))

# Vertical line at forecast start
forecast_start = forecast_df["time"].iloc[0]
fig_fc.add_shape(
    type="line",
    x0=forecast_start, x1=forecast_start,
    y0=0, y1=1,
    xref="x", yref="paper",
    line=dict(color="#FFD93D", width=2, dash="dash"),
)
fig_fc.add_annotation(
    x=forecast_start, y=1, yref="paper",
    text="Forecast Start",
    showarrow=False, xanchor="left",
    font=dict(color="#FFD93D"),
)

fig_fc.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="Time",
    yaxis_title="Demand (kWh)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=450,
    margin=dict(l=40, r=20, t=50, b=40),
)

st.plotly_chart(fig_fc, use_container_width=True)

st.caption(f"Forecast model: **{best_name}** · Recursive method (lag_1 & lag_24 from predictions)")


# ──────────────────── 5. Residual Analysis ────────────────────
st.markdown("---")
st.subheader(f"📉 Residual Analysis ({best_name or 'Loaded Model'})")

if test_preds is not None:
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        # Residual distribution histogram
        fig_hist = px.histogram(
            test_preds,
            x="Residual",
            nbins=60,
            color_discrete_sequence=["#4ECDC4"],
            title="Residual Distribution",
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#FFD93D", line_width=1.5)
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with res_col2:
        # Residual vs time scatter plot
        fig_scatter = px.scatter(
            test_preds.sample(min(2000, len(test_preds)), random_state=42),
            x="time",
            y="Residual",
            color_discrete_sequence=["#FF6B6B"],
            title="Residuals vs Time",
            opacity=0.4,
        )
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="#FFD93D", line_width=1.5)
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Mean residual metric
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
else:
    st.info("ℹ️ Retrain model to view residual analysis.")


# ──────────────────── 6. Feature Importance ────────────────────
st.markdown("---")
st.subheader("🌲 Feature Importance (Random Forest)")

if feat_imp_df is not None:
    top_n_feat = feat_imp_df.head(15)

    fig_imp = px.bar(
        top_n_feat,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Emrld",
    )

    fig_imp.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info("ℹ️ Retrain model to view feature importance.")


# ──────────────────── 7. Top 5 Zones Table ────────────────────
st.markdown("---")
st.subheader("🏙️ Top 5 Zones by Average Demand")

st.dataframe(
    top_zones_df.style.background_gradient(
        cmap="Greens", subset=["Avg Demand"]
    ),
    use_container_width=True,
)


# ──────────────────── 8. Peak Hour Insights ────────────────────
st.markdown("---")
st.subheader("⏰ Peak Hour Insights")
st.caption("Peak = Demand > (Mean + Std) for each zone")

if peak_hours_df.empty:
    st.info("No peak hours detected for the top zones.")
else:
    peak_zone_selector = st.selectbox(
        "Filter by zone:",
        options=["All"] + top_zone_ids,
        index=0,
        key="peak_zone_select",
    )

    if peak_zone_selector == "All":
        display_peaks = peak_hours_df.copy()
    else:
        display_peaks = peak_hours_df[
            peak_hours_df["Zone ID"] == peak_zone_selector
        ].copy()

    st.dataframe(
        display_peaks.head(200),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(f"Showing {min(200, len(display_peaks))} of {len(display_peaks)} peak records")


# ──────────────────── Footer ────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; font-size:0.85rem;'>"
    "EV Charging Demand Forecasting Dashboard &middot; Built with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
