# ⚡ EV Charging Demand Forecasting Dashboard

Hourly zone-level EV charging demand prediction using Linear Regression and Random Forest, deployed as an interactive Streamlit dashboard.

## 📊 Features

- **Data Pipeline** — Loads wide-format CSV → long format → temporal + lag feature engineering → time-based train/test split
- **Two Models** — Linear Regression (baseline) vs Random Forest Regressor (n=50, depth=15)
- **Model Persistence** — Best model saved via joblib; loads instantly on restart without retraining
- **CSV Upload** — Upload custom datasets via sidebar; validates format and trains fresh
- **7-Day Recursive Forecast** — 168-hour ahead predictions per zone using autoregressive lag features
- **Residual Analysis** — Distribution histogram, scatter plot, mean/std/bias metrics
- **Feature Importance** — Top 15 features from Random Forest
- **Peak Detection** — Top 5 zones by avg demand + peak hour identification (μ + σ threshold)

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit, Plotly |
| ML | Scikit-Learn (LinearRegression, RandomForestRegressor) |
| Data | Pandas, NumPy |
| Persistence | Joblib |

## 📁 Project Structure

```
ev-demand-dashboard/
├── app.py              # Streamlit dashboard (8 sections)
├── preprocessing.py    # Data loading, wide→long, features, lags, split
├── models.py           # LR + RF training, save/load via joblib
├── evaluation.py       # MAE/RMSE computation + model comparison
├── forecasting.py      # Recursive 7-day forward forecast
├── peak_analysis.py    # Top 5 zones + peak hour detection
├── requirements.txt    # Dependencies
├── report.tex          # LaTeX report
└── screenshots/        # Dashboard screenshots
```

## 🚀 Setup & Run

```bash
# Clone
git clone https://github.com/White-Devil2839/ev-demand-dashboard.git
cd ev-demand-dashboard

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

## 📂 Dataset

**Source:** UrbanEV — Zone-Cleaned Aggregated Hourly (`volume.csv`)

| Metric | Value |
|--------|-------|
| Zones | 275 |
| Time Range | Sep 2022 – Feb 2023 |
| Observations | 1,188,000 |
| Train / Test | 950,400 / 237,600 (80/20 time-based) |

## 📈 Features Used

| Feature | Description |
|---------|-------------|
| `hour` | Hour of day (0–23) |
| `day_of_week` | Day of week (0–6) |
| `month` | Month (1–12) |
| `lag_1` | Demand at t-1 |
| `lag_24` | Demand at t-24 |
| `zone_id` | One-hot encoded (275 zones) |

## 👤 Author

**Divyansh Choudhary**

## 📜 License

MIT