"""
forecasting.py
Recursive forward forecasting for 7 days (168 hours).
"""

import pandas as pd
import numpy as np


def recursive_forecast(
    model,
    last_known_data: pd.DataFrame,
    zone_id: str,
    scaler,
    ohe,
    feature_cols: list,
    horizon: int = 168,
) -> pd.DataFrame:
    """
    Produce a recursive forecast for `horizon` hours into the future
    for a single zone.

    Args:
        model: trained sklearn model
        last_known_data: DataFrame with columns [time, zone_id, demand,
                         hour, day_of_week, month, lag_1, lag_24]
                         sorted by time, for the target zone
        zone_id: the zone to forecast
        scaler: fitted StandardScaler (from prepare_features)
        ohe: fitted OneHotEncoder (from prepare_features)
        feature_cols: list of numerical feature col names
                      ["hour", "day_of_week", "month", "lag_1", "lag_24"]
        horizon: number of hours to forecast (default 168 = 7 days)

    Returns:
        DataFrame with [time, zone_id, demand_forecast]
    """
    # We need at least 24 recent rows to seed lag_24
    recent = last_known_data.tail(24).copy()
    demand_history = recent["demand"].tolist()

    last_time = recent["time"].iloc[-1]
    forecasts = []

    for step in range(horizon):
        # Next timestamp
        next_time = last_time + pd.Timedelta(hours=step + 1)

        # Temporal features
        hour = next_time.hour
        day_of_week = next_time.dayofweek
        month = next_time.month

        # Lag features from history
        lag_1 = demand_history[-1]
        lag_24 = demand_history[-24] if len(demand_history) >= 24 else demand_history[0]

        # Build feature row as DataFrames with proper column names
        num_row = pd.DataFrame(
            [[hour, day_of_week, month, lag_1, lag_24]],
            columns=feature_cols,
        )
        num_scaled = scaler.transform(num_row)
        zone_row = pd.DataFrame([[zone_id]], columns=["zone_id"])
        zone_encoded = ohe.transform(zone_row)
        X_row = np.hstack([num_scaled, zone_encoded])

        # Predict
        pred = model.predict(X_row)[0]
        pred = max(pred, 0.0)  # demand can't be negative

        demand_history.append(pred)

        forecasts.append({
            "time": next_time,
            "zone_id": zone_id,
            "demand_forecast": round(pred, 2),
        })

    return pd.DataFrame(forecasts)
