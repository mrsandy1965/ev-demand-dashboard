"""
preprocessing.py
Data loading, wide-to-long conversion, feature engineering,
lag feature creation, train-test split, and feature preparation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(path: str) -> pd.DataFrame:
    """Load volume.csv from the given path."""
    print(f"[PREPROCESS] Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"[PREPROCESS] Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide format to long format.
    
    From: | time | zone1 | zone2 | ... |
    To:   | time | zone_id | demand |
    """
    # The first column is 'time', remaining columns are zone IDs
    time_col = df.columns[0]
    zone_cols = df.columns[1:]

    print(f"[PREPROCESS] Converting wide → long ({len(zone_cols)} zones)...")
    df_long = df.melt(
        id_vars=[time_col],
        value_vars=zone_cols,
        var_name="zone_id",
        value_name="demand",
    )
    df_long.rename(columns={time_col: "time"}, inplace=True)
    print(f"[PREPROCESS] Long format: {len(df_long)} rows")
    return df_long


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert time to datetime, extract temporal features,
    create lag features per zone, and drop NaN rows.
    """
    print("[PREPROCESS] Engineering features...")
    df = df.copy()

    # Convert time column to datetime
    df["time"] = pd.to_datetime(df["time"])

    # SORT by zone_id and time BEFORE creating lag features
    df.sort_values(by=["zone_id", "time"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Extract temporal features
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    print("[PREPROCESS] Extracted hour, day_of_week, month")

    # Create lag features PER ZONE using groupby + shift
    df["lag_1"] = df.groupby("zone_id")["demand"].shift(1)
    df["lag_24"] = df.groupby("zone_id")["demand"].shift(24)
    print("[PREPROCESS] Created lag_1 and lag_24 per zone")

    # Drop rows with null lag values
    before = len(df)
    df.dropna(subset=["lag_1", "lag_24"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[PREPROCESS] Dropped {before - len(df)} NaN lag rows → {len(df)} remaining")

    return df


def train_test_split_time(df: pd.DataFrame, ratio: float = 0.8):
    """
    Time-based train-test split.
    Sort by time, use first 80% of timeline for training,
    last 20% for testing. No random split; no data leakage.
    """
    df = df.sort_values(by="time").reset_index(drop=True)

    # Find cutoff timestamp
    unique_times = df["time"].sort_values().unique()
    cutoff_idx = int(len(unique_times) * ratio)
    cutoff_time = unique_times[cutoff_idx]

    train_df = df[df["time"] < cutoff_time].copy()
    test_df = df[df["time"] >= cutoff_time].copy()

    return train_df, test_df


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Prepare features for modeling:
    - One-Hot Encode zone_id
    - Scale numerical features
    Returns X_train, X_test, y_train, y_test, feature_names
    """
    feature_cols = ["hour", "day_of_week", "month", "lag_1", "lag_24"]
    target_col = "demand"

    # --- One-Hot Encode zone_id ---
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    zone_train = ohe.fit_transform(train_df[["zone_id"]])
    zone_test = ohe.transform(test_df[["zone_id"]])
    zone_feature_names = ohe.get_feature_names_out(["zone_id"]).tolist()

    # --- Scale numerical features ---
    scaler = StandardScaler()
    num_train = scaler.fit_transform(train_df[feature_cols])
    num_test = scaler.transform(test_df[feature_cols])

    # --- Combine ---
    X_train = np.hstack([num_train, zone_train])
    X_test = np.hstack([num_test, zone_test])

    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    feature_names = feature_cols + zone_feature_names

    return X_train, X_test, y_train, y_test, feature_names