"""
models.py
Train Linear Regression and Random Forest Regressor models.
Single global model across all zones.
Model persistence via joblib.
"""

import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


def train_linear_regression(X_train, y_train):
    """Train and return a Linear Regression model."""
    print(f"[MODEL] Training Linear Regression on {X_train.shape[0]} samples...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("[MODEL] Linear Regression training complete")
    return model


def train_random_forest(X_train, y_train):
    """Train and return a Random Forest Regressor model."""
    print(f"[MODEL] Training Random Forest (n=50, depth=15) on {X_train.shape[0]} samples...")
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[MODEL] Random Forest training complete")
    return model


def save_model(model, filename: str = "best_model.joblib"):
    """Save a model to disk using joblib."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, path)
    print(f"[MODEL] Saved best model to {path}")
    return path


def load_model(filename: str = "best_model.joblib"):
    """Load a saved model from disk. Returns None if not found."""
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        print(f"[MODEL] Loading saved model from {path}...")
        model = joblib.load(path)
        print("[MODEL] Model loaded successfully")
        return model
    print(f"[MODEL] No saved model found at {path}")
    return None


def saved_model_exists(filename: str = "best_model.joblib") -> bool:
    """Check whether a saved model file exists."""
    return os.path.exists(os.path.join(MODEL_DIR, filename))
