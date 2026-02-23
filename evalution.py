"""
evaluation.py
Model evaluation: MAE and RMSE computation on test set.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate a model on the test set.
    Returns dict with 'mae', 'rmse', and 'predictions'.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"[EVAL] MAE={mae:.4f}  RMSE={rmse:.4f}")
    return {"mae": mae, "rmse": rmse, "predictions": y_pred}


def compare_models(results: dict) -> pd.DataFrame:
    """
    Compare multiple models.
    
    Args:
        results: {"Model Name": {"mae": ..., "rmse": ...}, ...}
    
    Returns:
        DataFrame with MAE, RMSE, and 'Best' column.
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model": name,
            "MAE": round(metrics["mae"], 4),
            "RMSE": round(metrics["rmse"], 4),
        })

    df = pd.DataFrame(rows)

    # Highlight best model (lowest RMSE)
    best_idx = df["RMSE"].idxmin()
    df["Best"] = ""
    df.loc[best_idx, "Best"] = "✅"

    return df