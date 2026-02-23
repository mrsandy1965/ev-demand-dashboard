"""
peak_analysis.py
Top 5 zone ranking and peak hour detection.
"""

import pandas as pd


def top_zones(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Rank zones by average demand using full historical data.
    
    Average Demand = (1/T) * sum of hourly demand
    Peak Demand = max hourly demand
    
    Returns top N zones as a DataFrame:
    | Rank | Zone ID | Avg Demand | Peak Demand |
    """
    zone_stats = df.groupby("zone_id")["demand"].agg(
        Avg_Demand="mean",
        Peak_Demand="max",
    ).reset_index()

    zone_stats.sort_values(by="Avg_Demand", ascending=False, inplace=True)
    zone_stats = zone_stats.head(n).reset_index(drop=True)
    zone_stats.index += 1
    zone_stats.index.name = "Rank"
    zone_stats.rename(columns={
        "zone_id": "Zone ID",
        "Avg_Demand": "Avg Demand",
        "Peak_Demand": "Peak Demand",
    }, inplace=True)

    zone_stats["Avg Demand"] = zone_stats["Avg Demand"].round(2)
    zone_stats["Peak Demand"] = zone_stats["Peak Demand"].round(2)

    return zone_stats


def detect_peak_hours(df: pd.DataFrame, top_zone_ids: list) -> pd.DataFrame:
    """
    For each top zone, detect peak hours where:
        demand > (mean + standard deviation)
    
    Returns DataFrame:
    | Zone ID | Peak Timestamp | Demand |
    """
    records = []

    for zone_id in top_zone_ids:
        zone_data = df[df["zone_id"] == zone_id].copy()
        mean_demand = zone_data["demand"].mean()
        std_demand = zone_data["demand"].std()
        threshold = mean_demand + std_demand

        peaks = zone_data[zone_data["demand"] > threshold]

        for _, row in peaks.iterrows():
            records.append({
                "Zone ID": zone_id,
                "Peak Timestamp": row["time"],
                "Demand (kWh)": round(row["demand"], 2),
            })

    return pd.DataFrame(records)