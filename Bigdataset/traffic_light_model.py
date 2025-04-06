import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the original dataset
df = pd.read_csv("traffic_export.csv")
df["DateTime"] = pd.to_datetime(df["DateTime"])
df["hour"] = df["DateTime"].dt.hour
df["weekday"] = df["DateTime"].dt.weekday
df["is_rainy"] = 0
df["delay_seconds"] = df["Vehicles"] * 1.5

X = df[["hour", "weekday", "is_rainy", "Junction"]].rename(columns={"Junction": "node_id"})
y = df["delay_seconds"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ✅ Save using your current scikit-learn version
joblib.dump(model, "delay_predictor_updated.pkl")
print("✅ Model retrained and saved!")
