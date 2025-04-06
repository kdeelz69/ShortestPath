import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your dataset
df = pd.read_csv("traffic_light_data.csv")

# Features and target
X = df[["hour", "weekday", "is_rainy", "node_id"]]
y = df["delay_seconds"]

# Train a Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model to disk
joblib.dump(model, "delay_predictor.pkl")
print("✅ Model trained and saved as delay_predictor.pkl")



