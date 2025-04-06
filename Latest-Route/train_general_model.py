import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# ✅ Load the simulated dataset
df = pd.read_csv("simulated_traffic_data.csv")

# ✅ Define features and target
X = df[["hour", "weekday", "is_rainy", "node_id"]]
y = df["delay_seconds"]

# ✅ Train the model using your version of scikit-learn
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ✅ Save the model (now it's compatible with your current version)
joblib.dump(model, "generalized_delay_predictor.pkl")

print("✅ Model retrained and saved locally using your system!")
