import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv("dataset.csv")
print("Loaded dataset columns:", data.columns.tolist())

# Features and target
X = data[["qualifying_position", "past_wins"]]
y = data["is_winner"]

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel trained and saved!")