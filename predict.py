import requests
import pickle
import pandas as pd
from collections import defaultdict

# Fetch session_key for the qualifying session of the next race (e.g., China)
url = "https://api.openf1.org/v1/sessions"
params = {"country_name": "China", "year": 2025, "session_name": "Qualifying"}
response = requests.get(url, params=params)
sessions = response.json()
qualifying_session = next((s for s in sessions if s["session_name"] == "Qualifying"), None)
if not qualifying_session:
    print("\nQualifying Session Not Found!...\n")
    exit(1)
session_key = qualifying_session["session_key"]

# Fetch all laps for the qualifying session
laps_url = f"https://api.openf1.org/v1/laps?session_key={session_key}"
response = requests.get(laps_url)
laps = response.json()

# Find each driver's best lap time
driver_best_times = defaultdict(lambda: {"time": float("inf"), "lap_number": None})
for lap in laps:
    driver_number = lap["driver_number"]
    lap_time = lap["time"]  # Assuming time is in seconds
    if lap_time < driver_best_times[driver_number]["time"]:
        driver_best_times[driver_number]["time"] = lap_time
        driver_best_times[driver_number]["lap_number"] = lap["lap_number"]

# Sort drivers by their best lap times for grid positions
driver_times = [(driver_number, data["time"]) for driver_number, data in driver_best_times.items() if data["time"] != float("inf")]
driver_times_sorted = sorted(driver_times, key=lambda x: x[1])
grid_positions = {driver_number: position + 1 for position, (driver_number, time) in enumerate(driver_times_sorted)}

# Map driver numbers to names
drivers_url = "https://api.openf1.org/v1/driver"
response = requests.get(drivers_url)
drivers = response.json()
driver_number_to_ref = {str(driver["number"]): driver["driver_ref"] for driver in drivers}

# Create grid DataFrame
data = []
for driver_number, position in grid_positions.items():
    driver_ref = driver_number_to_ref.get(driver_number, "Unknown")
    data.append((driver_ref, position))
grid_df = pd.DataFrame(data, columns=["driverRef", "grid_position"])
grid_df = grid_df.sort_values("grid_position")

# Load historical data to get past wins
historical_data = pd.read_csv("dataset.csv")
driver_past_wins = historical_data.groupby("driverRef")["is_winner"].sum().to_dict()

# Create new_race DataFrame
new_race = pd.merge(grid_df, pd.Series(driver_past_wins, name="past_wins"), left_on="driverRef", right_index=True)
new_race = new_race.set_index("driverRef")
new_race = new_race[["grid_position", "past_wins"]]
new_race = new_race.rename(columns={"grid_position": "qualifying_position"})

# Load the model and predict
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
probs = model.predict_proba(new_race)[:, 1]

# Show the results
print("\nWin Probabilities for the Next Race based on Real-Time Qualifying Data and Historical Wins:")
for driver, prob in zip(new_race.index, probs):
    print(f"{driver}: {prob:.2f}")