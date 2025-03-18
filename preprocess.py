import pandas as pd
import os

data_dir = "data"
races = pd.read_csv(os.path.join(data_dir, "races.csv"))
results = pd.read_csv(os.path.join(data_dir, "results.csv"))
drivers = pd.read_csv(os.path.join(data_dir, "drivers.csv"))
qualifying = pd.read_csv(os.path.join(data_dir, "qualifying.csv"))

# Step 1: Merge results with races
data = results.merge(races[["raceId", "year"]], on="raceId")
print("After merging with races, columns:", data.columns.tolist())

# Step 2: Merge with drivers
data = data.merge(drivers[["driverId", "driverRef"]], on="driverId")
print("After merging with drivers, columns:", data.columns.tolist())

# Step 3: Merge with qualifying, explicitly handle overlapping columns
data = data.merge(qualifying[["raceId", "driverId", "position"]], on=["raceId", "driverId"], suffixes=('', '_qual'))
print("After merging with qualifying, columns:", data.columns.tolist())

# Step 4: Rename qualifying position (which is now position_qual due to suffix)
if "position_qual" in data.columns:
    data = data.rename(columns={"position_qual": "qualifying_position"})
else:
    print("Error: position_qual not found after merge. Check qualifying.csv.")
    print("Current columns:", data.columns.tolist())
    exit(1)
print("After renaming qualifying position, columns:", data.columns.tolist())

# Step 5: Set is_winner using position from results.csv
if "position" in data.columns:
    data["is_winner"] = (data["position"] == 1).astype(int)
else:
    print("Error: position not found. Check results.csv.")
    print("Current columns:", data.columns.tolist())
    exit(1)
print("After setting is_winner, columns:", data.columns.tolist())

# Step 6: Sort by year and raceId for chronological order
data = data.sort_values(["year", "raceId"])

# Step 7: Calculate past_wins
data["past_wins"] = data.groupby("driverId")["is_winner"].shift().cumsum().fillna(0)
print("After calculating past_wins, columns:", data.columns.tolist())

# Step 8: Select required columns
required_cols = ["driverRef", "qualifying_position", "past_wins", "is_winner"]
if all(col in data.columns for col in required_cols):
    dataset = data[required_cols]
else:
    missing = [col for col in required_cols if col not in data.columns]
    print(f"Error: Missing columns {missing}.")
    print("Current columns:", data.columns.tolist())
    exit(1)

# Step 9: Save dataset
dataset.to_csv("dataset.csv", index=False)
print(f"\nDataset saved! Shape: {dataset.shape}")