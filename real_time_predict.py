import requests
import pandas as pd
from collections import defaultdict
import numpy as np

# Configure the race
RACE_COUNTRY = "China"
RACE_YEAR = 2025

# Fetch session key from OpenF1 API
def fetch_session_key(country, year, session_name):
    url = "https://api.openf1.org/v1/sessions"
    params = {"country_name": country, "year": year, "session_name": session_name}
    response = requests.get(url, params=params)
    response.raise_for_status()
    sessions = response.json()
    session = next((s for s in sessions if s["session_name"].lower() == session_name.lower()), None)
    if not session:
        raise ValueError(f"No qualifying session found for {country} {year}")
    return session

# Fetch lap data for a session
def fetch_laps(session_key):
    laps_url = f"https://api.openf1.org/v1/laps?session_key={session_key}"
    response = requests.get(laps_url)
    response.raise_for_status()
    laps = response.json()
    if not laps:
        raise ValueError(f"No lap data found for session_key {session_key}")
    return laps

# Main prediction logic
def main():
    # Fetch session data
    qualifying_session = fetch_session_key(RACE_COUNTRY, RACE_YEAR, "Qualifying")
    session_key = qualifying_session["session_key"]
    meeting_key = qualifying_session["meeting_key"]

    # Fetch lap data
    laps = fetch_laps(session_key)

    # Calculate best lap times to determine grid positions
    driver_best_times = defaultdict(lambda: {"time": float("inf")})
    for lap in laps:
        driver_number = lap["driver_number"]
        if lap.get("is_pit_out_lap", False) or "lap_duration" not in lap or lap["lap_duration"] is None:
            continue
        lap_time = lap["lap_duration"]
        if lap_time < driver_best_times[driver_number]["time"]:
            driver_best_times[driver_number]["time"] = lap_time

    driver_times = [(driver_number, data["time"]) for driver_number, data in driver_best_times.items() if data["time"] != float("inf")]
    if not driver_times:
        raise ValueError("No valid lap times found in API data")
    driver_times_sorted = sorted(driver_times, key=lambda x: x[1])
    grid_positions = {driver_number: position + 1 for position, (driver_number, time) in enumerate(driver_times_sorted)}

    # Map driver numbers to driver references AND team names
    drivers_url = f"https://api.openf1.org/v1/drivers?meeting_key={meeting_key}"
    response = requests.get(drivers_url)
    response.raise_for_status()
    drivers = response.json()
    if not drivers:
        raise ValueError("No driver data found for this meeting")
    driver_number_to_ref_team = {}
    for driver in drivers:
        if "driver_number" in driver and "last_name" in driver and "team_name" in driver:
            driver_number = str(driver["driver_number"])
            # Normalize driver name: "O PIASTRI" -> "piastri"
            driver_ref = driver["last_name"].replace(" ", "").lower()
            # Additional mapping to match common dataset formats
            # driver_ref = driver_ref.replace("verstappen", "max_verstappen").replace("hamilton", "lewis_hamilton").replace("leclerc", "charles_leclerc")
            team_name = driver["team_name"]
            driver_number_to_ref_team[driver_number] = (driver_ref, team_name)

    # Create grid DataFrame with team names
    data = [(driver_number_to_ref_team.get(str(driver_number), (f"driver_{driver_number}", "Unknown"))[0],
             driver_number_to_ref_team.get(str(driver_number), (f"driver_{driver_number}", "Unknown"))[1],
             position)
            for driver_number, position in grid_positions.items()]
    grid_df = pd.DataFrame(data, columns=["driverRef", "team", "grid_position"])
    grid_df = grid_df.sort_values("grid_position")

    # Load historical data for past wins
    historical_data = pd.read_csv("dataset.csv")
    driver_past_wins = historical_data.groupby("driverRef")["is_winner"].sum().to_dict()

    # Prepare new race data for prediction
    new_race = pd.merge(grid_df, pd.Series(driver_past_wins, name="past_wins"), left_on="driverRef", right_index=True, how="left")
    new_race["past_wins"] = new_race["past_wins"].fillna(0)
    new_race = new_race.set_index("driverRef")
    new_race = new_race[["team", "grid_position", "past_wins"]]
    new_race = new_race.rename(columns={"grid_position": "qualifying_position"})

    # Heuristic for prediction
    grid_scores = np.exp(-0.3 * (new_race["qualifying_position"] - 1))  # Adjusted decay: P1 >> P2
    max_wins = max(new_race["past_wins"].max(), 1)
    win_scores = (new_race["past_wins"] / max_wins) ** 0.5  # Square root to balance

    # Manual form factor for 2023-2024 dominance
    form_boost = pd.Series(1.0, index=new_race.index)
    dominant_drivers = {
        "max_verstappen": 1.5,  # Verstappen dominated 2023-2024
        "lewis_hamilton": 1.1,  # Hamilton has been strong
        "charles_leclerc": 1.1, # Leclerc has been competitive
        "piastri": 1.2,         # Piastri has been on the rise
        "norris": 1.2,          # Norris has been consistent
    }
    for driver, boost in dominant_drivers.items():
        if driver in form_boost.index:
            form_boost[driver] = boost

    # Combine scores: 35% grid, 35% past wins, 15% form boost, 10% team boost, 5% consistency
    probs = (grid_scores * 0.35 + win_scores * 0.35) * form_boost
    # Add randomness to reflect race unpredictability
    randomness = np.random.uniform(0.95, 1.05, size=len(probs))  # Tighter randomness
    probs = probs * randomness
    # Normalize to sum to 1
    probs = probs / probs.sum()

    # Sort drivers by probability (highest to lowest)
    prob_df = pd.DataFrame({"driver": new_race.index, "team": new_race["team"], "probability": probs})
    prob_df = prob_df.sort_values(by="probability", ascending=False)

    # Add display names for drivers (e.g., "max_verstappen" -> "Max Verstappen")
    prob_df["driver_display"] = prob_df["driver"].apply(lambda x: x.replace("_", " ").title())

    # Calculate column widths dynamically
    driver_width = max(len(name) for name in prob_df["driver_display"]) + 2  # Add padding
    team_width = max(len(team) for team in prob_df["team"]) + 2
    probability_width = 15  # Fixed width for "Win Probability"

    # Create header and separator
    header = f"{'Driver':<{driver_width}}  {'Team':<{team_width}}  {'Win Probability':>{probability_width}}"
    separator = "-" * (driver_width + team_width + probability_width + 4)  # 4 for spaces between columns

    # Print the table
    print(f"\nWin probabilities for the {RACE_COUNTRY} Grand Prix {RACE_YEAR}:\n")
    print(header)
    print(separator)
    for _, row in prob_df.iterrows():
        driver_display = row["driver_display"]
        team = row["team"]
        percentage = row["probability"] * 150
        print(f"{driver_display:<{driver_width}}  {team:<{team_width}}  {percentage:>{probability_width}.0f}%")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")