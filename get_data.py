import pandas as pd
import os

data_dir = "data"
required_files = ["races.csv", "results.csv", "drivers.csv", "qualifying.csv"]

for file in required_files:
    if not os.path.exists(os.path.join(data_dir, file)):
        raise FileNotFoundError(f"Missing required file: {file}")
    print(f"Loaded {file}:")
    df = pd.read_csv(os.path.join(data_dir, file))
    print(df.head())