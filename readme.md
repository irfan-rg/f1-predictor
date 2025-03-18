# F1 Race Winner Predictor 🏁

Welcome to the **F1 Race Winner Predictor**! This project harnesses machine learning to predict Formula 1 race Winners using Historical Data and Real-Time Qualifying results. 

It’s built for F1 enthusiasts and data science fans who want to explore predictive analytics in motorsport.

## Table of Contents
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

# Background and Motivation

Formula 1 racing produces a wealth of data—lap times, driver standings, and more—making it perfect for predictive modeling. Inspired by the Australian Grand Prix on March 16, 2025, and current trends in sports analytics, machine learning can forecast race outcomes effectively. As of 12:37 PM PDT on March 17, 2025, with the Australian GP just concluded, the next race is the Chinese Grand Prix on March 23, 2025. Leveraging real-time qualifying data from the OpenF1 API enhances prediction accuracy by incorporating live, up-to-date information.

## Data Collection: Sourcing Real-Time Qualifying Data

For real-time qualifying data, we need a dependable source updated during race weekends. The OpenF1 API—a free, open-source tool at [openf1.org](https://openf1.org)—offers real-time and historical F1 data, including session details, lap times, and driver info, ideal for fetching qualifying results.

With the 2025 F1 season featuring 24 races, starting with Australia on March 16, 2025, the Chinese Grand Prix follows on March 23, 2025. Qualifying for China, typically held on Saturday, is set for March 22, 2025. At 12:37 PM PDT on March 17, we’re pre-qualifying, but real-time data from the OpenF1 API will be available post-session for analysis.

## Features
- **Historical Analysis**: Leverages past F1 race data to train the prediction model.
- **Real-Time Insights**: Integrates live qualifying data for up-to-date predictions.
- **User-Friendly**: Simple setup and clear instructions for all users.

## Setup
Follow these steps to get started:
1. **Download Data**: Grab historical F1 data from [Ergast](http://ergast.com/mrd/db) and place it in the `data/` folder.
2. **Install Dependencies**: Run this in your terminal:
   ```bash
   pip install pandas scikit-learn requests
3. **Preprocess Data**: Prepare the dataset with:
   ```bash
   python preprocess.py
3. **Train the Model**: Build the model by running:
   ```bash
   python train_model.py
## Usage
Here’s how to use it:
- **Historical Predictions**: Test it on old races with:
  ```bash
  python predict.py
- **Real-Time Predictions:**: After qualifying, update real_time_predict.py with race details (e.g., "China" and "2025") and run:
  ```bash
  python real_time_predict.py
## Contributing
Wanna help out? Awesome! Just:
- Submit a pull request with your changes.
- Open an issue for bugs or ideas.

## License
This project is licensed under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for details.