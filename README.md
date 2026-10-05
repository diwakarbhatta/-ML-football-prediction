# ⚽ Premier League Match Outcome & Score Predictor (ML)

An end-to-end Machine Learning project to forecast **English Premier League** match outcomes before kickoff. Predicts the **3-way outcome (Home Win / Draw / Away Win)**, calibrated win/draw/loss probabilities, **Expected Goals (xG)**, and exact scorelines using **Dixon-Coles adjusted Poisson models**.

Includes a command-line interface, automated historical data pipeline, and an interactive modern web dashboard with live score heatmaps.

---

## 🌟 Key Features

1. **3-Way Match Outcome Prediction**:
   - Probabilities for **Home Win**, **Draw**, and **Away Win** calibrated using sigmoid probability calibration.
   - Outperforms benchmark naive frequency models (~55.5% test accuracy vs 43.4% dummy baseline).

2. **Goal Expectation & Scoreline Forecasting**:
   - Predicts Expected Goals (xG) for both home and away sides.
   - Evaluates a full $6 \times 6$ bivariate Poisson probability matrix with **Dixon-Coles** adjustment to accurately model low-scoring ties (0-0, 1-1, 1-0, 0-1).
   - Generates the **most probable exact scoreline** (e.g. `2 - 1`) and the Top 5 most likely scorelines with exact percentages.
   - Probability gauges for **Over / Under 2.5 Goals** and **Both Teams to Score (BTTS)**.

3. **Domain-Specific Feature Engineering (Zero Lookahead Leakage)**:
   - **Dynamic Elo Rating Engine**: Continually updated match-by-match with home-turf advantage (+65 Elo) and margin-of-victory scaling.
   - **Rolling Form**: Exponential & rolling 5-game / 3-game points average, goals scored/conceded, goal difference, and shots on target.
   - **Venue-Specific Splits**: Home team's performance at home vs Away team's away form.
   - **Head-to-Head (H2H)**: Historical results between the two specific clubs over past encounters.
   - **Rest & Congestion**: Days elapsed since each team's last fixture.

4. **Interactive Web Dashboard & REST API**:
   - Responsive web dashboard styled with the official Premier League visual identity.
   - Quick derby clash buttons (North London Derby, Manchester Derby, Merseyside Derby, London Derby).
   - Interactive Poisson score probability heatmap grid.
   - REST API endpoints for seamless integration.

---

## 📁 Project Structure

```text
├── app.py                      # Flask Web Dashboard and REST API
├── requirements.txt            # Python dependencies
├── src/
│   ├── config.py               # Paths, clubs, and Elo constants
│   ├── data_loader.py          # Multi-season downloader & data cleaner
│   ├── features.py             # Elo tracker, rolling form, and H2H engineering
│   ├── models.py               # Calibrated Classifier Ensemble & Dixon-Coles Poisson Regressor
│   ├── train.py                # Chronological training & validation pipeline
│   └── predict.py              # CLI and programmatic inference interface
├── templates/
│   └── index.html              # Responsive interactive dashboard UI
├── tests/
│   └── test_pipeline.py        # Automated test suite (Elo, Poisson, API tests)
└── data/
    ├── raw/                    # Cached raw season CSVs from football-data.co.uk
    └── processed/              # Cleaned chronological dataset
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/ml-football-predictor.git
cd "ml-football-predictor"
python -m pip install -r requirements.txt
```

### 2. Train the Models

Train the outcome classifier and score predictor across 7 historical Premier League seasons (2018/19 through 2024/25):

```bash
python -m src.train
```

*Output summary:*
- Downloads and cleans over 2,660 Premier League matches.
- Evaluates on holdout test seasons using strict chronological splitting (no time-travel leakage).
- Saves model artifacts to `models/model_artifacts.pkl`.

### 3. Predict via Command Line (CLI)

Predict any upcoming fixture between Premier League clubs:

```bash
python -m src.predict --home "Arsenal" --away "Chelsea"
```

**Example Terminal Output:**
```text
============================================================
      PREMIER LEAGUE MATCH PREDICTION
      Arsenal (H)  vs  Chelsea (A)
============================================================
  Outcome Prediction:     >>> Arsenal Win <<<
  Predicted Exact Score:  >>> Arsenal 1 - 1 Chelsea (11.4%) <<<
  Expected Goals (xG):    Arsenal: 1.96  |  Chelsea: 1.04
------------------------------------------------------------
  OUTCOME PROBABILITIES:
   Arsenal Win:  56.3% | ##################
   Draw:         20.5% | ######
   Chelsea Win:  23.2% | #######
------------------------------------------------------------
  TOP 5 MOST LIKELY SCORELINES:
   Arsenal 1 - 1 Chelsea         : 11.4%
   Arsenal 2 - 1 Chelsea         : 10.1%
   Arsenal 2 - 0 Chelsea         : 9.8%
   Arsenal 1 - 0 Chelsea         : 8.9%
   Arsenal 3 - 1 Chelsea         : 6.6%
------------------------------------------------------------
  Over 2.5 Goals:  56.9%  |  Under 2.5 Goals: 43.1%
  Both Teams to Score (BTTS): Yes 56.3%  |  No 43.7%
  Current Elo:     Arsenal: 1784.3  |  Chelsea: 1685.4
============================================================
```

Try another matchup:
```bash
python -m src.predict --home "Liverpool" --away "Manchester City"
```

---

## 🌐 Launch Web Dashboard

Run the interactive Flask web application:

```bash
python app.py
```

Then open your browser and visit: **`http://127.0.0.1:5000`**

### Features in the Web App:
- Select any two Premier League clubs with instant swap.
- Live probability progress bars (Home Win / Draw / Away Win).
- Projected scoreline badge and xG breakdown.
- Interactive Poisson score heatmap grid showing exact probabilities from 0-0 to 4-4.
- Model performance modal displaying test metrics.

---

## 🔌 REST API Endpoints

### 1. Match Prediction Endpoint
- **URL**: `POST /api/predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "home_team": "Liverpool",
    "away_team": "Manchester City"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "prediction": {
      "home_team": "Liverpool",
      "away_team": "Manchester City",
      "predicted_outcome": "Liverpool Win",
      "win_probabilities": {
        "home_win": 42.9,
        "draw": 20.9,
        "away_win": 36.2
      },
      "expected_goals": {
        "home": 1.79,
        "away": 1.21
      },
      "predicted_score": "1 - 1",
      "predicted_score_prob": 12.0,
      "top_scorelines": [
        {"home_goals": 1, "away_goals": 1, "probability": 0.120},
        {"home_goals": 2, "away_goals": 1, "probability": 0.098}
      ],
      "over_under_25": { "over": 57.1, "under": 42.9 },
      "btts": { "yes": 59.4, "no": 40.6 }
    }
  }
  ```

### 2. Available Teams Endpoint
- **URL**: `GET /api/teams`
- Returns a list of current active Premier League clubs.

### 3. Model Performance Endpoint
- **URL**: `GET /api/metrics`
- Returns test accuracy, log loss, and goals MAE.

---

## 🧪 Running Automated Tests

Run the test suite:

```bash
python -m unittest tests/test_pipeline.py
```

All 4 test suites verify:
1. Dynamic Elo tracker updates and home bonus mechanics.
2. Poisson distribution and Dixon-Coles correlation adjustment factors.
3. Prediction output contract and probability normalization.
4. Flask web routes and JSON API responses.

---

## 📊 Model Evaluation Summary

| Metric | Model Performance | Benchmark / Baseline |
| :--- | :--- | :--- |
| **Test Accuracy (3-Way)** | **55.53%** | 43.42% (Majority Home Win Baseline) |
| **Multi-Class Log Loss** | **0.9689** | 1.0986 (Uniform Random Guessing) |
| **Home Goals MAE** | **1.000 goals** | Dixon-Coles Poisson Regression |
| **Away Goals MAE** | **0.889 goals** | Dixon-Coles Poisson Regression |

---

## 📜 License
MIT License. Created for football analytics and sports modeling.
