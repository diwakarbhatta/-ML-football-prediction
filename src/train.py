import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.data_loader import load_and_preprocess_all
from src.features import FeatureEngineer, FEATURE_COLUMNS
from src.models import MatchOutcomeModel, ScorePredictor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, classification_report
from sklearn.dummy import DummyClassifier

def train_pipeline(save_artifacts: bool = True) -> Dict[str, Any]:
    print("=" * 65)
    print("      ENGLISH PREMIER LEAGUE ML PREDICTOR TRAINING PIPELINE")
    print("=" * 65)

    # 1. Load or prepare processed match data
    csv_path = PROCESSED_DATA_DIR / "epl_matches.csv"
    if not csv_path.exists():
        print("Processed data not found. Fetching raw match records...")
        df = load_and_preprocess_all()
    else:
        print(f"Loading cached match data from {csv_path}...")
        df = pd.read_csv(csv_path, parse_dates=["Date"])

    # 2. Extract features
    print("Extracting dynamic Elo ratings, rolling form, and H2H records...")
    fe = FeatureEngineer()
    feats_df, elo_tracker, latest_team_state = fe.build_features(df)

    # 3. Chronological Train/Test Split
    # Historical training: Seasons 1819 through 2223
    # Holdout evaluation: Seasons 2324 & 2425 (modern forward testing)
    train_mask = feats_df["Date"] < "2023-08-01"
    test_mask = feats_df["Date"] >= "2023-08-01"

    train_df = feats_df[train_mask].copy()
    test_df = feats_df[test_mask].copy()

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["FTR"].values
    y_train_hg = train_df["FTHG"].values
    y_train_ag = train_df["FTAG"].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["FTR"].values
    y_test_hg = test_df["FTHG"].values
    y_test_ag = test_df["FTAG"].values

    print(f"\nDataset Splitting:")
    print(f" - Training Matches: {len(train_df)} (Seasons 2018/19 - 2022/23)")
    print(f" - Test Matches:     {len(test_df)} (Seasons 2023/24 - 2024/25)")

    # 4. Train Outcome Classifier Ensemble
    print("\nTraining Calibrated Match Outcome Classifier Ensemble...")
    outcome_model = MatchOutcomeModel()
    outcome_model.fit(X_train, y_train)

    # Train Baseline Models for comparison
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_acc = accuracy_score(y_test, dummy.predict(X_test))

    train_preds = outcome_model.predict(X_train)
    test_preds = outcome_model.predict(X_test)
    test_probs = outcome_model.predict_proba(X_test)
    classes = outcome_model.classes_

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    test_loss = log_loss(y_test, test_probs, labels=classes)

    print("\n" + "-" * 45)
    print(" OUTCOME MODEL EVALUATION RESULTS (TEST SET):")
    print("-" * 45)
    print(f" > Benchmark Dummy Accuracy:    {dummy_acc * 100:.2f}%")
    print(f" > Train Accuracy:             {train_acc * 100:.2f}%")
    print(f" > Test Accuracy:              {test_acc * 100:.2f}%")
    print(f" > Test Multi-Class Log Loss:   {test_loss:.4f}")
    print("\nDetailed Test Classification Report:")
    print(classification_report(y_test, test_preds, labels=classes, zero_division=0))

    # 5. Train Score & Goal Expectation Model
    print("-" * 45)
    print("Training Goal Expectation & Score Predictor (Poisson)...")
    score_model = ScorePredictor()
    score_model.fit(X_train, y_train_hg, y_train_ag)

    exp_hg_test, exp_ag_test = score_model.predict_expected_goals(X_test)
    mae_hg = mean_absolute_error(y_test_hg, exp_hg_test)
    mae_ag = mean_absolute_error(y_test_ag, exp_ag_test)
    print(f" > Home Goals MAE: {mae_hg:.3f} goals")
    print(f" > Away Goals MAE: {mae_ag:.3f} goals")

    # 6. Betting Market Evaluation against Bookmaker Odds (if available in test set)
    test_prob_dicts = outcome_model.predict_proba_dict(X_test)
    test_indices = list(test_df.index)
    sim_profit = 0.0
    bets_placed = 0
    if "B365H" in df.columns:
        for i, idx in enumerate(test_indices):
            row = df.loc[idx]
            b365_h, b365_d, b365_a = row.get("B365H"), row.get("B365D"), row.get("B365A")
            if pd.notna(b365_h) and pd.notna(b365_d) and pd.notna(b365_a):
                p_dict = test_prob_dicts[i]
                odds = {"H": b365_h, "D": b365_d, "A": b365_a}
                
                # Check for value edge > 5% over implied bookmaker probability
                for outcome in ["H", "D", "A"]:
                    implied = 1.0 / odds[outcome]
                    if p_dict[outcome] > implied * 1.08 and p_dict[outcome] > 0.45:
                        bets_placed += 1
                        if row["FTR"] == outcome:
                            sim_profit += (odds[outcome] - 1.0)
                        else:
                            sim_profit -= 1.0

        roi = (sim_profit / bets_placed * 100) if bets_placed > 0 else 0.0
        print(f"\nValue Betting Simulation against Bet365 Odds (Conservative edge >8%, prob >45%):")
        print(f" > Bets placed with edge:      {bets_placed}")
        print(f" > Net units won/lost:         {sim_profit:+.2f}")
        print(f" > Simulated ROI:              {roi:+.2f}%")

    # 7. Save model artifacts
    artifacts = {
        "outcome_model": outcome_model,
        "score_model": score_model,
        "feature_columns": FEATURE_COLUMNS,
        "latest_team_state": latest_team_state,
        "metrics": {
            "dummy_acc": float(dummy_acc),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "test_log_loss": float(test_loss),
            "mae_hg": float(mae_hg),
            "mae_ag": float(mae_ag)
        }
    }

    if save_artifacts:
        artifact_path = MODELS_DIR / "model_artifacts.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(artifacts, f)
        print(f"\nModel artifacts successfully saved to {artifact_path}")

    return artifacts

if __name__ == "__main__":
    train_pipeline()
