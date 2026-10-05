import os
import sys
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR, CURRENT_EPL_TEAMS, HOME_ADVANTAGE_ELO, INITIAL_ELO
from src.data_loader import standardize_team_name
from src.features import FEATURE_COLUMNS

_ARTIFACTS = None

def get_artifacts() -> Dict[str, Any]:
    """Lazy load serialized model artifacts."""
    global _ARTIFACTS
    if _ARTIFACTS is None:
        model_path = MODELS_DIR / "model_artifacts.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found at {model_path}. Please run 'python -m src.train' first."
            )
        with open(model_path, "rb") as f:
            _ARTIFACTS = pickle.load(f)
    return _ARTIFACTS

def compute_team_stats(history: list, n: int = 5) -> Dict[str, float]:
    """Compute recent rolling statistics for a team."""
    if not history:
        return {
            "points_avg": 1.3,
            "goals_scored_avg": 1.3,
            "goals_conceded_avg": 1.3,
            "goal_diff_avg": 0.0,
            "win_rate": 0.33,
            "loss_rate": 0.33,
            "sot_avg": 4.5,
            "sot_conceded_avg": 4.5,
        }
    recent = history[-n:]
    pts = [m["points"] for m in recent]
    gs = [m["goals_scored"] for m in recent]
    gc = [m["goals_conceded"] for m in recent]
    sot = [m.get("sot", 4.5) for m in recent]
    wins = sum(1 for p in pts if p == 3)
    losses = sum(1 for p in pts if p == 0)
    count = len(recent)
    return {
        "points_avg": sum(pts) / count,
        "goals_scored_avg": sum(gs) / count,
        "goals_conceded_avg": sum(gc) / count,
        "goal_diff_avg": (sum(gs) - sum(gc)) / count,
        "win_rate": wins / count,
        "loss_rate": losses / count,
        "sot_avg": sum(sot) / count,
    }

def construct_match_features(home_team: str, away_team: str, team_state: Dict[str, Any]) -> Dict[str, float]:
    """Assemble all 34 pre-match features for the given fixture."""
    ratings = team_state.get("ratings", {})
    team_history = team_state.get("team_history", {})
    team_home_history = team_state.get("team_home_history", {})
    team_away_history = team_state.get("team_away_history", {})
    h2h_history = team_state.get("h2h_history", {})

    home_elo = ratings.get(home_team, INITIAL_ELO)
    away_elo = ratings.get(away_team, INITIAL_ELO)
    elo_diff = (home_elo + HOME_ADVANTAGE_ELO) - away_elo

    home_hist = team_history.get(home_team, [])
    away_hist = team_history.get(away_team, [])

    home_form_5 = compute_team_stats(home_hist, 5)
    away_form_5 = compute_team_stats(away_hist, 5)
    home_form_3 = compute_team_stats(home_hist, 3)
    away_form_3 = compute_team_stats(away_hist, 3)

    home_venue_form = compute_team_stats(team_home_history.get(home_team, []), 5)
    away_venue_form = compute_team_stats(team_away_history.get(away_team, []), 5)

    h2h_key = tuple(sorted([home_team, away_team]))
    h2h_matches = h2h_history.get(h2h_key, [])
    if h2h_matches:
        recent_h2h = h2h_matches[-4:]
        h2h_home_wins = sum(1 for m in recent_h2h if m["winner"] == home_team)
        h2h_away_wins = sum(1 for m in recent_h2h if m["winner"] == away_team)
        h2h_n = len(recent_h2h)
        h2h_home_win_ratio = h2h_home_wins / h2h_n
        h2h_away_win_ratio = h2h_away_wins / h2h_n
    else:
        h2h_home_win_ratio = 0.33
        h2h_away_win_ratio = 0.33

    home_rest = 7
    away_rest = 7
    rest_diff = 0

    expected_home_pressure = home_form_5["goals_scored_avg"] - away_form_5["goals_conceded_avg"]
    expected_away_pressure = away_form_5["goals_scored_avg"] - home_form_5["goals_conceded_avg"]

    feature_dict = {
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": elo_diff,
        "home_pts_avg_5": home_form_5["points_avg"],
        "away_pts_avg_5": away_form_5["points_avg"],
        "pts_diff_5": home_form_5["points_avg"] - away_form_5["points_avg"],
        "home_gs_avg_5": home_form_5["goals_scored_avg"],
        "away_gs_avg_5": away_form_5["goals_scored_avg"],
        "home_gc_avg_5": home_form_5["goals_conceded_avg"],
        "away_gc_avg_5": away_form_5["goals_conceded_avg"],
        "home_gd_avg_5": home_form_5["goal_diff_avg"],
        "away_gd_avg_5": away_form_5["goal_diff_avg"],
        "home_win_rate_5": home_form_5["win_rate"],
        "away_win_rate_5": away_form_5["win_rate"],
        "home_loss_rate_5": home_form_5["loss_rate"],
        "away_loss_rate_5": away_form_5["loss_rate"],
        "home_sot_avg_5": home_form_5["sot_avg"],
        "away_sot_avg_5": away_form_5["sot_avg"],
        "home_pts_avg_3": home_form_3["points_avg"],
        "away_pts_avg_3": away_form_3["points_avg"],
        "pts_diff_3": home_form_3["points_avg"] - away_form_3["points_avg"],
        "home_gd_avg_3": home_form_3["goal_diff_avg"],
        "away_gd_avg_3": away_form_3["goal_diff_avg"],
        "home_team_home_win_rate": home_venue_form["win_rate"],
        "away_team_away_win_rate": away_venue_form["win_rate"],
        "home_team_home_gd": home_venue_form["goal_diff_avg"],
        "away_team_away_gd": away_venue_form["goal_diff_avg"],
        "h2h_home_win_ratio": h2h_home_win_ratio,
        "h2h_away_win_ratio": h2h_away_win_ratio,
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "rest_diff": rest_diff,
        "expected_home_pressure": expected_home_pressure,
        "expected_away_pressure": expected_away_pressure,
    }
    return feature_dict

def predict_match(home_team: str, away_team: str) -> Dict[str, Any]:
    """
    Generate complete match outcome, win probabilities, expected goals,
    and exact score predictions for a Premier League fixture.
    """
    artifacts = get_artifacts()
    outcome_model = artifacts["outcome_model"]
    score_model = artifacts["score_model"]
    team_state = artifacts["latest_team_state"]

    home_norm = standardize_team_name(home_team)
    away_norm = standardize_team_name(away_team)

    features = construct_match_features(home_norm, away_norm, team_state)
    feature_vector = [[features[col] for col in FEATURE_COLUMNS]]

    # 1. Calibrated Outcome Probabilities
    prob_dict = outcome_model.predict_proba_dict(feature_vector)[0]
    p_home = round(prob_dict.get("H", 0.0) * 100, 1)
    p_draw = round(prob_dict.get("D", 0.0) * 100, 1)
    p_away = round(prob_dict.get("A", 0.0) * 100, 1)

    # 2. Predicted Expected Goals (xG) & Exact Score Distributions
    exp_h, exp_a = score_model.predict_expected_goals(feature_vector)
    lambda_h, lambda_a = float(exp_h[0]), float(exp_a[0])
    score_analysis = score_model.predict_score_distribution(lambda_h, lambda_a)

    # 3. Determine primary predicted outcome
    if p_home >= p_away and p_home >= p_draw:
        primary_outcome = f"{home_norm} Win"
    elif p_away >= p_home and p_away >= p_draw:
        primary_outcome = f"{away_norm} Win"
    else:
        primary_outcome = "Draw"

    most_likely_score = score_analysis["most_likely_score"]

    return {
        "home_team": home_norm,
        "away_team": away_norm,
        "predicted_outcome": primary_outcome,
        "win_probabilities": {
            "home_win": p_home,
            "draw": p_draw,
            "away_win": p_away
        },
        "expected_goals": {
            "home": round(lambda_h, 2),
            "away": round(lambda_a, 2)
        },
        "predicted_score": f"{most_likely_score[0]} - {most_likely_score[1]}",
        "predicted_score_prob": score_analysis["most_likely_prob"],
        "top_scorelines": score_analysis["top_scores"],
        "over_under_25": score_analysis["over_under_25"],
        "btts": score_analysis["btts"],
        "team_metrics": {
            "home_elo": round(features["home_elo"], 1),
            "away_elo": round(features["away_elo"], 1),
            "elo_diff": round(features["elo_diff"], 1),
            "home_form_pts_5": round(features["home_pts_avg_5"] * 5, 1),
            "away_form_pts_5": round(features["away_pts_avg_5"] * 5, 1),
        },
        "score_matrix": score_analysis["score_matrix"]
    }

def print_prediction_cli(res: Dict[str, Any]):
    """Pretty print prediction results in terminal."""
    h = res["home_team"]
    a = res["away_team"]
    probs = res["win_probabilities"]
    xg = res["expected_goals"]

    print("\n" + "=" * 60)
    print(f"      PREMIER LEAGUE MATCH PREDICTION")
    print(f"      {h} (H)  vs  {a} (A)")
    print("=" * 60)
    print(f"  Outcome Prediction:     >>> {res['predicted_outcome']} <<<")
    print(f"  Predicted Exact Score:  >>> {h} {res['predicted_score']} {a} ({res['predicted_score_prob']}%) <<<")
    print(f"  Expected Goals (xG):    {h}: {xg['home']}  |  {a}: {xg['away']}")
    print("-" * 60)
    print("  OUTCOME PROBABILITIES:")
    bar_h = "#" * int(probs["home_win"] // 3)
    bar_d = "#" * int(probs["draw"] // 3)
    bar_a = "#" * int(probs["away_win"] // 3)
    print(f"   {h} Win: {probs['home_win']:>5.1f}% | {bar_h}")
    print(f"   Draw:            {probs['draw']:>5.1f}% | {bar_d}")
    print(f"   {a} Win: {probs['away_win']:>5.1f}% | {bar_a}")
    print("-" * 60)
    print("  TOP 5 MOST LIKELY SCORELINES:")
    for s in res["top_scorelines"][:5]:
        score_str = f"{s['home_goals']} - {s['away_goals']}"
        print(f"   {h} {score_str} {a:<15} : {s['probability'] * 100:.1f}%")
    print("-" * 60)
    print(f"  Over 2.5 Goals:  {res['over_under_25']['over']}%  |  Under 2.5 Goals: {res['over_under_25']['under']}%")
    print(f"  Both Teams to Score (BTTS): Yes {res['btts']['yes']}%  |  No {res['btts']['no']}%")
    print(f"  Current Elo:     {h}: {res['team_metrics']['home_elo']}  |  {a}: {res['team_metrics']['away_elo']}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Premier League Match Outcome and Scoreline")
    parser.add_argument("--home", type=str, default="Arsenal", help="Home Team Name (e.g. 'Arsenal')")
    parser.add_argument("--away", type=str, default="Chelsea", help="Away Team Name (e.g. 'Chelsea')")
    args = parser.parse_args()

    try:
        prediction = predict_match(args.home, args.away)
        print_prediction_cli(prediction)
    except Exception as e:
        print(f"Error predicting match: {e}")
