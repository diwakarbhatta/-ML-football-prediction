import os
import sys
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR, CURRENT_EPL_TEAMS, HOME_ADVANTAGE_ELO, INITIAL_ELO
from src.data_loader import standardize_team_name
from src.features import FEATURE_COLUMNS

_ARTIFACTS = None

def get_artifacts() -> Dict[str, Any]:
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

def prob_to_decimal_odds(p: float) -> float:
    """Convert probability (0-100) to decimal odds."""
    if p <= 0.01:
        return 99.0
    return round(100.0 / p, 2)

def prob_to_american_odds(p: float) -> str:
    """Convert probability (0-100) to American moneyline format."""
    if p >= 50.0:
        val = int(round(-100.0 * (p / (100.0 - p)))) if p < 99.9 else -9999
        return f"{val}"
    else:
        val = int(round(100.0 * ((100.0 - p) / max(p, 0.01))))
        return f"+{val}"

def extract_form_details(history: List[Dict], n: int = 5) -> Dict[str, Any]:
    """Extract granular form data including individual match scores and aggregates."""
    if not history:
        return {
            "form_string": "D-D-D-D-D",
            "points": 5,
            "goals_scored": 5,
            "goals_conceded": 5,
            "goal_diff": 0,
            "win_rate": 0.33,
            "matches": []
        }
    recent = history[-n:]
    matches = []
    pts_total = 0
    gs_total = 0
    gc_total = 0
    form_chars = []

    for m in recent:
        pts = m.get("points", 0)
        res = m.get("result", "W" if pts == 3 else ("D" if pts == 1 else "L"))
        form_chars.append(res)
        pts_total += pts
        gs_total += m.get("goals_scored", 0)
        gc_total += m.get("goals_conceded", 0)
        matches.append({
            "date": m.get("date_str", str(m.get("date", ""))[:10]),
            "opponent": m.get("opponent", "Opponent"),
            "venue": m.get("venue", "Home" if m.get("is_home") else "Away"),
            "result": res,
            "score": m.get("score", f"{m.get('goals_scored')}-{m.get('goals_conceded')}"),
            "points": pts
        })

    return {
        "form_string": "-".join(form_chars),
        "points": pts_total,
        "goals_scored": gs_total,
        "goals_conceded": gc_total,
        "goal_diff": gs_total - gc_total,
        "win_rate": round(sum(1 for c in form_chars if c == "W") / len(form_chars) * 100, 1),
        "matches": matches[::-1] # Most recent first
    }

def construct_match_features(home_team: str, away_team: str, team_state: Dict[str, Any]) -> Dict[str, float]:
    ratings = team_state.get("ratings", {})
    team_history = team_state.get("team_history", {})
    team_home_history = team_state.get("team_home_history", {})
    team_away_history = team_state.get("team_away_history", {})
    h2h_history = team_state.get("h2h_history", {})
    standings_table = team_state.get("standings_table", {})
    rankings = team_state.get("rankings", {})

    home_elo = ratings.get(home_team, INITIAL_ELO)
    away_elo = ratings.get(away_team, INITIAL_ELO)
    elo_diff = (home_elo + HOME_ADVANTAGE_ELO) - away_elo

    # League position & Goal Difference
    home_pos = rankings.get(home_team, 10)
    away_pos = rankings.get(away_team, 10)
    pos_diff = away_pos - home_pos
    home_table = standings_table.get(home_team, {"gd": 0, "pts": 0, "p": 0})
    away_table = standings_table.get(away_team, {"gd": 0, "pts": 0, "p": 0})
    home_table_gd = home_table.get("gd", 0)
    away_table_gd = away_table.get("gd", 0)
    table_gd_diff = home_table_gd - away_table_gd

    home_hist = team_history.get(home_team, [])
    away_hist = team_history.get(away_team, [])

    home_form_5 = extract_form_details(home_hist, 5)
    away_form_5 = extract_form_details(away_hist, 5)
    home_form_3 = extract_form_details(home_hist, 3)
    away_form_3 = extract_form_details(away_hist, 3)

    home_venue_form = extract_form_details(team_home_history.get(home_team, []), 5)
    away_venue_form = extract_form_details(team_away_history.get(away_team, []), 5)

    h2h_key = tuple(sorted([home_team, away_team]))
    h2h_matches = h2h_history.get(h2h_key, [])
    if h2h_matches:
        recent_h2h = h2h_matches[-5:]
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

    expected_home_pressure = (home_form_5["goals_scored"] / 5.0) - (away_form_5["goals_conceded"] / 5.0)
    expected_away_pressure = (away_form_5["goals_scored"] / 5.0) - (home_form_5["goals_conceded"] / 5.0)

    feature_dict = {
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": elo_diff,
        "home_league_pos": home_pos,
        "away_league_pos": away_pos,
        "pos_diff": pos_diff,
        "home_table_gd": home_table_gd,
        "away_table_gd": away_table_gd,
        "table_gd_diff": table_gd_diff,
        "home_pts_sum_5": home_form_5["points"],
        "away_pts_sum_5": away_form_5["points"],
        "home_pts_avg_5": home_form_5["points"] / 5.0,
        "away_pts_avg_5": away_form_5["points"] / 5.0,
        "pts_diff_5": (home_form_5["points"] - away_form_5["points"]) / 5.0,
        "home_gs_avg_5": home_form_5["goals_scored"] / 5.0,
        "away_gs_avg_5": away_form_5["goals_scored"] / 5.0,
        "home_gc_avg_5": home_form_5["goals_conceded"] / 5.0,
        "away_gc_avg_5": away_form_5["goals_conceded"] / 5.0,
        "home_gd_avg_5": home_form_5["goal_diff"] / 5.0,
        "away_gd_avg_5": away_form_5["goal_diff"] / 5.0,
        "home_win_rate_5": home_form_5["win_rate"] / 100.0,
        "away_win_rate_5": away_form_5["win_rate"] / 100.0,
        "home_loss_rate_5": 0.33,
        "away_loss_rate_5": 0.33,
        "home_sot_avg_5": 4.5,
        "away_sot_avg_5": 4.5,
        "home_pts_avg_3": home_form_3["points"] / 3.0,
        "away_pts_avg_3": away_form_3["points"] / 3.0,
        "pts_diff_3": (home_form_3["points"] - away_form_3["points"]) / 3.0,
        "home_gd_avg_3": home_form_3["goal_diff"] / 3.0,
        "away_gd_avg_3": away_form_3["goal_diff"] / 3.0,
        "home_team_home_win_rate": home_venue_form["win_rate"] / 100.0,
        "away_team_away_win_rate": away_venue_form["win_rate"] / 100.0,
        "home_team_home_gd": home_venue_form["goal_diff"] / 5.0,
        "away_team_away_gd": away_venue_form["goal_diff"] / 5.0,
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

    # 2. Expected Goals & Exact Scores
    exp_h, exp_a = score_model.predict_expected_goals(feature_vector)
    lambda_h, lambda_a = float(exp_h[0]), float(exp_a[0])
    score_analysis = score_model.predict_score_distribution(lambda_h, lambda_a)

    # 3. Primary Outcome Determination
    if p_home >= p_away and p_home >= p_draw:
        primary_outcome = f"{home_norm} Win"
    elif p_away >= p_home and p_away >= p_draw:
        primary_outcome = f"{away_norm} Win"
    else:
        primary_outcome = "Draw"

    most_likely_score = score_analysis["most_likely_score"]

    # 4. Form (Last 5 matches: points, goals scored/conceded)
    team_history = team_state.get("team_history", {})
    home_form_5 = extract_form_details(team_history.get(home_norm, []), 5)
    away_form_5 = extract_form_details(team_history.get(away_norm, []), 5)

    # 5. Home vs Away Records
    home_records = team_state.get("home_records", {})
    away_records = team_state.get("away_records", {})
    h_home_rec = home_records.get(home_norm, {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "gd": 0, "pts": 0})
    a_away_rec = away_records.get(away_norm, {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "gd": 0, "pts": 0})

    # 6. Head-to-Head History
    h2h_key = tuple(sorted([home_norm, away_norm]))
    h2h_matches = team_state.get("h2h_history", {}).get(h2h_key, [])
    recent_h2h_list = []
    h2h_home_w = 0
    h2h_away_w = 0
    h2h_d = 0
    for m in h2h_matches[-5:]:
        winner = m["winner"]
        if winner == home_norm:
            h2h_home_w += 1
        elif winner == away_norm:
            h2h_away_w += 1
        else:
            h2h_d += 1
        recent_h2h_list.append({
            "date": m.get("date_str", str(m.get("date", ""))[:10]),
            "fixture": f"{m['home_team']} {m['score']} {m['away_team']}",
            "winner": winner
        })
    recent_h2h_list.reverse()

    # 7. League Position & Goal Difference
    rankings = team_state.get("rankings", {})
    standings = team_state.get("standings_table", {})
    home_table = standings.get(home_norm, {"p": 0, "pts": 0, "gd": 0, "gf": 0, "ga": 0})
    away_table = standings.get(away_norm, {"p": 0, "pts": 0, "gd": 0, "gf": 0, "ga": 0})

    # 8. Bookmaker Odds & Market Comparison (Optional requirement)
    odds_data = {
        "decimal": {
            "home": prob_to_decimal_odds(p_home),
            "draw": prob_to_decimal_odds(p_draw),
            "away": prob_to_decimal_odds(p_away)
        },
        "american": {
            "home": prob_to_american_odds(p_home),
            "draw": prob_to_american_odds(p_draw),
            "away": prob_to_american_odds(p_away)
        }
    }

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
        # 1. Last 5 matches form
        "form_last_5": {
            "home": {
                "team": home_norm,
                "form_string": home_form_5["form_string"],
                "points": home_form_5["points"],
                "goals_scored": home_form_5["goals_scored"],
                "goals_conceded": home_form_5["goals_conceded"],
                "goal_diff": home_form_5["goal_diff"],
                "matches": home_form_5["matches"]
            },
            "away": {
                "team": away_norm,
                "form_string": away_form_5["form_string"],
                "points": away_form_5["points"],
                "goals_scored": away_form_5["goals_scored"],
                "goals_conceded": away_form_5["goals_conceded"],
                "goal_diff": away_form_5["goal_diff"],
                "matches": away_form_5["matches"]
            }
        },
        # 2. Home vs away record
        "venue_records": {
            "home_team_at_home": {
                "team": home_norm,
                "played": h_home_rec.get("p", 0),
                "won": h_home_rec.get("w", 0),
                "drawn": h_home_rec.get("d", 0),
                "lost": h_home_rec.get("l", 0),
                "gf": h_home_rec.get("gf", 0),
                "ga": h_home_rec.get("ga", 0),
                "gd": h_home_rec.get("gd", 0),
                "pts": h_home_rec.get("pts", 0)
            },
            "away_team_away": {
                "team": away_norm,
                "played": a_away_rec.get("p", 0),
                "won": a_away_rec.get("w", 0),
                "drawn": a_away_rec.get("d", 0),
                "lost": a_away_rec.get("l", 0),
                "gf": a_away_rec.get("gf", 0),
                "ga": a_away_rec.get("ga", 0),
                "gd": a_away_rec.get("gd", 0),
                "pts": a_away_rec.get("pts", 0)
            }
        },
        # 3. Head-to-head history
        "h2h": {
            "total_matches": len(h2h_matches),
            "home_wins": h2h_home_w,
            "draws": h2h_d,
            "away_wins": h2h_away_w,
            "recent_meetings": recent_h2h_list
        },
        # 4. League position and goal difference
        "league_standings": {
            "home": {
                "team": home_norm,
                "position": rankings.get(home_norm, 10),
                "played": home_table.get("p", 0),
                "points": home_table.get("pts", 0),
                "goal_diff": home_table.get("gd", 0),
                "goals_scored": home_table.get("gf", 0),
                "goals_conceded": home_table.get("ga", 0)
            },
            "away": {
                "team": away_norm,
                "position": rankings.get(away_norm, 10),
                "played": away_table.get("p", 0),
                "points": away_table.get("pts", 0),
                "goal_diff": away_table.get("gd", 0),
                "goals_scored": away_table.get("gf", 0),
                "goals_conceded": away_table.get("ga", 0)
            }
        },
        # 5. Bookmaker Odds
        "bookmaker_odds": odds_data,
        "team_metrics": {
            "home_elo": round(features["home_elo"], 1),
            "away_elo": round(features["away_elo"], 1),
            "elo_diff": round(features["elo_diff"], 1)
        },
        "score_matrix": score_analysis["score_matrix"]
    }

def print_prediction_cli(res: Dict[str, Any]):
    h = res["home_team"]
    a = res["away_team"]
    probs = res["win_probabilities"]
    xg = res["expected_goals"]
    f_h = res["form_last_5"]["home"]
    f_a = res["form_last_5"]["away"]
    rec_h = res["venue_records"]["home_team_at_home"]
    rec_a = res["venue_records"]["away_team_away"]
    h2h = res["h2h"]
    st_h = res["league_standings"]["home"]
    st_a = res["league_standings"]["away"]
    odds = res["bookmaker_odds"]["decimal"]
    american = res["bookmaker_odds"]["american"]

    print("\n" + "=" * 68)
    print(f"       PREMIER LEAGUE MATCH PREDICTION & ANALYTICS")
    print(f"       {h} (H)  vs  {a} (A)")
    print("=" * 68)
    print(f"  Outcome Prediction:     >>> {res['predicted_outcome']} <<<")
    print(f"  Predicted Exact Score:  >>> {h} {res['predicted_score']} {a} ({res['predicted_score_prob']}%) <<<")
    print(f"  Expected Goals (xG):    {h}: {xg['home']}  |  {a}: {xg['away']}")
    print("-" * 68)
    print("  OUTCOME PROBABILITIES & BOOKMAKER FAIR ODDS:")
    print(f"   {h} Win: {probs['home_win']:>5.1f}% | Decimal: {odds['home']:<5} | Moneyline: {american['home']}")
    print(f"   Draw:            {probs['draw']:>5.1f}% | Decimal: {odds['draw']:<5} | Moneyline: {american['draw']}")
    print(f"   {a} Win: {probs['away_win']:>5.1f}% | Decimal: {odds['away']:<5} | Moneyline: {american['away']}")
    print("-" * 68)
    print("  1. LAST 5 MATCHES FORM (Points, Goals Scored & Conceded):")
    print(f"   {h}: Form [{f_h['form_string']}] | Pts: {f_h['points']}/15 | Goals: {f_h['goals_scored']} scored, {f_h['goals_conceded']} conceded (GD: {f_h['goal_diff']:+d})")
    print(f"   {a}: Form [{f_a['form_string']}] | Pts: {f_a['points']}/15 | Goals: {f_a['goals_scored']} scored, {f_a['goals_conceded']} conceded (GD: {f_a['goal_diff']:+d})")
    print("-" * 68)
    print("  2. HOME VS AWAY VENUE RECORD:")
    print(f"   {h} at Home:  P:{rec_h['played']}  W:{rec_h['won']}  D:{rec_h['drawn']}  L:{rec_h['lost']} | GF:{rec_h['gf']} GA:{rec_h['ga']} GD:{rec_h['gd']:+d} | Pts:{rec_h['pts']}")
    print(f"   {a} Away:     P:{rec_a['played']}  W:{rec_a['won']}  D:{rec_a['drawn']}  L:{rec_a['lost']} | GF:{rec_a['gf']} GA:{rec_a['ga']} GD:{rec_a['gd']:+d} | Pts:{rec_a['pts']}")
    print("-" * 68)
    print("  3. HEAD-TO-HEAD (H2H) HISTORY:")
    print(f"   Last Meetings: {h} {h2h['home_wins']} wins | Draws: {h2h['draws']} | {a} {h2h['away_wins']} wins")
    for m in h2h["recent_meetings"][:3]:
        print(f"    - {m['date']}: {m['fixture']} (Winner: {m['winner']})")
    print("-" * 68)
    print("  4. LEAGUE POSITION & GOAL DIFFERENCE:")
    print(f"   {h}: #{st_h['position']} | Pts: {st_h['points']} | GD: {st_h['goal_diff']:+d} (GF: {st_h['goals_scored']}, GA: {st_h['goals_conceded']})")
    print(f"   {a}: #{st_a['position']} | Pts: {st_a['points']} | GD: {st_a['goal_diff']:+d} (GF: {st_a['goals_scored']}, GA: {st_a['goals_conceded']})")
    print("-" * 68)
    print("  5. TOP 5 MOST LIKELY SCORELINES:")
    for s in res["top_scorelines"][:5]:
        score_str = f"{s['home_goals']} - {s['away_goals']}"
        print(f"   {h} {score_str} {a:<15} : {s['probability'] * 100:.1f}%")
    print("-" * 68)
    print(f"  Over 2.5 Goals: {res['over_under_25']['over']}%  |  Under 2.5 Goals: {res['over_under_25']['under']}%")
    print(f"  Both Teams to Score (BTTS): Yes {res['btts']['yes']}%  |  No {res['btts']['no']}%")
    print(f"  Dynamic Elo:    {h}: {res['team_metrics']['home_elo']}  |  {a}: {res['team_metrics']['away_elo']}")
    print("=" * 68 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Premier League Match Outcome and Scoreline")
    parser.add_argument("--home", type=str, default="Arsenal", help="Home Team Name")
    parser.add_argument("--away", type=str, default="Chelsea", help="Away Team Name")
    args = parser.parse_args()

    try:
        prediction = predict_match(args.home, args.away)
        print_prediction_cli(prediction)
    except Exception as e:
        print(f"Error predicting match: {e}")
