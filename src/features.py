import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional
from src.config import INITIAL_ELO, ELO_K_FACTOR, HOME_ADVANTAGE_ELO

class EloTracker:
    """Tracks dynamic Elo ratings for clubs chronologically."""
    def __init__(self, k_factor: float = ELO_K_FACTOR, home_bonus: float = HOME_ADVANTAGE_ELO, default_elo: float = INITIAL_ELO):
        self.k_factor = k_factor
        self.home_bonus = home_bonus
        self.default_elo = default_elo
        self.ratings: Dict[str, float] = {}

    def get_elo(self, team: str) -> float:
        return self.ratings.get(team, self.default_elo)

    def expected_outcome(self, elo_a: float, elo_b: float) -> float:
        """Expected score of A against B: 1 / (1 + 10^((elo_b - elo_a)/400))."""
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    def update(self, home_team: str, away_team: str, home_goals: int, away_goals: int) -> Tuple[float, float]:
        """
        Record match outcome and update ratings.
        Returns pre-match ratings: (home_elo, away_elo).
        """
        home_elo = self.get_elo(home_team)
        away_elo = self.get_elo(away_team)

        exp_home = self.expected_outcome(home_elo + self.home_bonus, away_elo)
        exp_away = 1.0 - exp_home

        if home_goals > away_goals:
            actual_home, actual_away = 1.0, 0.0
        elif home_goals == away_goals:
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0.0, 1.0

        gd = abs(home_goals - away_goals)
        if gd <= 1:
            margin_mult = 1.0
        elif gd == 2:
            margin_mult = 1.5
        else:
            margin_mult = (11.0 + gd) / 8.0

        new_home_elo = home_elo + self.k_factor * margin_mult * (actual_home - exp_home)
        new_away_elo = away_elo + self.k_factor * margin_mult * (actual_away - exp_away)

        self.ratings[home_team] = new_home_elo
        self.ratings[away_team] = new_away_elo

        return home_elo, away_elo


class StandingsTracker:
    """Tracks seasonal Premier League tables and standings match-by-match."""
    def __init__(self):
        self.current_season = None
        self.table: Dict[str, Dict[str, int]] = {}
        self.home_records: Dict[str, Dict[str, int]] = {}
        self.away_records: Dict[str, Dict[str, int]] = {}

    def _init_team(self, team: str):
        if team not in self.table:
            self.table[team] = {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "gd": 0, "pts": 0}
        if team not in self.home_records:
            self.home_records[team] = {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "gd": 0, "pts": 0}
        if team not in self.away_records:
            self.away_records[team] = {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "gd": 0, "pts": 0}

    def start_season_if_new(self, season: str):
        if season != self.current_season:
            self.current_season = season
            self.table = {}
            self.home_records = {}
            self.away_records = {}

    def get_rankings(self) -> Dict[str, int]:
        """Rank teams by: Points -> Goal Difference -> Goals Scored."""
        if not self.table:
            return {}
        sorted_teams = sorted(
            self.table.items(),
            key=lambda item: (item[1]["pts"], item[1]["gd"], item[1]["gf"]),
            reverse=True
        )
        return {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}

    def get_team_table_stats(self, team: str) -> Dict[str, Any]:
        self._init_team(team)
        rankings = self.get_rankings()
        pos = rankings.get(team, 10)
        overall = self.table[team]
        home = self.home_records[team]
        away = self.away_records[team]
        return {
            "position": pos,
            "played": overall["p"],
            "won": overall["w"],
            "drawn": overall["d"],
            "lost": overall["l"],
            "gf": overall["gf"],
            "ga": overall["ga"],
            "gd": overall["gd"],
            "points": overall["pts"],
            "home_record": home,
            "away_record": away,
        }

    def update(self, home_team: str, away_team: str, home_goals: int, away_goals: int):
        self._init_team(home_team)
        self._init_team(away_team)

        # Home team update
        h = self.table[home_team]
        h_rec = self.home_records[home_team]
        h["p"] += 1
        h_rec["p"] += 1
        h["gf"] += home_goals
        h_rec["gf"] += home_goals
        h["ga"] += away_goals
        h_rec["ga"] += away_goals
        h["gd"] = h["gf"] - h["ga"]
        h_rec["gd"] = h_rec["gf"] - h_rec["ga"]

        # Away team update
        a = self.table[away_team]
        a_rec = self.away_records[away_team]
        a["p"] += 1
        a_rec["p"] += 1
        a["gf"] += away_goals
        a_rec["gf"] += away_goals
        a["ga"] += home_goals
        a_rec["ga"] += home_goals
        a["gd"] = a["gf"] - a["ga"]
        a_rec["gd"] = a_rec["gf"] - a_rec["ga"]

        if home_goals > away_goals:
            h["w"] += 1
            h["pts"] += 3
            h_rec["w"] += 1
            h_rec["pts"] += 3
            a["l"] += 1
            a_rec["l"] += 1
        elif home_goals == away_goals:
            h["d"] += 1
            h["pts"] += 1
            h_rec["d"] += 1
            h_rec["pts"] += 1
            a["d"] += 1
            a["pts"] += 1
            a_rec["d"] += 1
            a_rec["pts"] += 1
        else:
            a["w"] += 1
            a["pts"] += 3
            a_rec["w"] += 1
            a_rec["pts"] += 3
            h["l"] += 1
            h_rec["l"] += 1


class FeatureEngineer:
    """
    Computes rolling match statistics, head-to-head records,
    schedule density, league positions, goal differences, and Elo ratings
    strictly without lookahead bias.
    """
    def __init__(self, elo_k: float = ELO_K_FACTOR, home_bonus: float = HOME_ADVANTAGE_ELO):
        self.elo_k = elo_k
        self.home_bonus = home_bonus

    def build_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, EloTracker, Dict]:
        df = df.sort_values(by=["Date", "HomeTeam"]).reset_index(drop=True).copy()

        elo_tracker = EloTracker(k_factor=self.elo_k, home_bonus=self.home_bonus)
        standings_tracker = StandingsTracker()

        team_history: Dict[str, List[Dict]] = {}
        team_home_history: Dict[str, List[Dict]] = {}
        team_away_history: Dict[str, List[Dict]] = {}
        h2h_history: Dict[Tuple[str, str], List[Dict]] = {}

        feature_rows = []

        for idx, row in df.iterrows():
            date = row["Date"]
            season = str(row.get("Season", "2425"))
            home_team = row["HomeTeam"]
            away_team = row["AwayTeam"]
            fthg = int(row["FTHG"])
            ftag = int(row["FTAG"])
            ftr = row["FTR"]

            standings_tracker.start_season_if_new(season)

            # 1. Elo Ratings before match
            home_elo_pre = elo_tracker.get_elo(home_team)
            away_elo_pre = elo_tracker.get_elo(away_team)
            elo_diff = (home_elo_pre + self.home_bonus) - away_elo_pre

            # 2. League Standings & Goal Difference before match
            home_table = standings_tracker.get_team_table_stats(home_team)
            away_table = standings_tracker.get_team_table_stats(away_team)
            home_pos = home_table["position"]
            away_pos = away_table["position"]
            pos_diff = away_pos - home_pos # Positive means home team is higher placed
            home_table_gd = home_table["gd"]
            away_table_gd = away_table["gd"]
            table_gd_diff = home_table_gd - away_table_gd

            # 3. Rolling Form (Last 5 & Last 3 matches)
            def compute_rolling_stats(history: List[Dict], n: int = 5) -> Dict[str, Any]:
                if not history:
                    return {
                        "points_sum": 6,
                        "points_avg": 1.2,
                        "goals_scored_sum": 6,
                        "goals_scored_avg": 1.2,
                        "goals_conceded_sum": 6,
                        "goals_conceded_avg": 1.2,
                        "goal_diff_sum": 0,
                        "goal_diff_avg": 0.0,
                        "win_rate": 0.33,
                        "loss_rate": 0.33,
                        "sot_avg": 4.5,
                        "sot_conceded_avg": 4.5,
                        "form_str": "D-D-D-D-D",
                        "recent_matches": []
                    }
                recent = history[-n:]
                pts = [m["points"] for m in recent]
                gs = [m["goals_scored"] for m in recent]
                gc = [m["goals_conceded"] for m in recent]
                sot = [m.get("sot", 4.5) for m in recent]
                sot_opp = [m.get("sot_conceded", 4.5) for m in recent]
                wins = sum(1 for p in pts if p == 3)
                losses = sum(1 for p in pts if p == 0)
                count = len(recent)
                form_codes = ["W" if p == 3 else ("D" if p == 1 else "L") for p in pts]
                return {
                    "points_sum": sum(pts),
                    "points_avg": sum(pts) / count,
                    "goals_scored_sum": sum(gs),
                    "goals_scored_avg": sum(gs) / count,
                    "goals_conceded_sum": sum(gc),
                    "goals_conceded_avg": sum(gc) / count,
                    "goal_diff_sum": sum(gs) - sum(gc),
                    "goal_diff_avg": (sum(gs) - sum(gc)) / count,
                    "win_rate": wins / count,
                    "loss_rate": losses / count,
                    "sot_avg": sum(sot) / count,
                    "sot_conceded_avg": sum(sot_opp) / count,
                    "form_str": "-".join(form_codes),
                    "recent_matches": recent
                }

            home_hist = team_history.get(home_team, [])
            away_hist = team_history.get(away_team, [])

            home_form_5 = compute_rolling_stats(home_hist, 5)
            away_form_5 = compute_rolling_stats(away_hist, 5)
            home_form_3 = compute_rolling_stats(home_hist, 3)
            away_form_3 = compute_rolling_stats(away_hist, 3)

            # 4. Venue-Specific Form (Home team at home, Away team away)
            home_venue_hist = team_home_history.get(home_team, [])
            away_venue_hist = team_away_history.get(away_team, [])
            home_venue_form = compute_rolling_stats(home_venue_hist, 5)
            away_venue_form = compute_rolling_stats(away_venue_hist, 5)

            # 5. Head-to-Head (H2H) past record
            h2h_key = tuple(sorted([home_team, away_team]))
            h2h_matches = h2h_history.get(h2h_key, [])
            if h2h_matches:
                recent_h2h = h2h_matches[-5:]
                h2h_home_wins = sum(1 for m in recent_h2h if m["winner"] == home_team)
                h2h_away_wins = sum(1 for m in recent_h2h if m["winner"] == away_team)
                h2h_draws = sum(1 for m in recent_h2h if m["winner"] == "Draw")
                h2h_n = len(recent_h2h)
                h2h_home_win_ratio = h2h_home_wins / h2h_n
                h2h_away_win_ratio = h2h_away_wins / h2h_n
            else:
                h2h_home_win_ratio = 0.33
                h2h_away_win_ratio = 0.33
                h2h_home_wins = 0
                h2h_away_wins = 0
                h2h_draws = 0

            # 6. Rest days
            last_date_home = home_hist[-1]["date"] if home_hist else None
            last_date_away = away_hist[-1]["date"] if away_hist else None
            home_rest = (date - last_date_home).days if last_date_home else 7
            away_rest = (date - last_date_away).days if last_date_away else 7
            home_rest = min(max(home_rest, 2), 21)
            away_rest = min(max(away_rest, 2), 21)
            rest_diff = home_rest - away_rest

            # 7. Pressure metrics
            expected_home_pressure = home_form_5["goals_scored_avg"] - away_form_5["goals_conceded_avg"]
            expected_away_pressure = away_form_5["goals_scored_avg"] - home_form_5["goals_conceded_avg"]

            # Store computed row
            feat_row = {
                "Date": date,
                "Season": season,
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "FTHG": fthg,
                "FTAG": ftag,
                "FTR": ftr,
                "TotalGoals": fthg + ftag,
                "Over25": 1 if (fthg + ftag) > 2.5 else 0,
                "BTTS": 1 if (fthg > 0 and ftag > 0) else 0,
                # Elo features
                "home_elo": home_elo_pre,
                "away_elo": away_elo_pre,
                "elo_diff": elo_diff,
                # League Position & Table GD features
                "home_league_pos": home_pos,
                "away_league_pos": away_pos,
                "pos_diff": pos_diff,
                "home_table_gd": home_table_gd,
                "away_table_gd": away_table_gd,
                "table_gd_diff": table_gd_diff,
                # Form features (Last 5)
                "home_pts_sum_5": home_form_5["points_sum"],
                "away_pts_sum_5": away_form_5["points_sum"],
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
                # Form features (Last 3)
                "home_pts_avg_3": home_form_3["points_avg"],
                "away_pts_avg_3": away_form_3["points_avg"],
                "pts_diff_3": home_form_3["points_avg"] - away_form_3["points_avg"],
                "home_gd_avg_3": home_form_3["goal_diff_avg"],
                "away_gd_avg_3": away_form_3["goal_diff_avg"],
                # Venue-specific form
                "home_team_home_win_rate": home_venue_form["win_rate"],
                "away_team_away_win_rate": away_venue_form["win_rate"],
                "home_team_home_gd": home_venue_form["goal_diff_avg"],
                "away_team_away_gd": away_venue_form["goal_diff_avg"],
                # H2H
                "h2h_home_win_ratio": h2h_home_win_ratio,
                "h2h_away_win_ratio": h2h_away_win_ratio,
                # Rest & Matchup
                "home_rest_days": home_rest,
                "away_rest_days": away_rest,
                "rest_diff": rest_diff,
                "expected_home_pressure": expected_home_pressure,
                "expected_away_pressure": expected_away_pressure,
            }
            feature_rows.append(feat_row)

            # Update match trackers with actual result
            home_pts = 3 if ftr == "H" else (1 if ftr == "D" else 0)
            away_pts = 3 if ftr == "A" else (1 if ftr == "D" else 0)
            home_sot = row.get("HST", 4)
            away_sot = row.get("AST", 4)
            res_home = "W" if ftr == "H" else ("D" if ftr == "D" else "L")
            res_away = "W" if ftr == "A" else ("D" if ftr == "D" else "L")

            date_str = date.strftime("%d %b %Y") if hasattr(date, "strftime") else str(date)[:10]

            home_record = {
                "date": date,
                "date_str": date_str,
                "opponent": away_team,
                "is_home": True,
                "venue": "Home",
                "result": res_home,
                "score": f"{fthg} - {ftag}",
                "points": home_pts,
                "goals_scored": fthg,
                "goals_conceded": ftag,
                "sot": home_sot,
                "sot_conceded": away_sot,
            }
            away_record = {
                "date": date,
                "date_str": date_str,
                "opponent": home_team,
                "is_home": False,
                "venue": "Away",
                "result": res_away,
                "score": f"{ftag} - {fthg}",
                "points": away_pts,
                "goals_scored": ftag,
                "goals_conceded": fthg,
                "sot": away_sot,
                "sot_conceded": home_sot,
            }

            team_history.setdefault(home_team, []).append(home_record)
            team_history.setdefault(away_team, []).append(away_record)
            team_home_history.setdefault(home_team, []).append(home_record)
            team_away_history.setdefault(away_team, []).append(away_record)

            winner = home_team if ftr == "H" else (away_team if ftr == "A" else "Draw")
            h2h_history.setdefault(h2h_key, []).append({
                "date": date,
                "date_str": date_str,
                "home_team": home_team,
                "away_team": away_team,
                "score": f"{fthg} - {ftag}",
                "winner": winner,
                "fthg": fthg,
                "ftag": ftag
            })

            # Update Standings & Elo AFTER recording pre-match features
            standings_tracker.update(home_team, away_team, fthg, ftag)
            elo_tracker.update(home_team, away_team, fthg, ftag)

        features_df = pd.DataFrame(feature_rows)

        # Snapshot of final state
        latest_team_state = {
            "team_history": team_history,
            "team_home_history": team_home_history,
            "team_away_history": team_away_history,
            "h2h_history": h2h_history,
            "ratings": elo_tracker.ratings,
            "standings_table": standings_tracker.table,
            "home_records": standings_tracker.home_records,
            "away_records": standings_tracker.away_records,
            "rankings": standings_tracker.get_rankings()
        }

        return features_df, elo_tracker, latest_team_state

FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_diff",
    "home_league_pos", "away_league_pos", "pos_diff",
    "home_table_gd", "away_table_gd", "table_gd_diff",
    "home_pts_sum_5", "away_pts_sum_5",
    "home_pts_avg_5", "away_pts_avg_5", "pts_diff_5",
    "home_gs_avg_5", "away_gs_avg_5", "home_gc_avg_5", "away_gc_avg_5",
    "home_gd_avg_5", "away_gd_avg_5",
    "home_win_rate_5", "away_win_rate_5", "home_loss_rate_5", "away_loss_rate_5",
    "home_sot_avg_5", "away_sot_avg_5",
    "home_pts_avg_3", "away_pts_avg_3", "pts_diff_3",
    "home_gd_avg_3", "away_gd_avg_3",
    "home_team_home_win_rate", "away_team_away_win_rate",
    "home_team_home_gd", "away_team_away_gd",
    "h2h_home_win_ratio", "h2h_away_win_ratio",
    "home_rest_days", "away_rest_days", "rest_diff",
    "expected_home_pressure", "expected_away_pressure"
]
