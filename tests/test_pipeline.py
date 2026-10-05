import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import CURRENT_EPL_TEAMS
from src.features import EloTracker, FeatureEngineer, FEATURE_COLUMNS
from src.models import ScorePredictor, MatchOutcomeModel, dixon_coles_tau, poisson_pmf
from src.predict import predict_match
from app import app

class TestEPLPredictor(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_elo_tracker(self):
        tracker = EloTracker(k_factor=30.0, home_bonus=65.0)
        h_elo, a_elo = tracker.update("Arsenal", "Chelsea", home_goals=2, away_goals=0)
        self.assertEqual(h_elo, 1500.0)
        self.assertEqual(a_elo, 1500.0)
        # Arsenal won at home, rating should have increased
        self.assertGreater(tracker.get_elo("Arsenal"), 1500.0)
        self.assertLess(tracker.get_elo("Chelsea"), 1500.0)

    def test_poisson_and_dixon_coles(self):
        pmf_0 = poisson_pmf(0, 1.5)
        self.assertGreater(pmf_0, 0.0)
        self.assertLess(pmf_0, 1.0)

        tau = dixon_coles_tau(0, 0, 1.5, 1.2, rho=-0.11)
        self.assertGreater(tau, 1.0) # Dixon-Coles boosts low score 0-0 probability

        score_pred = ScorePredictor()
        matrix = score_pred.predict_score_matrix(1.8, 1.1, max_goals=6)
        self.assertAlmostEqual(float(matrix.sum()), 1.0, places=4)

    def test_prediction_output_contract(self):
        res = predict_match("Arsenal", "Chelsea")
        self.assertIn("predicted_outcome", res)
        self.assertIn("win_probabilities", res)
        self.assertIn("expected_goals", res)
        self.assertIn("predicted_score", res)
        self.assertIn("top_scorelines", res)
        self.assertIn("score_matrix", res)

        probs = res["win_probabilities"]
        prob_sum = probs["home_win"] + probs["draw"] + probs["away_win"]
        self.assertAlmostEqual(prob_sum, 100.0, delta=1.0)

        # Expected goals should be positive realistic values
        self.assertGreater(res["expected_goals"]["home"], 0.2)
        self.assertGreater(res["expected_goals"]["away"], 0.2)

    def test_flask_routes(self):
        # 1. Main index
        resp = self.app.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"PREMIER LEAGUE ML PREDICTOR", resp.data)

        # 2. Teams endpoint
        resp_teams = self.app.get("/api/teams")
        self.assertEqual(resp_teams.status_code, 200)
        json_teams = resp_teams.get_json()
        self.assertIn("Arsenal", json_teams["teams"])

        # 3. Predict endpoint
        resp_pred = self.app.post("/api/predict", json={
            "home_team": "Liverpool",
            "away_team": "Everton"
        })
        self.assertEqual(resp_pred.status_code, 200)
        data = resp_pred.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["prediction"]["home_team"], "Liverpool")

        # 4. Metrics endpoint
        resp_metrics = self.app.get("/api/metrics")
        self.assertEqual(resp_metrics.status_code, 200)
        data_metrics = resp_metrics.get_json()
        self.assertIn("metrics", data_metrics)

if __name__ == "__main__":
    unittest.main()
