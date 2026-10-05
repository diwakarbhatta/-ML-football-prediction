import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from math import factorial, exp
from typing import Dict, Tuple, List, Any
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, classification_report, brier_score_loss

def dixon_coles_tau(x: int, y: int, lambda_h: float, lambda_a: float, rho: float = -0.11) -> float:
    """
    Dixon-Coles adjustment factor for low-scoring match dependence.
    Corrects the standard independent Poisson model for 0-0, 1-0, 0-1, 1-1 scores.
    """
    if x == 0 and y == 0:
        return 1.0 - lambda_h * lambda_a * rho
    elif x == 0 and y == 1:
        return 1.0 + lambda_h * rho
    elif x == 1 and y == 0:
        return 1.0 + lambda_a * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0

def poisson_pmf(k: int, lambd: float) -> float:
    """Poisson probability mass function P(X = k | lambd)."""
    if lambd <= 0.0:
        return 1.0 if k == 0 else 0.0
    return (lambd ** k) * exp(-lambd) / factorial(k)


class MatchOutcomeModel:
    """
    Multi-class classifier predicting match outcome (Home Win 'H', Draw 'D', Away Win 'A').
    Uses an ensemble of calibrated HistGradientBoosting and Logistic Regression.
    """
    def __init__(self):
        # We order classes as ['H', 'D', 'A']
        self.classes_ = ["H", "D", "A"]
        
        lr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, C=0.3, random_state=42))
        ])

        hgb = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.04,
            max_depth=4,
            min_samples_leaf=20,
            l2_regularization=1.5,
            random_state=42
        )

        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=15,
            random_state=42,
            n_jobs=-1
        )

        ensemble = VotingClassifier(
            estimators=[
                ("hgb", hgb),
                ("lr", lr_pipe),
                ("rf", rf)
            ],
            voting="soft",
            weights=[2, 1, 1]
        )

        # Calibrate probabilities using sigmoid calibration
        self.model = CalibratedClassifierCV(estimator=ensemble, method="sigmoid", cv=3)
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.classes_ = list(self.model.classes_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_proba_dict(self, X: np.ndarray) -> List[Dict[str, float]]:
        """Return probabilities as a list of dicts with keys 'H', 'D', 'A'."""
        probs = self.predict_proba(X)
        results = []
        for row in probs:
            p_dict = {cls: float(row[idx]) for idx, cls in enumerate(self.classes_)}
            results.append(p_dict)
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        preds = self.predict(X)
        probs = self.predict_proba(X)
        acc = accuracy_score(y, preds)
        loss = log_loss(y, probs, labels=self.classes_)
        report = classification_report(y, preds, labels=self.classes_, output_dict=True)

        return {
            "accuracy": float(acc),
            "log_loss": float(loss),
            "classification_report": report
        }


class ScorePredictor:
    """
    Predicts expected home and away goals (Expected Goals) and derives
    exact scoreline probabilities via Dixon-Coles adjusted bivariate Poisson.
    """
    def __init__(self, rho: float = -0.10):
        self.rho = rho
        self.home_goal_model = Pipeline([
            ("scaler", StandardScaler()),
            ("poisson", PoissonRegressor(alpha=0.5, max_iter=500))
        ])
        self.away_goal_model = Pipeline([
            ("scaler", StandardScaler()),
            ("poisson", PoissonRegressor(alpha=0.5, max_iter=500))
        ])

    def fit(self, X: np.ndarray, y_home_goals: np.ndarray, y_away_goals: np.ndarray):
        self.home_goal_model.fit(X, y_home_goals)
        self.away_goal_model.fit(X, y_away_goals)
        return self

    def predict_expected_goals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict expected goals (lambda_home, lambda_away)."""
        exp_home = self.home_goal_model.predict(X)
        exp_away = self.away_goal_model.predict(X)
        # Enforce realistic bounds
        exp_home = np.clip(exp_home, 0.2, 5.0)
        exp_away = np.clip(exp_away, 0.2, 5.0)
        return exp_home, exp_away

    def predict_score_matrix(self, lambda_home: float, lambda_away: float, max_goals: int = 6) -> np.ndarray:
        """
        Calculates matrix M[i, j] = P(Home = i, Away = j)
        using Dixon-Coles adjusted Poisson distribution.
        """
        matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
        for i in range(max_goals + 1):
            p_i = poisson_pmf(i, lambda_home)
            for j in range(max_goals + 1):
                p_j = poisson_pmf(j, lambda_away)
                tau = dixon_coles_tau(i, j, lambda_home, lambda_away, self.rho)
                matrix[i, j] = max(0.0, p_i * p_j * tau)

        # Normalize matrix so probabilities sum to 1.0
        total = matrix.sum()
        if total > 0:
            matrix /= total
        return matrix

    def predict_score_distribution(self, lambda_home: float, lambda_away: float, max_goals: int = 5) -> Dict[str, Any]:
        """
        Returns full score probabilities, most likely scoreline,
        outcome probabilities derived from Poisson, Over/Under 2.5, and BTTS.
        """
        matrix = self.predict_score_matrix(lambda_home, lambda_away, max_goals=max_goals)
        
        # Most probable score
        best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        most_likely_score = (int(best_idx[0]), int(best_idx[1]))
        most_likely_prob = float(matrix[best_idx])

        # Top 5 scorelines
        flat_indices = np.argsort(matrix.ravel())[::-1][:6]
        top_scores = []
        for idx in flat_indices:
            i, j = divmod(idx, matrix.shape[1])
            top_scores.append({
                "home_goals": int(i),
                "away_goals": int(j),
                "probability": float(matrix[i, j])
            })

        # Poisson derived outcome probabilities
        prob_home = float(np.sum(np.tril(matrix, -1)))
        prob_draw = float(np.sum(np.diag(matrix)))
        prob_away = float(np.sum(np.triu(matrix, 1)))

        # Over / Under 2.5 goals
        prob_under_25 = 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i + j < 2.5:
                    prob_under_25 += matrix[i, j]
        prob_over_25 = 1.0 - prob_under_25

        # Both Teams to Score (BTTS)
        prob_btts = float(np.sum(matrix[1:, 1:]))

        return {
            "expected_goals": {"home": round(float(lambda_home), 2), "away": round(float(lambda_away), 2)},
            "most_likely_score": most_likely_score,
            "most_likely_prob": round(most_likely_prob * 100, 1),
            "top_scores": top_scores,
            "outcome_probs": {
                "Home Win": round(prob_home * 100, 1),
                "Draw": round(prob_draw * 100, 1),
                "Away Win": round(prob_away * 100, 1)
            },
            "over_under_25": {
                "over": round(prob_over_25 * 100, 1),
                "under": round(prob_under_25 * 100, 1)
            },
            "btts": {
                "yes": round(prob_btts * 100, 1),
                "no": round((1.0 - prob_btts) * 100, 1)
            },
            "score_matrix": matrix.tolist()
        }
