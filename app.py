import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CURRENT_EPL_TEAMS
from src.predict import predict_match, get_artifacts

app = Flask(__name__)

@app.route("/")
def index():
    artifacts = get_artifacts()
    metrics = artifacts.get("metrics", {})
    return render_template("index.html", teams=CURRENT_EPL_TEAMS, metrics=metrics)

@app.route("/api/teams", methods=["GET"])
def get_teams():
    return jsonify({"teams": CURRENT_EPL_TEAMS})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json() or {}
    home_team = data.get("home_team", "").strip()
    away_team = data.get("away_team", "").strip()

    if not home_team or not away_team:
        return jsonify({"error": "Both home_team and away_team are required."}), 400

    if home_team.lower() == away_team.lower():
        return jsonify({"error": "Home team and away team cannot be the same club."}), 400

    try:
        result = predict_match(home_team, away_team)
        return jsonify({"success": True, "prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    artifacts = get_artifacts()
    return jsonify({
        "metrics": artifacts.get("metrics", {}),
        "feature_count": len(artifacts.get("feature_columns", []))
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting EPL Predictor Web App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
