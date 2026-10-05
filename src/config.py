import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Historical Premier League seasons to download (from 2018/19 to 2024/25)
SEASONS = [
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425"
]

# Standard Premier League team names normalization
TEAM_ALIASES = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Tottenham": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers",
    "Brighton": "Brighton & Hove Albion",
    "West Ham": "West Ham United",
    "Newcastle": "Newcastle United",
    "Leicester": "Leicester City",
    "Leeds": "Leeds United",
    "Norwich": "Norwich City",
    "Sheffield United": "Sheffield United",
    "Nott'm Forest": "Nottingham Forest",
    "Luton": "Luton Town",
    "Ipswich": "Ipswich Town",
    "Southampton": "Southampton",
    "Bournemouth": "AFC Bournemouth",
    "Brentford": "Brentford",
    "Fulham": "Fulham",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Chelsea": "Chelsea",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Aston Villa": "Aston Villa",
    "Watford": "Watford",
    "Burnley": "Burnley",
    "West Brom": "West Bromwich Albion",
    "Huddersfield": "Huddersfield Town",
    "Cardiff": "Cardiff City"
}

# Current Premier League (2024/2025 - 2025/2026) active clubs
CURRENT_EPL_TEAMS = sorted([
    "Arsenal",
    "Aston Villa",
    "AFC Bournemouth",
    "Brentford",
    "Brighton & Hove Albion",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Ipswich Town",
    "Leicester City",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Southampton",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers"
])

# Elo configuration
INITIAL_ELO = 1500.0
ELO_K_FACTOR = 30.0
HOME_ADVANTAGE_ELO = 65.0
