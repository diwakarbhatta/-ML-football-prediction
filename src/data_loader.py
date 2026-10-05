import io
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import pandas as pd
from typing import List, Optional

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SEASONS, TEAM_ALIASES

BASE_URL = "https://www.football-data.co.uk/mmz4281"

def standardize_team_name(name: str) -> str:
    """Normalize team names to canonical representations."""
    if not isinstance(name, str):
        return name
    name_clean = name.strip()
    return TEAM_ALIASES.get(name_clean, name_clean)

def download_season_data(season_code: str, force_download: bool = False) -> Path:
    """Download single season CSV from football-data.co.uk and cache locally."""
    target_file = RAW_DATA_DIR / f"epl_{season_code}.csv"
    if target_file.exists() and not force_download:
        return target_file

    url = f"{BASE_URL}/{season_code}/E0.csv"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    print(f"Downloading EPL season {season_code} from {url}...")
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    with open(target_file, "wb") as f:
        f.write(response.content)
    return target_file

def parse_date(date_str: str) -> pd.Timestamp:
    """Parse match dates supporting various UK date formats."""
    if pd.isna(date_str):
        return pd.NaT
    date_str = str(date_str).strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(date_str, errors="coerce")

def load_and_preprocess_all(seasons: Optional[List[str]] = None, force_download: bool = False) -> pd.DataFrame:
    """
    Download and clean multi-season EPL match records.
    Returns a unified chronological DataFrame.
    """
    if seasons is None:
        seasons = SEASONS

    dfs = []
    core_columns = [
        "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
        "B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA"
    ]

    for season in seasons:
        csv_path = download_season_data(season, force_download=force_download)
        try:
            df = pd.read_csv(csv_path, encoding="latin1")
        except Exception:
            df = pd.read_csv(csv_path, encoding="utf-8", errors="ignore")

        # Keep existing available columns
        available_cols = [c for c in core_columns if c in df.columns]
        df_subset = df[available_cols].copy()
        df_subset["Season"] = season

        # Parse Date
        df_subset["Date"] = df_subset["Date"].apply(parse_date)
        df_subset = df_subset.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])

        # Standardize team names
        df_subset["HomeTeam"] = df_subset["HomeTeam"].apply(standardize_team_name)
        df_subset["AwayTeam"] = df_subset["AwayTeam"].apply(standardize_team_name)

        # Convert goals and results
        df_subset["FTHG"] = df_subset["FTHG"].astype(int)
        df_subset["FTAG"] = df_subset["FTAG"].astype(int)
        df_subset["FTR"] = df_subset["FTR"].astype(str).str.strip().str.upper()

        # Only retain valid outcomes: H (Home win), D (Draw), A (Away win)
        df_subset = df_subset[df_subset["FTR"].isin(["H", "D", "A"])]

        # Ensure numeric for match stats
        num_cols = ["HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
                    "B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA"]
        for c in num_cols:
            if c in df_subset.columns:
                df_subset[c] = pd.to_numeric(df_subset[c], errors="coerce")

        dfs.append(df_subset)

    merged = pd.concat(dfs, ignore_index=True)
    # Sort strictly chronologically
    merged = merged.sort_values(by=["Date", "HomeTeam"]).reset_index(drop=True)

    # Save processed dataset
    output_path = PROCESSED_DATA_DIR / "epl_matches.csv"
    merged.to_csv(output_path, index=False)
    print(f"Loaded {len(merged)} matches across {len(seasons)} seasons. Saved to {output_path}")
    return merged

if __name__ == "__main__":
    df = load_and_preprocess_all()
    print("Dataset Summary:")
    print(df.info())
    print("Outcome Distribution:")
    print(df["FTR"].value_counts(normalize=True))
