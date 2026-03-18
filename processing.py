import pandas as pd
import numpy as np

def process_data(teams, playerstats, players, player_gw_stats):
    teams = teams.rename(columns={"id": "team_id"})

    player_gw_stats = player_gw_stats.rename(columns={"id": "player_id"})
    playerstats = playerstats.rename(columns={"id": "player_id"})

    merged_df = (
        player_gw_stats
        .merge(playerstats, on="player_id", suffixes=("_gw", "_season"))
        .merge(players, on="player_id", how="left")
        .merge(teams, left_on="team_code", right_on="team_id", how="left")
    )

    df_clean = merged_df.copy()

    # CLEANING
    threshold = 0.6
    df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isnull().mean() > threshold])

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns

    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    df_clean[non_numeric_cols] = df_clean[non_numeric_cols].fillna("Unknown")

    # TARGET
    df_clean['next_gw_points'] = df_clean.groupby('player_id')['event_points_gw'].shift(-1)

    return df_clean
