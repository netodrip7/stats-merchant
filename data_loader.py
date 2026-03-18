import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/olbauday/FPL-Elo-Insights/main/data/2025-2026"
BY_TOURNAMENT = f"{BASE_URL}/By%20Tournament/Premier%20League"

def load_all_data():
    teams = pd.read_csv(f"{BASE_URL}/teams.csv")
    playerstats = pd.read_csv(f"{BASE_URL}/playerstats.csv")
    gameweek_summaries = pd.read_csv(f"{BASE_URL}/gameweek_summaries.csv")
    players = pd.read_csv(f"{BY_TOURNAMENT}/GW11/players.csv")

    gw_data = []
    for i in range(1, 39):
        try:
            df = pd.read_csv(f"{BY_TOURNAMENT}/GW{i}/player_gameweek_stats.csv")
            df["gameweek"] = i
            gw_data.append(df)
        except:
            pass

    player_gw_stats = pd.concat(gw_data, ignore_index=True)

    return teams, playerstats, gameweek_summaries, players, player_gw_stats
