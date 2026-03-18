import streamlit as st
import pandas as pd
import numpy as np
import requests
import unicodedata
import re

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ===============================================
# 🎨 PAGE CONFIG + STYLING
# ===============================================
st.set_page_config(layout="wide")

st.markdown("""
<style>
body {
    background-color: #0a0f2c;
    color: white;
}
.big-title {
    text-align: center;
    font-size: 60px;
    font-weight: 800;
    color: #4da6ff;
}
.tagline {
    text-align: center;
    font-size: 22px;
    color: #7ec8ff;
}
.small-note {
    text-align: center;
    font-size: 16px;
    color: #a3cfff;
}
.desc {
    text-align: center;
    font-size: 20px;
    color: #cce6ff;
    line-height: 1.3;
}
.input-label {
    color: #9ed0ff;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">STATS MERCHANT</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">ball knowledge backed by stats</div>', unsafe_allow_html=True)
st.markdown('<div class="small-note">here for the first time? please wait for 30-50 seconds. it’ll be so much quicker next time.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="desc">
FPL managers, tap in.<br>
This ain’t just another stats site, this is your differential factory.<br>
Get clean data, fixture swings, xG juice, and captaincy calls that actually hit.<br>
No more picking your team on vibes only — we move with data now.
</div>
""", unsafe_allow_html=True)

# ===============================================
# ⚡ CACHE EVERYTHING (CRUCIAL FOR SPEED)
# ===============================================
@st.cache_data(show_spinner=True)
def load_and_train():
    # ===============================================
    # 📥 LOAD DATA
    # ===============================================
    base_url = "https://raw.githubusercontent.com/olbauday/FPL-Elo-Insights/main/data/2025-2026"
    by_tournament = f"{base_url}/By%20Tournament/Premier%20League"

    urls = {
        "teams": f"{base_url}/teams.csv",
        "players": f"{by_tournament}/GW11/players.csv",
        "playerstats": f"{base_url}/playerstats.csv",
        "gameweek_summaries": f"{base_url}/gameweek_summaries.csv"
    }

    teams = pd.read_csv(urls["teams"])
    playerstats = pd.read_csv(urls["playerstats"])
    players = pd.read_csv(urls["players"])

    gw_data = []
    for i in range(1, 39):
        try:
            df = pd.read_csv(f"{by_tournament}/GW{i}/player_gameweek_stats.csv")
            df["gameweek"] = i
            gw_data.append(df)
        except:
            pass

    player_gw_stats = pd.concat(gw_data, ignore_index=True)

    # ===============================================
    # 🔗 MERGE
    # ===============================================
    player_gw_stats = player_gw_stats.rename(columns={"id": "player_id"})
    playerstats = playerstats.rename(columns={"id": "player_id"})
    teams = teams.rename(columns={"id": "team_id"})

    df = (
        player_gw_stats
        .merge(playerstats, on="player_id", suffixes=("_gw", "_season"))
        .merge(players, on="player_id", how="left")
        .merge(teams, left_on="team_code", right_on="team_id", how="left")
    )

    # ===============================================
    # 🧹 CLEANING
    # ===============================================
    threshold = 0.6
    df = df.drop(columns=df.columns[df.isnull().mean() > threshold])

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("Unknown").infer_objects(copy=False)

    # TARGET
    df['next_gw_points'] = df.groupby('player_id')['event_points_gw'].shift(-1)

    df = df.sort_values(['player_id', 'gameweek'])

    # ===============================================
    # ⚙️ FEATURE ENGINEERING
    # ===============================================
    for col in ['event_points_gw','goals_scored_gw','assists_gw']:
        if col in df.columns:
            df[f'{col}_roll3'] = df.groupby('player_id')[col].transform(lambda x: x.rolling(3,1).mean())

    team_form = df.groupby(['team_id','gameweek'])['event_points_gw'].mean().reset_index(name='team_avg_points')
    df = df.merge(team_form, on=['team_id','gameweek'], how='left')

    df['opp_difficulty_proxy'] = df['strength']
    df['team_strength_avg'] = df['strength']

    # ===============================================
    # 🤖 MODEL (XGBOOST)
    # ===============================================
    features = ['form_gw','points_per_game_gw','minutes_gw','total_points_gw',
                'event_points_gw_roll3','team_avg_points','team_strength_avg','opp_difficulty_proxy']

    features = [f for f in features if f in df.columns]

    df_model = df[df['next_gw_points'].notnull()]
    X = df_model[features].fillna(0)
    y = df_model['next_gw_points']

    model = XGBRegressor(n_estimators=150, max_depth=5)
    model.fit(X, y)

    df['predicted_next_points'] = model.predict(df[features].fillna(0))

    # ===============================================
    # 🤖 LOGISTIC MODEL
    # ===============================================
    df['value_for_money'] = df['predicted_next_points'] / df['now_cost_gw']

    def label(row):
        if row['predicted_next_points'] >= 10: return "Start"
        elif row['predicted_next_points'] >= 6: return "Bench"
        else: return "Sell"

    df['label'] = df.apply(label, axis=1)

    clf = LogisticRegression(max_iter=500)
    clf.fit(df[features].fillna(0), df['label'])

    df['recommendation'] = clf.predict(df[features].fillna(0))

    return df, teams

df_clean, teams = load_and_train()

# ===============================================
# 🔎 USER INPUTS
# ===============================================
st.markdown("### 🔍 Enter a player’s name to see predicted points")
player_name = st.text_input("")

if player_name:
    results = df_clean[df_clean['web_name_gw'].str.contains(player_name, case=False, na=False)]
    st.dataframe(results[['web_name_gw','predicted_next_points']].sort_values(by='predicted_next_points', ascending=False).head(10))

st.markdown("### 🔄 Enter a player’s name to get replacements")
rep_name = st.text_input(" ", key="rep")

if rep_name:
    player = df_clean[df_clean['web_name_gw'].str.contains(rep_name, case=False, na=False)].iloc[0]
    pos = player['position']

    candidates = df_clean[df_clean['position'] == pos]
    candidates = candidates[candidates['player_id'] != player['player_id']]

    candidates['score'] = candidates['predicted_next_points']*0.6

    st.dataframe(candidates.sort_values('score', ascending=False).head(5)[
        ['web_name_gw','predicted_next_points','value_for_money']
    ])

st.markdown("### 🧠 Enter player to get recommendation")
rec_name = st.text_input("  ", key="rec")

if rec_name:
    player = df_clean[df_clean['web_name_gw'].str.contains(rec_name, case=False, na=False)].iloc[0]
    st.write("Recommendation:", player['recommendation'])

# ===============================================
# 📊 ALWAYS SHOW DATA
# ===============================================
df_latest = df_clean.sort_values(['player_id','gameweek']).drop_duplicates('player_id', keep='last')

st.markdown("## ⚡ Top Players by Position")
st.dataframe(
    df_latest.sort_values('predicted_next_points', ascending=False)
    .groupby('position').head(5)[['web_name_gw','position','predicted_next_points']]
)

st.markdown("## 💸 Best Value Players")
st.dataframe(
    df_latest.sort_values('value_for_money', ascending=False)
    .groupby('position').head(5)[['web_name_gw','value_for_money']]
)

st.markdown("## 🧱 Team Difficulty")
st.dataframe(
    df_latest.groupby('team_name_final')['opp_difficulty_proxy']
    .mean().sort_values()
)

st.markdown("## ⚽ Team Predicted Points")
st.dataframe(
    df_latest.groupby('team_name_final')['predicted_next_points']
    .sum().sort_values(ascending=False)
)



