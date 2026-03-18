import streamlit as st
import pandas as pd
from data_loader import load_all_data
from processing import process_data
from models import train_model
from utils import get_player_prediction

st.set_page_config(page_title="Stat Merchant", layout="wide")

# ================== PREMIUM UI ==================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Playfair+Display:wght@500;700&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    background-color: #0a192f;
    color: #e6edf3;
}

.main {
    background-color: #0a192f;
}

/* Title */
.title {
    font-family: 'Playfair Display', serif;
    font-size: 64px;
    font-weight: 700;
    color: #f0f6fc;
    margin-bottom: 10px;
}

/* Tagline */
.tagline {
    font-size: 20px;
    color: #9fb3c8;
    margin-bottom: 8px;
}

/* Subtext */
.subtext {
    font-size: 14px;
    color: #6b85a6;
    font-style: italic;
    margin-bottom: 40px;
}

/* Section */
.section {
    font-size: 26px;
    font-weight: 600;
    margin-top: 40px;
    margin-bottom: 10px;
    color: #dce6f2;
}

/* Input */
.stTextInput>div>div>input {
    background-color: #112240;
    color: white;
    border: 1px solid #1f4068;
}

/* Tables */
.stDataFrame {
    background-color: #112240;
}
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown("""
<div class="title">Stat Merchant</div>
<div class="tagline">ball knowledge, certified by stats.</div>
<div class="subtext">
here for the first time? please wait for 30-50 seconds. its be so very quick the next time youre here i promise
</div>
""", unsafe_allow_html=True)

# ================== DATA PIPELINE ==================
@st.cache_data(ttl=300)
def load_pipeline():
    teams, playerstats, gameweek_summaries, players, player_gw_stats = load_all_data()
    df = process_data(teams, playerstats, players, player_gw_stats)

    feature_cols = [
        'form_gw', 'points_per_game_gw', 'value_form_gw', 'selected_by_percent_gw',
        'minutes_gw', 'total_points_gw',
        'goals_scored_gw', 'assists_gw', 'clean_sheets_gw',
        'bps_gw', 'ict_index_gw',
        'expected_goals_gw', 'expected_assists_gw', 'expected_goal_involvements_gw',
        'expected_goals_conceded_gw',
        'influence_gw', 'creativity_gw', 'threat_gw',

        'event_points_gw_roll3', 'goals_scored_gw_roll3',
        'assists_gw_roll3', 'expected_goals_gw_roll3', 'expected_assists_gw_roll3',

        'team_avg_points', 'team_strength_avg',
        'opp_difficulty_proxy',

        'form_season', 'points_per_game_season', 'total_points_season',
        'expected_goals_season', 'expected_assists_season',
        'expected_goal_involvements_season', 'value_form_season', 'value_season_season',
        'influence_season', 'creativity_season', 'threat_season', 'ict_index_season'
    ]

    feature_cols = [col for col in feature_cols if col in df.columns]

    model = train_model(df, feature_cols)

    X_all = df[feature_cols].select_dtypes(include=['number']).fillna(0)
    df['predicted_next_points'] = model.predict(X_all)

    return df

df_clean = load_pipeline()

# ================== INPUT SECTION ==================
st.markdown('<div class="section">Player Analysis</div>', unsafe_allow_html=True)

player_name = st.text_input("Search player")

if player_name:
    results = get_player_prediction(df_clean, player_name)
    st.dataframe(results, use_container_width=True)

# ================== ANALYTICS ==================
st.markdown('<div class="section">Top Players</div>', unsafe_allow_html=True)

top_players = df_clean[['web_name_gw','predicted_next_points']]\
    .drop_duplicates()\
    .sort_values('predicted_next_points', ascending=False)\
    .head(10)

st.dataframe(top_players, use_container_width=True)



