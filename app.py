import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata

# ===============================================
# ⚡ LOAD PRE-PROCESSED DATA (FAST)
# ===============================================




st.write("App starting...")

url = "https://raw.githubusercontent.com/netodrip7/stats-merchant/main/final_data.parquet"

try:
    df = pd.read_parquet(url)
    st.write(" Data loaded successfully")
    st.write(df.shape)
except Exception as e:
    st.error(f" Error loading data: {e}")
    st.stop()

# ===============================================
# 🎨 PAGE CONFIG
# ===============================================

st.set_page_config(page_title="Stats Merchant", layout="wide")

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #05070d;
    color: #e6edf3;
}

.title {
    text-align: center;
    font-size: 72px;
    font-weight: 900;
    color: #0b3d91;
    letter-spacing: 3px;
}

.tagline {
    text-align: center;
    color: #1e3a8a;
    font-size: 18px;
}

.smalltext {
    text-align: center;
    color: #6e7681;
    font-size: 14px;
}

.center-text {
    text-align: center;
    max-width: 650px;
    margin: auto;
    line-height: 1.8;
    color: #1f4ed8;
}

[data-testid="stDataFrame"] {
    background-color: #0b0f17;
    border-radius: 12px;
    border: 1px solid #1f2937;
}

thead tr th {
    color: #1f6feb !important;
    font-weight: 600 !important;
}

tbody tr {
    color: #d1d5db !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ===============================================
# 🧠 HEADER
# ===============================================

st.markdown('<div class="title">STATS MERCHANT</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">ball knowledge backed by stats</div>', unsafe_allow_html=True)
st.markdown("""
<div class="center-text">
FPL managers, tap in. this ain’t just another stats <br>
site, this is your differential factory We move with <br>
         data, not vibes <b>GET THAT RANK UP.</b>
</div>
""", unsafe_allow_html=True)

# ===============================================
# 🧹 PREP DATA
# ===============================================
df_latest = df.copy()

# ===============================================
# 🔍 SEARCH HELPER
# ===============================================

def normalize(text):
    text = unicodedata.normalize('NFKD', str(text))
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9 ]', '', text.lower())

df_latest["search_name"] = (
    df_latest["first_name_gw"].fillna('') + " " + df_latest["second_name_gw"].fillna('')
).apply(normalize)

df_latest["web_name_norm"] = df_latest["web_name_gw"].apply(normalize)

# ===============================================
# 🔮 PLAYER PREDICTION
# ===============================================

st.markdown("### Enter a player’s name to see predicted points")
player_input = st.text_input("", key="pred")

if player_input:
    q = normalize(player_input)
    res = df_latest[
        df_latest["search_name"].str.contains(q) |
        df_latest["web_name_norm"].str.contains(q)
    ]

    if res.empty:
        st.warning("No player found")
    else:
        st.dataframe(
            res[["first_name_gw","second_name_gw","team_name_final","predicted_next_points"]]
            .sort_values("predicted_next_points", ascending=False)
        )

# ===============================================
# 🧠 RECOMMENDATION
# ===============================================

st.markdown("### Enter a player’s name to get recommendation")
rec_input = st.text_input("", key="rec")

if rec_input:
    q = normalize(rec_input)
    player = df_latest[
        df_latest["search_name"].str.contains(q) |
        df_latest["web_name_norm"].str.contains(q)
    ]

    if not player.empty:
        p = player.iloc[0]
        st.success(f"""
        **{p['first_name_gw']} {p['second_name_gw']}**
        
        Predicted Points: {p['predicted_next_points']:.2f}  
        Value for Money: {p['value_for_money']:.2f}  
        Recommendation: **{p['recommendation']}**
        """)

# ===============================================
# 🔁 REPLACEMENTS
# ===============================================

st.markdown("### Enter a player’s name for replacements")
rep_input = st.text_input("", key="rep")

if rep_input:
    q = normalize(rep_input)
    player = df_latest[
        df_latest["search_name"].str.contains(q) |
        df_latest["web_name_norm"].str.contains(q)
    ]

    if not player.empty:
        p = player.iloc[0]
        pos = p["position"]

        candidates = df_latest[df_latest["position"] == pos].copy()
        candidates = candidates[candidates["player_id"] != p["player_id"]]

        candidates["score"] = (
            candidates["predicted_next_points"] * 0.5 +
            candidates["value_for_money"] * 0.3 +
            candidates["form_gw"] * 0.2
        )

        top = candidates.sort_values("score", ascending=False).head(3)

        st.dataframe(top[[
            "first_name_gw","second_name_gw","predicted_next_points","value_for_money"
        ]])

# ===============================================
# 📊 ALWAYS VISIBLE TABLES
# ===============================================

st.markdown("##  Top Players by Position")
top5 = (
    df_latest[df_latest["position"] != "Unknown"]
    .sort_values(['position','predicted_next_points'], ascending=[True,False])
    .groupby('position')
    .head(5)
)
st.dataframe(top5[["first_name_gw","second_name_gw","position","predicted_next_points"]])

st.markdown("## Best Value Players")
vfm = (
    df_latest[df_latest["position"] != "Unknown"]
    .sort_values(['position','value_for_money'], ascending=[True,False])
    .groupby('position')
    .head(5)
)
st.dataframe(vfm[["first_name_gw","second_name_gw","value_for_money"]])



st.markdown("## Team Difficulty")

st.markdown("""
Average difficulty of upcoming opponents  
Lower = easier fixtures  
Higher = tougher fixtures
""")

team_diff = (
    df_latest.groupby("team_name_final")["opp_difficulty_proxy"]
    .mean()
    .reset_index()
    .rename(columns={
        "team_name_final": "Team",
        "opp_difficulty_proxy": "Difficulty Rating"
    })
    .sort_values("Difficulty Rating")
)

st.dataframe(team_diff, use_container_width=True)

st.markdown("## Team Expected Points")

team_pts = (
    df_latest.groupby("team_name_final")["predicted_next_points"]
    .sum()
    .reset_index()
    .rename(columns={
        "team_name_final": "Team",
        "predicted_next_points": "Total Expected Points"
    })
    .sort_values("Total Expected Points", ascending=False)
)

st.dataframe(team_pts, use_container_width=True)


