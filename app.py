
# ============================================================
# 1. IMPORTS & CONFIGURATION
# ============================================================
import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px


# ============================================================
# 2. LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    players = pd.read_csv("players.csv")
    teams = pd.read_csv("teams.csv")

    with zipfile.ZipFile("games.csv.zip") as z:
        games = pd.read_csv(z.open("games.csv"))

    with zipfile.ZipFile("games_details.csv.zip") as z:
        game_details = pd.read_csv(z.open("games_details.csv"))

    with zipfile.ZipFile("ranking.csv.zip") as z:
        ranking = pd.read_csv(z.open("ranking.csv"))

    return players, teams, games, game_details, ranking


players, teams, games, game_details, ranking = load_data()


# ============================================================
# 3. DATA CLEANING
# ============================================================
games['PTS_home'] = pd.to_numeric(games['PTS_home'], errors='coerce')
games['PTS_away'] = pd.to_numeric(games['PTS_away'], errors='coerce')
games.dropna(subset=['PTS_home', 'PTS_away'], inplace=True)

for col in ['PTS', 'AST', 'REB']:
    game_details[col] = pd.to_numeric(game_details[col], errors='coerce')

def convert_minutes(x):
    if isinstance(x, str) and ':' in x:
        m, s = x.split(':')
        return float(m) + float(s) / 60
    return pd.to_numeric(x, errors='coerce')

game_details['MIN'] = game_details['MIN'].apply(convert_minutes)
game_details.dropna(subset=['MIN'], inplace=True)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

# ---- Total points ----
games['TOTAL_POINTS'] = games['PTS_home'] + games['PTS_away']

# ---- Season trend ----
season_trend = games.groupby("SEASON", as_index=False)['TOTAL_POINTS'].mean()

# ---- Player stats ----
player_stats = (
    game_details
    .groupby("PLAYER_ID", as_index=False)
    .agg({"PTS": "mean", "AST": "mean", "REB": "mean", "MIN": "mean"})
    .merge(players, on="PLAYER_ID")
)

# ============================================================
# HOME VS AWAY – LINE TREND (SEPARATE)
# ============================================================
home_away_trend = (
    games
    .groupby("SEASON", as_index=False)
    .agg({
        "PTS_home": "mean",
        "PTS_away": "mean"
    })
)

home_away_long = home_away_trend.melt(
    id_vars="SEASON",
    value_vars=["PTS_home", "PTS_away"],
    var_name="Location",
    value_name="Average Points"
)

home_away_long["Location"] = home_away_long["Location"].map({
    "PTS_home": "Home",
    "PTS_away": "Away"
})


# ============================================================
# CONFERENCE ANALYSIS – LINE TREND (SEPARATE)
# ============================================================
team_conf = ranking[['TEAM_ID', 'CONFERENCE', 'SEASON_ID']].drop_duplicates()

conf_games = games.merge(
    team_conf,
    left_on="HOME_TEAM_ID",
    right_on="TEAM_ID",
    how="left"
)

conf_trend = (
    conf_games
    .groupby(["SEASON", "CONFERENCE"], as_index=False)['PTS_home']
    .mean()
)


# ============================================================
# 5. DASHBOARD LAYOUT
# ============================================================
st.set_page_config(layout="wide")
st.title("🏀 NBA Analytics Dashboard (2004 – Present)")

tabs = st.tabs([
    "📈 Season Trends",
    "🏆 Team Analysis",
    "👤 Player Insights",
    "🏠 Home vs Away (Trend)",
    "⚔ Conference Analysis (Trend)",
    "📊 Rankings"
])


# ============================================================
# TAB 1 – SEASON TREND
# ============================================================
with tabs[0]:
    fig = px.line(
        season_trend,
        x="SEASON",
        y="TOTAL_POINTS",
        markers=True,
        title="Average Total Points per Game (Season-wise)"
    )
    st.plotly_chart(fig, width="stretch")


# ============================================================
# TAB 2 – TEAM ANALYSIS
# ============================================================
with tabs[1]:
    season = st.selectbox("Select Season", sorted(games['SEASON'].unique()))
    team_scores = games[games['SEASON'] == season] \
        .groupby("HOME_TEAM_ID", as_index=False)['PTS_home'].mean()

    fig = px.bar(team_scores, x="HOME_TEAM_ID", y="PTS_home")
    st.plotly_chart(fig, width="stretch")


# ============================================================
# TAB 3 – PLAYER INSIGHTS
# ============================================================
with tabs[2]:
    fig = px.scatter(
        player_stats,
        x="MIN",
        y="PTS",
        size="REB",
        hover_name="PLAYER_NAME",
        title="Player Efficiency (PTS vs Minutes)"
    )
    st.plotly_chart(fig, width="stretch")


# ============================================================
# TAB 4 – HOME VS AWAY (LINE)
# ============================================================
with tabs[3]:
    st.header("Home vs Away Scoring Trend")

    fig = px.line(
        home_away_long,
        x="SEASON",
        y="Average Points",
        color="Location",
        markers=True,
        title="Home vs Away Average Points per Game (Season-wise)"
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("""
    **Insight:**
    - Home teams consistently score more than away teams
    - Gap reduces slightly in recent seasons
    """)


# ============================================================
# TAB 5 – CONFERENCE ANALYSIS (LINE)
# ============================================================
with tabs[4]:
    st.header("Eastern vs Western Conference Trend")

    fig = px.line(
        conf_trend,
        x="SEASON",
        y="PTS_home",
        color="CONFERENCE",
        markers=True,
        title="Conference-wise Average Home Points (Season-wise)"
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("""
    **Insight:**
    - Western Conference dominated scoring for many years
    - Eastern Conference shows gradual resurgence
    """)


# ============================================================
# TAB 6 – RANKINGS
# ============================================================
with tabs[5]:
    ranking_table = ranking[['TEAM_ID', 'SEASON_ID', 'W', 'L', 'W_PCT']] \
        .sort_values(['SEASON_ID', 'W_PCT'], ascending=False)
    st.dataframe(ranking_table)


# ============================================================
# END
# ============================================================

st.success("NBA Analytics Dashboard Loaded Successfully ✔")
