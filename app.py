import streamlit as st
import pandas as pd


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Football Player Rating Predictor",
    page_icon="⚽",
    layout="wide"
)


# ---------------------------------------------------
# Load Model and Dataset
# ---------------------------------------------------

from xgboost import XGBRegressor

@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model("football_rating_model.json")
    return model

@st.cache_data
def load_data():
    return pd.read_csv("football_players_processed.csv")


model = load_model()
df = load_data()


# ---------------------------------------------------
# Feature List
# ---------------------------------------------------

features = [
    "appearances",
    "matches_started",
    "minutes_played",
    "goals",
    "assists",
    "expected_goals",
    "expected_assists",
    "total_shots",
    "shots_on_target",
    "yellow_cards",
    "red_cards",
    "tackles",
    "interceptions",
    "saves",
    "goals_per_90",
    "assists_per_90",
    "contribution",
    "contribution_per_90",
    "goal_per_appearance",
    "shot_accuracy",
    "xg_difference",
    "xa_difference"
]


# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.title("⚽ Football Player Performance Predictor")

st.write(
    "Predict football player performance ratings using a trained XGBoost Machine Learning model."
)

st.divider()


# ---------------------------------------------------
# Player Search
# ---------------------------------------------------

st.subheader("🔎 Search for a Player")

player_names = sorted(df["name"].dropna().unique())

selected_player = st.selectbox(
    "Select Player",
    player_names
)


# ---------------------------------------------------
# Get Player
# ---------------------------------------------------

player = df[df["name"] == selected_player].iloc[0]


# ---------------------------------------------------
# Player Information
# ---------------------------------------------------

st.subheader("👤 Player Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Player**")
    st.write(player["name"])

with col2:
    st.write("**Position**")
    st.write(player["position"])

with col3:
    st.write("**League**")
    st.write(player["league"])


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

player_features = player[features].values.reshape(1, -1)

predicted_rating = model.predict(player_features)[0]

actual_rating = player["rating"]

difference = abs(actual_rating - predicted_rating)


st.divider()

st.subheader("📊 Performance Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Actual Rating",
        f"{actual_rating:.2f}"
    )

with col2:
    st.metric(
        "Predicted Rating",
        f"{predicted_rating:.2f}"
    )

with col3:
    st.metric(
        "Prediction Difference",
        f"{difference:.2f}"
    )


# ---------------------------------------------------
# Player Statistics
# ---------------------------------------------------

st.divider()

st.subheader("📈 Player Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Goals",
        int(player["goals"])
    )

with col2:
    st.metric(
        "Assists",
        int(player["assists"])
    )

with col3:
    st.metric(
        "Minutes",
        int(player["minutes_played"])
    )

with col4:
    st.metric(
        "Shots",
        int(player["total_shots"])
    )


# ---------------------------------------------------
# Additional Statistics
# ---------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "xG",
        f"{player['expected_goals']:.2f}"
    )

with col2:
    st.metric(
        "xA",
        f"{player['expected_assists']:.2f}"
    )

with col3:
    st.metric(
        "Tackles",
        int(player["tackles"])
    )

with col4:
    st.metric(
        "Interceptions",
        int(player["interceptions"])
    )


# ---------------------------------------------------
# Model Information
# ---------------------------------------------------

st.divider()

st.caption(
    "Model: XGBoost Regressor | "
    "22 performance features | "
    "80/20 train-test split"
)