import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Football Player Performance Predictor",
    page_icon="⚽",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #777;
    margin-bottom: 25px;
}

.rating-card {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    background-color: #f5f5f5;
}

.rating-number {
    font-size: 34px;
    font-weight: bold;
}

.section-title {
    font-size: 25px;
    font-weight: 600;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = XGBRegressor()

    model.load_model("football_rating_model.json")

    return model


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv("football_players_processed.csv")


model = load_model()
df = load_data()


# ============================================================
# MODEL FEATURES
# ============================================================

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


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_

}).sort_values(

    "Importance",
    ascending=False

)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚽ Football ML")

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigate",

    [
        "Player Predictor",
        "Top Rated Players",
        "League Analysis",
        "Model Information"
    ]

)

st.sidebar.markdown("---")

st.sidebar.info(

    "Machine Learning model: "
    "XGBoost Regressor\n\n"
    "Dataset: 8 European leagues"

)


# ============================================================
# HEADER
# ============================================================

st.markdown(

    '<div class="main-title">⚽ Football Player Performance Predictor</div>',

    unsafe_allow_html=True

)

st.markdown(

    '<div class="subtitle">'
    "Predict and analyze football player performance using Machine Learning"
    "</div>",

    unsafe_allow_html=True

)


# ============================================================
# PLAYER PREDICTOR
# ============================================================

if page == "Player Predictor":

    st.header("🔎 Player Search")

    # Search box
    search = st.text_input(

        "Search player by name",

        placeholder="Example: Salah, Mbappe, Kane..."

    )


    # Filter players
    if search:

        filtered_players = df[
            df["name"]
            .astype(str)
            .str.contains(search, case=False, na=False)
        ]

    else:

        filtered_players = df


    if len(filtered_players) == 0:

        st.warning("No player found.")

        st.stop()


    player_options = (

        filtered_players["name"]
        .dropna()
        .unique()
        .tolist()

    )


    selected_player = st.selectbox(

        "Select Player",

        sorted(player_options)

    )


    # Select player row
    player_rows = df[
        df["name"] == selected_player
    ]


    # If duplicate names exist, select first
    player = player_rows.iloc[0]


    # ========================================================
    # PLAYER INFORMATION
    # ========================================================

    st.markdown("---")

    st.subheader("👤 Player Information")


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(

            "Player",

            player["name"]

        )


    with col2:

        st.metric(

            "Position",

            player["position"]

        )


    with col3:

        st.metric(

            "League",

            player["league"]

        )


    with col4:

        if "market_value" in player.index:

            st.metric(

                "Market Value",

                f"{player['market_value']:,.0f}"

            )


    # ========================================================
    # PREDICTION
    # ========================================================

    player_features = (

        player[features]
        .astype(float)
        .values
        .reshape(1, -1)

    )


    predicted_rating = model.predict(

        player_features

    )[0]


    actual_rating = float(

        player["rating"]

    )


    difference = abs(

        actual_rating - predicted_rating

    )


    # ========================================================
    # RATING DISPLAY
    # ========================================================

    st.markdown("---")

    st.subheader("🎯 Performance Rating")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(

            "Actual Rating",

            f"{actual_rating:.2f}"

        )


    with col2:

        st.metric(

            "Predicted Rating",

            f"{predicted_rating:.2f}",

            delta=f"{predicted_rating - actual_rating:.2f}"

        )


    with col3:

        st.metric(

            "Prediction Difference",

            f"{difference:.2f}"

        )


    # Rating bar
    st.write("Rating Scale")


    rating_percentage = min(

        max((predicted_rating / 10) * 100, 0),

        100

    )


    st.progress(

        int(rating_percentage)

    )


    # ========================================================
    # PLAYER STATISTICS
    # ========================================================

    st.markdown("---")

    st.subheader("📊 Player Statistics")


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


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(

            "Expected Goals",

            f"{player['expected_goals']:.2f}"

        )


    with col2:

        st.metric(

            "Expected Assists",

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


    # ========================================================
    # CHART
    # ========================================================

    st.markdown("---")

    st.subheader("📈 Performance Statistics")


    chart_data = pd.DataFrame({

        "Statistic": [

            "Goals",
            "Assists",
            "Expected Goals",
            "Expected Assists",
            "Tackles",
            "Interceptions"

        ],

        "Value": [

            player["goals"],
            player["assists"],
            player["expected_goals"],
            player["expected_assists"],
            player["tackles"],
            player["interceptions"]

        ]

    })


    st.bar_chart(

        chart_data.set_index("Statistic")

    )


    # ========================================================
    # MODEL EXPLANATION
    # ========================================================

    st.markdown("---")

    st.subheader("🧠 Important Model Features")


    top_features = feature_importance.head(8).copy()


    top_features = top_features.rename(

        columns={

            "Feature": "Feature",

            "Importance": "Importance"

        }

    )


    st.bar_chart(

        top_features.set_index("Feature")

    )


# ============================================================
# TOP RATED PLAYERS
# ============================================================

elif page == "Top Rated Players":

    st.header("🏆 Top Rated Players")


    top_players = (

        df.sort_values(

            "rating",

            ascending=False

        )

        .head(10)

        [[

            "name",
            "position",
            "league",
            "rating"

        ]]

        .copy()

    )


    top_players["rating"] = top_players["rating"].round(2)


    st.dataframe(

        top_players,

        use_container_width=True,

        hide_index=True

    )


    st.subheader("📊 Rating Distribution")


    st.bar_chart(

        top_players.set_index("name")["rating"]

    )


# ============================================================
# LEAGUE ANALYSIS
# ============================================================

elif page == "League Analysis":

    st.header("🌍 League Analysis")


    league_stats = (

        df.groupby("league")

        .agg(

            Players=("player_id", "count"),

            Average_Rating=("rating", "mean"),

            Highest_Rating=("rating", "max")

        )

        .reset_index()

    )


    league_stats["Average_Rating"] = (

        league_stats["Average_Rating"]

        .round(2)

    )


    league_stats["Highest_Rating"] = (

        league_stats["Highest_Rating"]

        .round(2)

    )


    st.dataframe(

        league_stats,

        use_container_width=True,

        hide_index=True

    )


    st.subheader("Average Rating by League")


    st.bar_chart(

        league_stats.set_index("league")["Average_Rating"]

    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.header("🧠 Machine Learning Model")


    st.write(

        "The application uses a trained XGBoost Regressor "
        "to predict football player performance ratings "
        "from player statistics."

    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(

            "Model",

            "XGBoost"

        )


    with col2:

        st.metric(

            "Features",

            "22"

        )


    with col3:

        st.metric(

            "Train/Test Split",

            "80 / 20"

        )


    with col4:

        st.metric(

            "Dataset Rows",

            f"{len(df):,}"

        )


    st.markdown("---")


    st.subheader("📏 Model Evaluation")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(

            "MAE",

            "0.1248"

        )


    with col2:

        st.metric(

            "RMSE",

            "0.1767"

        )


    with col3:

        st.metric(

            "R²",

            "0.4974"

        )


    st.markdown("---")


    st.subheader("🔬 Feature Importance")


    st.dataframe(

        feature_importance,

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(

    "Football Player Performance Rating Prediction | "
    "Machine Learning Mini Project by Prem Patil[me :) ] | "
    "XGBoost Regressor"

)