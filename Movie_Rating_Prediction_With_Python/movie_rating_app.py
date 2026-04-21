import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("movie_rating_prediction_pipeline.pkl")
METADATA_PATH = Path("model_metadata.json")
FALLBACK_GENRES = [
    "Drama",
    "Action",
    "Comedy",
    "Crime",
    "Romance",
    "Horror",
    "Thriller",
    "Adventure",
    "Documentary",
]


@st.cache_resource(show_spinner=False)
def load_pipeline() -> object:
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


st.set_page_config(page_title="Movie Rating Predictor", layout="centered")
st.title("Movie Rating Predictor")
st.caption("Fast local inference with a pre-trained pipeline for Streamlit Community Cloud")

if not MODEL_PATH.exists():
    st.error(
        "Model file missing: movie_rating_prediction_pipeline.pkl. "
        "Run the notebook once to generate it before deployment."
    )
    st.stop()

try:
    pipeline = load_pipeline()
except Exception as err:
    st.error(f"Failed to load pipeline: {err}")
    st.stop()

metadata = load_metadata()
year_range = metadata.get("year_range", [1930, 2025])
duration_range = metadata.get("duration_range", [60, 240])
votes_range = metadata.get("votes_range", [0, 1_000_000])
top_genres = metadata.get("top_genres", FALLBACK_GENRES)
top_directors = metadata.get("top_directors", ["Unknown"])
metrics = metadata.get("metrics", [])

year_min, year_max = int(year_range[0]), int(year_range[1])
duration_min, duration_max = int(duration_range[0]), int(duration_range[1])
votes_min, votes_max = int(votes_range[0]), int(votes_range[1])

st.markdown("Predict an IMDb rating using movie metadata (year, duration, genre, votes, and director).")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        year = st.slider(
            "Release Year",
            min_value=year_min,
            max_value=year_max,
            value=clamp(2018, year_min, year_max),
            step=1,
        )
        duration = st.slider(
            "Duration (minutes)",
            min_value=duration_min,
            max_value=duration_max,
            value=clamp(130, duration_min, duration_max),
            step=1,
        )
    with col2:
        votes = st.slider(
            "Votes",
            min_value=votes_min,
            max_value=votes_max,
            value=clamp(5000, votes_min, votes_max),
            step=max(1, (votes_max - votes_min) // 1000),
        )
        genre = st.selectbox("Main Genre", options=top_genres, index=0)

    director = st.selectbox("Director (Top Known)", options=top_directors, index=0)
    custom_director = st.text_input("Or type a different director", value="")

    submitted = st.form_submit_button("Predict IMDb Rating")

if submitted:
    selected_director = custom_director.strip() if custom_director.strip() else director
    input_data = pd.DataFrame(
        [
            {
                "Year": int(year),
                "Duration": int(duration),
                "Genre": str(genre),
                "Votes": int(votes),
                "Director": str(selected_director),
            }
        ]
    )

    try:
        predicted_rating = float(pipeline.predict(input_data)[0])
        predicted_rating = clamp(int(round(predicted_rating * 100)), 0, 1000) / 100
        st.success(f"Predicted IMDb Rating: {predicted_rating:.2f} / 10")
    except Exception as err:
        st.error(f"Prediction failed: {err}")

if metrics:
    st.subheader("Model Performance Snapshot")
    metrics_df = pd.DataFrame(metrics)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Built with Streamlit, scikit-learn, and XGBoost")
