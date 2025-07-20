import streamlit as st
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = 'movie_rating_prediction_pipeline.pkl'
DEFAULT_GENRES = ["Action", "Drama", "Comedy", "Thriller", "Romance", "Crime", "Horror", "Sci-Fi", "Mystery", "Adventure", "Fantasy", "Family", "Animation", "Biography", "History", "War", "Music", "Sport", "Documentary", "Musical", "Western"]
DEFAULT_DIRECTORS = ["Director A", "Director B", "Director C", "Director D", "Director E", "Unknown"]

try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure the trained pipeline is saved.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.set_page_config(page_title="Movie Rating Predictor", layout="centered")

st.title("🎬 Movie Rating Predictor")
st.markdown("""
    Enter the details of a movie below to get a predicted IMDb rating.
    The model uses features like year, duration, genre, votes, and director.
""")

st.header("Movie Details")

year = st.slider("Release Year", min_value=1900, max_value=2025, value=2020, step=1)

duration = st.number_input("Duration (minutes)", min_value=10, max_value=300, value=120, step=5)

genre = st.selectbox("Genre", DEFAULT_GENRES)

votes = st.number_input("Number of Votes (e.g., 10000 for 10,000 votes)", min_value=0, max_value=10000000, value=10000, step=1000)

director = st.text_input("Director's Name (e.g., 'Christopher Nolan')", value="Unknown")

if st.button("Predict Rating"):
    input_data = pd.DataFrame([{
        'Year': year,
        'Duration': duration,
        'Genre': genre,
        'Votes': votes,
        'Director': director
    }])

    try:
        predicted_rating = pipeline.predict(input_data)[0]

        st.success(f"### Predicted IMDb Rating: {predicted_rating:.2f} / 10.0")
        st.info("Note: This is a prediction based on the trained model and available features.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please check your input values and try again. Ensure the model file is correct.")

st.markdown("---")
st.caption("Built with Streamlit and Scikit-learn.")
