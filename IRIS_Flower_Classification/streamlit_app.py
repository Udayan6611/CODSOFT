import streamlit as st
import pandas as pd
import joblib
import os

st.title("🌸 Iris Flower Predictor (CODSOFT)")

st.markdown("Adjust sliders below and click to predict the Iris species.")

# Sliders for input
sl = st.slider("Sepal Length", 4.0, 8.0, 5.8)
sw = st.slider("Sepal Width", 2.0, 4.5, 3.0)
pl = st.slider("Petal Length", 1.0, 7.0, 4.0)
pw = st.slider("Petal Width", 0.1, 2.5, 1.2)

# Load model and scaler
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "iris_random_forest_model.pkl")

model = joblib.load(model_path)

input_data = pd.DataFrame([[sl, sw, pl, pw]], columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

pred = model.predict(input_data)
species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

st.success(f"🌼 Predicted Species: **{species[pred[0]]}**")