import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    # Cache this function so the app is fast; the model only loads once.
    model = joblib.load("iris_xgboost_pipeline.pkl")
    encoder = joblib.load("iris_label_encoder.pkl")
    return model, encoder

model, encoder = load_artifacts()

st.title("Iris Species Predictor 🌿")
st.markdown("Use the sidebar to input flower measurements.")

st.sidebar.header("Input Features")

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.2)
    
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    
    # The model was trained on a DataFrame, so we create one here.
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader("Current Input Features")
st.write(input_df)

if st.sidebar.button("Predict Species"):
    
    # The pipeline automatically handles all steps (like scaling and predicting).
    prediction_encoded = model.predict(input_df)
    
    # Convert the numeric prediction (e.g., 0) back to the text label (e.g., "Setosa").
    prediction_label = encoder.inverse_transform(prediction_encoded)
    
    st.subheader("Prediction")
    st.success(f"**The predicted species is: `{prediction_label[0]}`**")

else:
    st.info("Click the 'Predict Species' button to see the result.")