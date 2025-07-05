import streamlit as st
import joblib
import pandas as pd

st.title("🚢 Titanic Survival Prediction App")

pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 50.0)

sex = st.selectbox("Sex", ["Male", "Female"])
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

sex_male = 1 if sex == "Male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

input_df = pd.DataFrame([{
    'PassengerId': 0,         
    'Pclass': pclass,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Sex_male': sex_male,
    'Embarked_Q': embarked_Q,
    'Embarked_S': embarked_S
}])

model = joblib.load("titanic_random_forest_model.pkl")

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("🎉 The passenger is likely to survive!")
    else:
        st.error("💀 The passenger is unlikely to survive.")
