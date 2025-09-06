import streamlit as st
import joblib
import pandas as pd

model = joblib.load('model.pkl')

st.title("Titanic Survival Prediction")

st.write("Enter passenger details:")

pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=32.2)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

embarked_map = {'C': 1, 'Q': 2, 'S': 3}

input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked_map[embarked]]
})

if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success(f"Survived! Probability: {proba[0][1]:.2f}")
    else:
        st.error(f"Did not survive. Probability: {proba[0][0]:.2f}")