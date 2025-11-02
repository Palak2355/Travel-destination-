import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load model assets
try:
    model = load('best_model.joblib')
    scaler = load('scaler.joblib')
    feature_columns = load('feature_columns.joblib')
    label_encoder = load('label_encoder.joblib')
except FileNotFoundError:
    st.error("FATAL ERROR: Model assets not found! Ensure all .joblib files are in the current working directory.")
    st.stop()

# Prediction function
def predict_destination(age, gender, income_level, travel_companion, activity_level, budget):
    input_data = pd.DataFrame({
        'Age': [age],
        'Activity_Level': [activity_level],
        'Gender': [gender],
        'Income_Level': [income_level],
        'Travel_Companion': [travel_companion],
        'Budget': [budget]
    })

    numerical_features = ['Age', 'Activity_Level']
    input_encoded = pd.get_dummies(
        input_data,
        columns=['Gender', 'Income_Level', 'Travel_Companion', 'Budget'],
        drop_first=True
    )

    input_encoded[numerical_features] = scaler.transform(input_encoded[numerical_features])
    final_input = pd.DataFrame(0, index=[0], columns=feature_columns)
    for col in input_encoded.columns:
        if col in final_input.columns:
            final_input[col] = input_encoded[col].iloc[0]

    try:
        prediction_encoded = model.predict(final_input)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
        probabilities = model.predict_proba(final_input)[0]
        confidence = probabilities[prediction_encoded] * 100

        emoji_map = {
            "Adventure/Nature": "🏔️ Adventure/Nature",
            "Relaxation/Beach": "🏖️ Relaxation/Beach",
            "Cultural/Historical": "🏛️ Cultural/Historical",
            "Party/Nightlife": "🎉 Party/Nightlife"
        }

        return prediction_label, confidence, emoji_map.get(prediction_label, '🌍 Unknown')
    except Exception as e:
        return None, None, f"Prediction Error: {e}"

# Streamlit UI
st.set_page_config(page_title="Travel Destination Recommender", page_icon="🌍", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #fdf6ec;
        color: #3f2e1e;
    }
    .stButton>button {
        background-color: #d97706;
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #b45309;
    }
    .result-box {
        background-color: #fff4e6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #d97706;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌍 Personalized Travel Destination Recommender")
st.markdown("*Powered by Tuned Machine Learning Model*")
st.markdown("✨ Tell us about your preferences and we'll recommend the perfect travel destination!")

# Inputs
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", min_value=18, max_value=75, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])
with col2:
    activity_level = st.slider("Activity Level (1=Relaxing, 5=Intense)", min_value=1, max_value=5, value=4)
    travel_companion = st.selectbox("Travel Companion", ["Solo", "Family", "Group"])
    budget = st.selectbox("Budget", ["Economical", "Mid-range", "Luxury"])

if st.button("Get Recommendation"):
    prediction, confidence, emoji = predict_destination(age, gender, income_level, travel_companion, activity_level, budget)
    if prediction:
        st.markdown(f"""
        <div class="result-box">
            <h3>Recommended Destination:</h3>
            <h1>{emoji}</h1>
            <p><strong>Model Confidence: {confidence:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(emoji)
