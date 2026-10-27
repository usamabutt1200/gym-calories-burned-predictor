import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# ==========================================
# 1. APPLICATION CONFIGURATION & MODEL LOAD
# ==========================================
st.set_page_config(page_title="Gym Calories Burned Predictor", layout="centered", page_icon="🏋️‍♂️")

@st.cache_resource
def load_gym_model():
    # Pehle pickle format check karein, backup par joblib check karein
    try:
        with open('gym_calories_model.pkl', 'rb') as file:
            return pickle.load(file)
    except Exception:
        return joblib.load('gym_calories_model.pkl')

try:
    model_pipeline = load_gym_model()
    st.success("🤖 Gym Prediction Model loaded successfully!")
except Exception as e:
    st.error("❌ Error loading model: Ensure 'gym_calories_model.pkl' is uploaded in the same GitHub repository.")
    st.stop()

# ==========================================
# 2. WEB APPLICATION USER INTERFACE (UI)
# ==========================================
st.title("🏋️‍♂️ Gym Exercise Calories Predictor")
st.markdown("Enter your workout details below to estimate total calories burned using the trained machine learning model.")

st.subheader("📋 Member & Workout Metrics")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age:", min_value=12, max_value=100, value=25)
    gender = st.selectbox("Gender:", options=['Male', 'Female'])
    weight = st.number_input("Weight (kg):", min_value=30, max_value=200, value=70)
    heart_rate = st.number_input("Average Heart Rate during workout (BPM):", min_value=60, max_value=220, value=130)

with col2:
    session_duration = st.number_input("Session Duration (Minutes):", min_value=5, max_value=300, value=45)
    workout_type = st.selectbox("Workout Type:", options=['Cardio', 'Strength', 'HIIT', 'Yoga'])
    experience_level = st.selectbox("Experience Level:", options=['Beginner', 'Intermediate', 'Advanced'])
    avg_met_score = st.number_input("Average MET Score (Intensity):", min_value=1.0, max_value=20.0, value=6.5)

# ==========================================
# 3. ROBUST PREDICTION LOGIC WITH FALLBACK
# ==========================================
if st.button("🔥 Calculate Calories Burned", type="primary"):
    
    # Base user input format
    user_inputs = {
        'Age': [age],
        'Gender': [gender],
        'Weight_kg': [weight],
        'Heart_Rate': [heart_rate],
        'Session_Duration_Min': [session_duration],
        'Workout_Type': [workout_type],
        'Experience_Level': [experience_level],
        'Avg_Met_Score': [avg_met_score]
    }
    
    # Model pipeline ke actual features dynamically read karein
    try:
        original_features = model_pipeline.named_steps['preprocessor'].feature_names_in_
    except AttributeError:
        original_features = ['Age', 'Gender', 'Weight_kg', 'Heart_Rate', 'Session_Duration_Min', 'Workout_Type', 'Experience_Level', 'Avg_Met_Score']

    # Auto-adjust columns to match training schema target tarteeb
    full_input_dict = {}
    for col in original_features:
        if col in user_inputs:
            full_input_dict[col] = user_inputs[col]
        else:
            full_input_dict[col] = [np.nan]  # Missing features handling

    input_df = pd.DataFrame(full_input_dict)[list(original_features)]
    
    try:
        # Run prediction process
        predicted_calories = model_pipeline.predict(input_df)
        final_calories = float(predicted_calories[0])
        
        st.markdown("---")
        st.balloons()  # Celebration effect on success 🎈
        st.metric(label="📊 Estimated Calories Burned", value=f"{final_calories:,.1f} kcal")
        st.info("💡 Note: This dynamic estimation is calculated by the pre-trained XGBoost Model Pipeline based on your individual exertion vectors.")
        
    except Exception as prediction_error:
        st.error(f"⚠️ Prediction Matrix Error: {str(prediction_error)}")
