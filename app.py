import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model and scaler
model, scaler = joblib.load('model.pkl')

# Streamlit App Title
st.title('📚 Student Failure Predictor')

# Sidebar Inputs
st.sidebar.header('🎯 Enter Student Data')
study_time = st.sidebar.slider('📖 Study Time (hours/week)', 0.0, 20.0, 10.0)
attendance = st.sidebar.slider('🏫 Attendance (days/week)', 0.0, 5.0, 4.0)

# Prepare input for prediction (apply feature scaling)
input_features = np.array([[study_time, attendance]])
scaled_features = scaler.transform(input_features)

# Prediction
if st.sidebar.button('🔮 Predict'):
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)

    # Display results
    st.subheader('🎯 Prediction Result')
    result_text = "❌ Will Fail" if prediction[0] == 1 else "✅ Will Pass"
    st.markdown(f'### {result_text}')
    
    st.write(f'**📊 Probability of Passing:** {probability[0][0]:.2f}')
    st.write(f'**📊 Probability of Failing:** {probability[0][1]:.2f}')

# Optional Data Visualization
# Optional Data Visualization
if st.checkbox('📊 Show Data Visualization'):
    st.subheader('📈 Study Time vs. Attendance with Predictions')

    # Load dataset
    df = pd.read_csv('data.csv')

    # Prepare features for model prediction
    X = df[['study_time', 'attendance']]
    X_scaled = scaler.transform(X)  # Apply same scaling as training

    # Make predictions
    df['prediction'] = model.predict(X_scaled)

    # Map predictions to labels
    df['prediction'] = df['prediction'].map({0: "Pass", 1: "Fail"})

    # Plot scatter plot based on model predictions
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='study_time', y='attendance', hue='prediction', data=df, ax=ax, palette={"Pass": "green", "Fail": "red"})
    
    ax.set_title('Pass/Fail Distribution Based on Model Predictions')
    ax.set_xlabel('Study Time (hours/week)')
    ax.set_ylabel('Attendance (days/week)')
    st.pyplot(fig)

# Footer
st.markdown("👨‍🎓 Created by **Naveen Kumar**")
