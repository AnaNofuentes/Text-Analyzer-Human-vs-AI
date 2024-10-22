import streamlit as st
import pandas as pd
import joblib

# Load the trained model
loaded_model = joblib.load('xgboost_model_correct.pkl')
loaded_vectorizer = joblib.load('vectorizer.pkl')
loaded_standarizer = joblib.load('standarizer.pkl')




# App title
st.title("Test Analizer: AI vs Human")

# App description
st.write("This application predicts whether text is written by an AI or human.")

# Function to collect user input using Streamlit widgets
def collect_user_input():
    user_input = st.text_area("Enter text here")
    return user_input
