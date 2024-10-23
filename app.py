import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado y los objetos de preprocesamiento
loaded_model = joblib.load('xgboost_model_correct.pkl')
loaded_vectorizer = joblib.load('vectorizer.pkl')
loaded_standarizer = joblib.load('standarizer.pkl')

# Título de la aplicación
st.title("Test Analyzer: AI vs Human")

# Descripción de la aplicación
st.write("Esta aplicación predice si un texto fue escrito por una IA o por un humano.")

# Función para recolectar el input del usuario usando los widgets de Streamlit
def collect_user_input():
    user_input = st.text_area("Introduce el texto aquí")
    return user_input

# Recoger el input del usuario
user_input = collect_user_input()

# Si el usuario ha introducido un texto
if st.button('Predecir'):
    if user_input:
        # Vectorizar el texto ingresado por el usuario
        text_vectorized = loaded_vectorizer.transform([user_input])
        
        # Estandarizar los datos vectorizados
        text_standardized = loaded_standarizer.transform(text_vectorized)
        
        # Realizar la predicción con el modelo cargado
        prediction = loaded_model.predict(text_standardized)
        
        # Mostrar el resultado de la predicción
        if prediction[0] == 1:
            st.write("El modelo predice que este texto fue escrito por una **IA**.")
        else:
            st.write("El modelo predice que este texto fue escrito por un **humano**.")
    else:
        st.write("Por favor, introduce un texto para realizar la predicción.") 