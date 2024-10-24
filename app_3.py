import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado y los objetos de preprocesamiento
loaded_model = joblib.load('xgboost_model_correct.pkl')
loaded_vectorizer = joblib.load('vectorizer.pkl')
loaded_standarizer = joblib.load('standarizer.pkl')

# Título de la aplicación
st.markdown(
    "<h1 style='text-align: center; color: #6A1B9A;'>Test Analyzer: AI vs Human</h1>", 
    unsafe_allow_html=True
)

# Mostrar la imagen debajo del título
st.image("image.png", use_column_width=False, width=300)  

# Descripción de la aplicación
st.markdown(
    "<p style='text-align: center; color: #9C27B0; font-size: 18px;'>🧠 Esta aplicación predice si un texto fue escrito por una <b>IA</b> o por un <b>humano</b>.</p>", 
    unsafe_allow_html=True
)

# Función para recolectar el input del usuario
def collect_user_input():
    st.markdown(
        """
        <div style='color: #4A148C; font-size: 16px; font-weight: bold;'>✍️ Introduce el texto aquí:</div>
        """, unsafe_allow_html=True
    )
    return st.text_area("", height=150)

# Recoger el input del usuario
user_input = collect_user_input()

# Si el usuario ha introducido un texto
if st.button('🔮 Predecir 🔮'):
    if user_input:
        # Vectorizar el texto ingresado por el usuario
        text_vectorized = loaded_vectorizer.transform([user_input])
        
        # Estandarizar los datos vectorizados
        text_standardized = loaded_standarizer.transform(text_vectorized)
        
        # Realizar la predicción con el modelo cargado
        prediction = loaded_model.predict(text_standardized)
        
        # Mostrar el resultado de la predicción con colores armoniosos
        if prediction[0] == 1:
            st.markdown(
                "<div style='background-color:#D4E157; padding: 10px; border-radius: 10px;'>"
                "<h2 style='text-align: center; color: #4E342E;'>🤖 El modelo predice que este texto fue escrito por una <b>IA</b>.</h2>"
                "</div>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background-color:#80CBC4; padding: 10px; border-radius: 10px;'>"
                "<h2 style='text-align: center; color: #004D40;'>👤 El modelo predice que este texto fue escrito por un <b>humano</b>.</h2>"
                "</div>", 
                unsafe_allow_html=True
            )
    else:
        # Cambiar el color del mensaje para que sea claramente visible
        st.markdown("<p style='text-align: center; color: #E53935;'>⚠️ Por favor, introduce un texto para realizar la predicción.</p>", unsafe_allow_html=True)
