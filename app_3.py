import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado y los objetos de preprocesamiento
loaded_model = joblib.load('xgboost_model_correct.pkl')
loaded_vectorizer = joblib.load('vectorizer.pkl')
loaded_standarizer = joblib.load('standarizer.pkl')

# Estilo CSS para el color de fondo pastel
st.markdown(
    """
    <style>
    /* Fondo de toda la página */
    .stApp {
        background-color: #F3E5F5;  /* Lila claro */
    }
    /* Estilo del placeholder dentro del área de texto */
    ::placeholder {
        color: #6A1B9A; /* Color morado para el placeholder */
        opacity: 1; /* Asegurarse de que el placeholder sea completamente visible */
    }
    </style>
    """, unsafe_allow_html=True
)

# Título de la aplicación con estilo y colores armónicos
st.markdown("<h1 style='text-align: center; color: #6A1B9A;'>Test Analyzer: AI vs Human</h1>", unsafe_allow_html=True)

# Descripción de la aplicación con colores suaves
st.markdown("<p style='text-align: center; color: #9C27B0;'>🧠 Esta aplicación predice si un texto fue escrito por una <b>IA</b> o por un <b>humano</b>.</p>", unsafe_allow_html=True)

# Función para recolectar el input del usuario
def collect_user_input():
    # Estilo CSS para el área de texto
    st.markdown(
        """
        <style>
        /* Estilo del área de entrada de texto */
        textarea {
            background-color: #FFF3E0;  /* Color melocotón claro */
            color: #4A148C;  /* Texto en morado oscuro */
            border: 2px solid #9C27B0; /* Borde en un tono más oscuro */
            border-radius: 5px; /* Bordes redondeados */
        }
        </style>
        """, unsafe_allow_html=True
    )
    return st.text_area("✍️ Introduce el texto aquí:", height=150)

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
