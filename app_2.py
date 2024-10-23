import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado y los objetos de preprocesamiento
loaded_model = joblib.load('xgboost_model_correct.pkl')
loaded_vectorizer = joblib.load('vectorizer.pkl')
loaded_standarizer = joblib.load('standarizer.pkl')

# Configuración de la página
st.set_page_config(page_title="Test Analyzer: AI vs Human", layout="wide")
st.markdown(
    """
    <style>
    .font {
        font-size:35px;
        text-align: center;
        color: #4CAF50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .bgcolor {
        background-color: #f0f2f5;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Función para la introducción
def show_intro():
    st.markdown('<div class="font">Introducción al Aprendizaje Automático</div>', unsafe_allow_html=True)
    st.markdown("""
    El aprendizaje automático (ML) es una rama de la inteligencia artificial que permite a los sistemas aprender y mejorar automáticamente a partir de la experiencia. 
    Uno de los modelos más populares y efectivos en la actualidad es **XGBoost**, que se utiliza para tareas de clasificación y regresión. 
    Este modelo se ha destacado por su alto rendimiento y eficiencia, especialmente en competiciones de ciencia de datos.
    """)

# Función para el uso del modelo
def show_model_usage():
    st.markdown('<div class="font">Uso del Modelo</div>', unsafe_allow_html=True)
    st.write("Este modelo ha sido entrenado con datos de texto para predecir si un texto fue escrito por una IA o por un humano.")
    st.write("El modelo ha demostrado una precisión del 90% en los datos de entrenamiento.")

    # Recolectar el input del usuario usando los widgets de Streamlit
    user_input = st.text_area("Introduce el texto aquí")

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
                st.success("El modelo predice que este texto fue escrito por una **IA**.")
            else:
                st.success("El modelo predice que este texto fue escrito por un **humano**.")
        else:
            st.warning("Por favor, introduce un texto para realizar la predicción.")

# Función para mostrar los resultados en una tabla
def show_results(dataframe):
    st.markdown('<div class="font">Resultados de la Prueba</div>', unsafe_allow_html=True)
    st.dataframe(dataframe)

# Función principal
def main():
    st.sidebar.title("Navegación")
    options = ["Introducción", "Uso del Modelo", "Resultados"]
    choice = st.sidebar.radio("Selecciona una página:", options)

    if choice == "Introducción":
        show_intro()
    elif choice == "Uso del Modelo":
        show_model_usage()
    elif choice == "Resultados":
        # Cargar el DataFrame de resultados, asumiendo que ya lo tienes preparado
        # dataframe_results debe ser un DataFrame que ya hayas creado con tus datos
        dataframe_results = pd.DataFrame({
            "Texto": ["Ejemplo de texto 1", "Ejemplo de texto 2"],
            "Etiqueta Verdadera": [1, 0],
            "Predicción del Modelo": [1, 0],
            "Predicción Correcta": [True, True]
        })
        show_results(dataframe_results)

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
