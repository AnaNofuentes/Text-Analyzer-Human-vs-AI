import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
        font-size: 36px;
        text-align: center;
        color: #4CAF50;
        font-family: 'Helvetica Neue', sans-serif;
        margin-top: 20px;
    }
    .bgcolor {
        background-color: #f0f2f5;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar {
        background-color: #4CAF50;
        color: white;
    }
    .title {
        text-align: center;
        color: #fff;
        font-size: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Función para la introducción
def show_intro():
    st.markdown('<div class="font">Introducción al Aprendizaje Automático</div>', unsafe_allow_html=True)
    st.markdown("""El **aprendizaje automático** (ML) es una rama de la inteligencia artificial que permite a los sistemas aprender y mejorar automáticamente a partir de la experiencia.""")

# Función para mostrar los datos de entrenamiento
def show_training_data():
    st.markdown('<div class="font">Datos de Entrenamiento</div>', unsafe_allow_html=True)
    # Simular datos de entrenamiento
    training_data = {
        "Texto": ["Texto ejemplo 1", "Texto ejemplo 2", "Texto ejemplo 3"],
        "Etiqueta Verdadera": [1, 0, 1],
        "Predicción": [1, 0, 1],
    }
    df_training = pd.DataFrame(training_data)
    st.write("Estos son los resultados del modelo con los datos de entrenamiento:")
    st.dataframe(df_training)

    # Visualización
    fig, ax = plt.subplots()
    sns.countplot(data=df_training, x='Predicción', hue='Etiqueta Verdadera', palette='viridis', ax=ax)
    ax.set_title('Predicciones del Modelo vs Etiquetas Verdaderas')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Cantidad')
    st.pyplot(fig)

# Función para mostrar el uso del modelo
def show_model_usage():
    st.markdown('<div class="font">Uso del Modelo</div>', unsafe_allow_html=True)
    st.write("Este modelo ha sido entrenado con datos de texto para predecir si un texto fue escrito por una IA o por un humano.")

    user_input = st.text_area("Introduce el texto aquí", height=200)
    if st.button('Predecir'):
        if user_input:
            text_vectorized = loaded_vectorizer.transform([user_input])
            text_standardized = loaded_standarizer.transform(text_vectorized)
            prediction = loaded_model.predict(text_standardized)
            if prediction[0] == 1:
                st.success("El modelo predice que este texto fue escrito por una **IA**.")
            else:
                st.success("El modelo predice que este texto fue escrito por un **humano**.")
        else:
            st.warning("Por favor, introduce un texto para realizar la predicción.")

# Función para mostrar las predicciones de nuevos datos
def show_new_data_predictions(new_data):
    st.markdown('<div class="font">Resultados de Predicciones con Nuevos Datos</div>', unsafe_allow_html=True)

    text_vectorized = loaded_vectorizer.transform(new_data['Texto'])
    text_standardized = loaded_standarizer.transform(text_vectorized)
    new_data['Predicción'] = loaded_model.predict(text_standardized)
    new_data['Predicción Correcta'] = new_data['Predicción'] == new_data['Etiqueta Verdadera']
    
    st.dataframe(new_data)

    fig, ax = plt.subplots()
    sns.countplot(data=new_data, x='Predicción Correcta', palette='viridis', ax=ax)
    ax.set_title('Distribución de Predicciones Correctas')
    ax.set_xlabel('Predicción Correcta (True/False)')
    ax.set_ylabel('Cantidad')
    st.pyplot(fig)

# Función principal
def main():
    st.sidebar.title("Navegación")
    st.sidebar.markdown('<div class="title">Test Analyzer</div>', unsafe_allow_html=True)
    
    options = ["Introducción", "Datos de Entrenamiento", "Uso del Modelo", "Predicciones con Nuevos Datos"]
    choice = st.sidebar.radio("Selecciona una página:", options)

    if choice == "Introducción":
        show_intro()
    elif choice == "Datos de Entrenamiento":
        show_training_data()
    elif choice == "Uso del Modelo":
        show_model_usage()
    elif choice == "Predicciones con Nuevos Datos":
        # Simular nuevos datos para prueba
        new_data = pd.DataFrame({
            "Texto": ["Texto de prueba 1", "Texto de prueba 2"],
            "Etiqueta Verdadera": [1, 0],
        })
        show_new_data_predictions(new_data)

# Ejecutar la aplicación
if __name__ == "__main__":
    main()

