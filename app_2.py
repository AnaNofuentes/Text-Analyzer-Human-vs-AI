import streamlit as st
import pandas as pd
import joblib
import 
# Configuración de la página
st.set_page_config(page_title="Test Analyzer: AI vs Human", layout="wide")

# Cargar los modelos y objetos preprocesadores
loaded_model = joblib.load('xgboost_model_correct.pkl')  # Cargar modelo entrenado
loaded_vectorizer = joblib.load('vectorizer.pkl')  # Cargar vectorizador
loaded_standarizer = joblib.load('standarizer.pkl')  # Cargar estandarizador

# Función para mostrar los datos de entrenamiento
def show_training_data():
    st.markdown('<div class="font">Datos de Entrenamiento con Predicciones</div>', unsafe_allow_html=True)
    
    # Cargar los datos de entrenamiento (puedes cambiar esto según tus datos)
    df_train = pd.read_csv('complete_without_outliers.csv')  # Este CSV puede ser tu dataset de entrenamiento original
    
    X = df_train["text_cleaned"]
    y = df_train["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # Preprocesar y hacer predicciones
    text_vectorized = loaded_vectorizer.transform(X)
    text_standardized = loaded_standarizer.transform(text_vectorized)
    df_train['Predicción'] = loaded_model.predict(text_standardized)
    
    st.write("Resultados del modelo con los datos de entrenamiento:")
    st.dataframe(df_train)

# Función para mostrar las predicciones con nuevos datos
def show_new_data_predictions():
    st.markdown('<div class="font">Resultados de Predicciones con Nuevos Datos</div>', unsafe_allow_html=True)

    # Cargar los nuevos datos (puedes cambiar esto según tus datos)
    df_new = pd.read_csv('example_reduced.csv')  # Este CSV puede ser el nuevo dataset que quieras probar
    
    # Preprocesar y hacer predicciones
    text_vectorized = loaded_vectorizer.transform(df_new['text_cleaned'])
    text_standardized = loaded_standarizer.transform(text_vectorized)
    df_new['Predicción'] = loaded_model.predict(text_standardized)
    
    st.write("Resultados del modelo con nuevos datos:")
    st.dataframe(df_new)

# Función principal
def main():
    st.sidebar.title("Navegación")
    st.sidebar.markdown('<div class="title">Test Analyzer</div>', unsafe_allow_html=True)
    
    options = ["Datos de Entrenamiento", "Predicciones con Nuevos Datos"]
    choice = st.sidebar.radio("Selecciona una página:", options)

    if choice == "Datos de Entrenamiento":
        show_training_data()
    elif choice == "Predicciones con Nuevos Datos":
        show_new_data_predictions()

# Ejecutar la aplicación
if __name__ == "__main__":
    main()

