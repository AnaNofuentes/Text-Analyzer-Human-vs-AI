import streamlit as st
import pandas as pd
import gensim
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Inicializamos PorterStemmer
ps = PorterStemmer()

# Cargamos las stopwords en inglés
stop_words = set(stopwords.words('english'))

# Función para limpiar y procesar el texto
def limpiar_texto(text):
    # Verificar si el texto no es nulo o vacío
    if isinstance(text, str):
        # Eliminar saltos de línea y múltiples espacios
        text = re.sub(r'\s+', ' ', text)  # Reemplaza saltos de línea y tabs por un espacio
        text = text.strip()  # Elimina espacios en blanco iniciales y finales

        # Convertir a palabras en minúsculas y filtrar stopwords
        words = [
            ps.stem(word) for word in gensim.utils.simple_preprocess(text)
            if word not in gensim.parsing.preprocessing.STOPWORDS and word not in stop_words
        ]
        return ' '.join(words)
    else:
        return None  # Devuelve None si el texto es inválido

# Función para aplicar la limpieza al DataFrame
def procesar_dataframe(df):
    # Verifica si la columna 'text' existe en el DataFrame
    if 'text' not in df.columns:
        st.error("El DataFrame no contiene una columna 'text'.")
        return None
    
    # Aplicar la función de limpieza al DataFrame
    df['text_cleaned'] = df['text'].apply(limpiar_texto)
    
    # Eliminar filas donde el texto limpio es None o vacío
    df_cleaned = df[df['text_cleaned'].notnull() & (df['text_cleaned'] != '')]
    
    # Reiniciar el índice del DataFrame después de eliminar las filas
    df_cleaned.reset_index(drop=True, inplace=True)
    
    return df_cleaned

# Integrar en Streamlit
def main():
    st.title("Procesador de Texto y Limpieza de Datos")
    
    # Subir el archivo CSV para procesar
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        # Leer el archivo en un DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("Datos originales:")
        st.write(df.head())
        
        # Aplicar la función de procesamiento
        df_cleaned = procesar_dataframe(df)
        
        if df_cleaned is not None:
            st.write("Datos procesados:")
            st.write(df_cleaned.head())

            # Permitir la descarga del DataFrame procesado
            csv = df_cleaned.to_csv(index=False).encode('utf-8')
            st.download_button(label="Descargar CSV Limpio", data=csv, file_name='datos_procesados.csv', mime='text/csv')

if __name__ == '__main__':
    main()

