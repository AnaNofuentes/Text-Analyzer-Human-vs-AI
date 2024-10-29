# 🌐 Detección de Texto Humano vs IA

## 📖 Descripción del Proyecto
Este proyecto tiene como objetivo desarrollar un modelo de **machine learning** capaz de distinguir entre textos generados por humanos y aquellos generados por **inteligencia artificial (IA)**. Utilizando técnicas de procesamiento del lenguaje natural, buscamos proporcionar una herramienta útil para identificar el origen de los textos en diversos contextos.

---

## 🛠 Tecnologías Utilizadas
- **Lenguaje de programación**: Python
- **Bibliotecas**:
  - **`scikit-learn`**: Implementación de modelos de machine learning.
  - **`joblib`**: Carga y guardado de modelos.
  - **`pandas`**: Manipulación y análisis de datos.
  - **`streamlit`**: Creación de la interfaz de usuario.
  - **`SentenceTransformers`**: Generación de embeddings de texto.

---

## 🤖 Modelos
El modelo principal utilizado en este proyecto es **XGBoost**, entrenado para clasificar el texto en dos categorías: **humano** e **IA**. Se utilizan **embeddings** para representar los textos, permitiendo que el modelo entienda el contenido semántico.

### 💡 ¿Qué son los Embeddings?
Los embeddings son representaciones numéricas de textos que capturan su significado semántico. A diferencia de la vectorización, que asigna un índice único a cada palabra, los embeddings permiten que palabras y frases con significados similares tengan representaciones numéricas similares.

### ⚖️ Comparativa: Embeddings vs. Vectorización
- **Embeddings**: 
  - Capturan el contexto y significado del texto.
  - Proporcionan vectores densos de menor dimensión que reflejan la semántica.
  
- **Vectorización**: 
  - Asigna un número único a cada palabra en el vocabulario.
  - No captura el significado contextual, resultando en vectores dispersos y de mayor dimensión.

---

## 🌟 Interfaz de Usuario
La aplicación se desarrolla con **Streamlit**, ofreciendo una interfaz intuitiva donde los usuarios pueden ingresar texto y obtener predicciones sobre si fue escrito por un humano o una IA. La aplicación incluye:
- 📝 Un área de texto para ingresar el texto a analizar.
- 🔮 Un botón para realizar la predicción.
- 🎨 Resultados presentados con colores y estilos visuales atractivos.

---


## 💻 Ejecución
Para ejecutar la aplicación, utiliza el siguiente comando:
streamlit run app.py
## Presentación

[enlace a la presentacion](https://www.canva.com/design/DAGUgP3VxLQ/8SGYaYvDgH6j3SBlBrlsjw/edit?utm_content=DAGUgP3VxLQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) 

🤝 Contribuciones
Si deseas contribuir a este proyecto, no dudes en abrir un issue o enviar un pull request. Todas las contribuciones son bienvenidas.

📬 Contacto
Para preguntas o comentarios, puedes contactar a [Ana Nofuentes Solano](https://www.linkedin.com/in/ana-nofuentes-solano-654026a3/).
