# 📄 Sistema Inteligente de Extracción de Información Académica

Aplicación web de **Procesamiento de Lenguaje Natural (PLN)** para extraer y analizar información relevante de documentos académicos en PDF.

## 🚀 Demo

👉 [Abrir aplicación](https://pc3-pln-academico-mkgnulnwfozgs6evux6pfp.streamlit.app)

## 🧠 Técnicas implementadas

- **NER** – Reconocimiento de Entidades Nombradas con spaCy (`es_core_news_sm`)
- **TF-IDF** – Extracción de keywords con scikit-learn
- **Análisis de Sentimiento** – Polaridad y subjetividad por segmento con TextBlob
- **WordCloud** – Nube de palabras del documento procesado
- **Topic Modeling (LDA)** – Descubrimiento de temas latentes con scikit-learn
- **Extracción de secciones** – Resumen, Abstract, Introducción, Metodología, Resultados y Conclusiones mediante Regex

## 📋 Requisitos

```
streamlit
pdfplumber
spacy==3.7.4
nltk
scikit-learn
textblob
wordcloud
matplotlib
pandas
numpy<2.0
```

## ⚙️ Instalación local

```bash
# 1. Clonar el repositorio
git clone https://github.com/fgeronimoc/pc3-pln-academico.git
cd pc3-pln-academico

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar modelo de spaCy en español
python -m spacy download es_core_news_sm

# 4. Correr la aplicación
streamlit run app.py
```

## 📁 Estructura del proyecto

```
pc3-pln-academico/
├── app.py                          # Aplicación Streamlit
├── requirements.txt                # Dependencias Python
├── runtime.txt                     # Versión de Python (Streamlit Cloud)
├── .python-version                 # Versión de Python (uv)
├── PC3_Notebook_PLN.ipynb          # Notebook de desarrollo
├── Informe_Final_PC3.docx          # Informe del proyecto
├── Presentacion_PC3.pptx           # Presentación del proyecto
└── Dialnet-DesercionUniversitaria.pdf  # Documento de prueba
```

## 👥 Integrantes

Proyecto académico – Curso de Ciencia de Datos / PLN
