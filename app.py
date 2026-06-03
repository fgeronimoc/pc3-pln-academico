import re
import io
import pandas as pd
import streamlit as st
import pdfplumber
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ──────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Extractor Académico PLN",
    page_icon="📄",
    layout="wide"
)

# ──────────────────────────────────────────────
# Recursos NLTK
# ──────────────────────────────────────────────
@st.cache_resource
def cargar_stopwords():
    try:
        return set(stopwords.words("spanish"))
    except LookupError:
        nltk.download("stopwords")
        nltk.download("punkt")
        return set(stopwords.words("spanish"))

stop_words = cargar_stopwords()

# ──────────────────────────────────────────────
# Modelo spaCy
# ──────────────────────────────────────────────
@st.cache_resource
def cargar_modelo_spacy():
    return spacy.load("es_core_news_sm")

nlp = cargar_modelo_spacy()

# ══════════════════════════════════════════════
# FUNCIONES DE PROCESAMIENTO
# ══════════════════════════════════════════════

def extraer_texto_pdf(archivo_pdf):
    """Extrae texto de todas las páginas del PDF."""
    texto_total = ""
    with pdfplumber.open(archivo_pdf) as pdf:
        for i, pagina in enumerate(pdf.pages):
            texto = pagina.extract_text()
            if texto:
                texto_total += f"\n--- Página {i+1} ---\n{texto}\n"
    return texto_total


def limpiar_texto(texto):
    """Limpia y lematiza el texto eliminando stopwords y caracteres no deseados."""
    texto = texto.lower().replace("\n", " ")
    texto = re.sub(r"\d+", "", texto)
    texto = re.sub(r"[^\w\s]", "", texto)
    doc = nlp(texto)
    tokens = [token.lemma_ for token in doc
              if token.text not in stop_words and not token.is_space and len(token.text) > 2]
    return " ".join(tokens)


def extraer_entidades(texto):
    """Reconocimiento de Entidades Nombradas (NER) con spaCy."""
    doc = nlp(texto)
    entidades = [{"Entidad": ent.text, "Tipo": ent.label_, "Inicio": ent.start_char, "Fin": ent.end_char}
                 for ent in doc.ents]
    return pd.DataFrame(entidades)


def extraer_keywords_tfidf(texto, n=25):
    """Extrae términos más relevantes mediante TF-IDF."""
    vectorizer = TfidfVectorizer(
        max_features=1500,
        stop_words=list(stop_words),
        ngram_range=(1, 2)
    )
    matriz = vectorizer.fit_transform([texto])
    palabras = vectorizer.get_feature_names_out()
    puntajes = matriz.toarray()[0]
    df = pd.DataFrame({"Término": palabras, "Puntaje TF-IDF": puntajes})
    return df.sort_values("Puntaje TF-IDF", ascending=False).head(n).reset_index(drop=True)


def analisis_sentimiento(texto):
    """
    Análisis de sentimiento por párrafos usando TextBlob.
    Retorna DataFrame con polaridad y subjetividad por segmento.
    """
    parrafos = [p.strip() for p in texto.split("\n") if len(p.strip()) > 60]
    resultados = []
    for p in parrafos[:30]:
        blob = TextBlob(p)
        pol = blob.sentiment.polarity
        subj = blob.sentiment.subjectivity
        etiqueta = "Positivo" if pol > 0.05 else ("Negativo" if pol < -0.05 else "Neutro")
        resultados.append({
            "Segmento": p[:100] + "...",
            "Polaridad": round(pol, 4),
            "Subjetividad": round(subj, 4),
            "Sentimiento": etiqueta
        })
    return pd.DataFrame(resultados)


def topic_modeling_lda(texto_limpio, n_topics=4, n_palabras=8):
    """
    Topic Modeling con LDA para descubrir temas latentes en el documento.
    Retorna lista de temas con sus palabras principales.
    """
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words=list(stop_words),
        ngram_range=(1, 1),
        min_df=1
    )
    try:
        matriz = vectorizer.fit_transform([texto_limpio])
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20
        )
        lda.fit(matriz)
        palabras_vocab = vectorizer.get_feature_names_out()
        temas = []
        for i, topic in enumerate(lda.components_):
            top_idx = topic.argsort()[-n_palabras:][::-1]
            top_palabras = [palabras_vocab[j] for j in top_idx]
            temas.append({"Tema": f"Tema {i+1}", "Palabras clave": ", ".join(top_palabras)})
        return pd.DataFrame(temas)
    except Exception:
        return pd.DataFrame({"Tema": ["Sin resultados"], "Palabras clave": ["Texto insuficiente"]})


# ── AQUÍ ESTÁ LA FUNCIÓN CORREGIDA PARA EXTRAER LAS SECCIONES BIEN ──
def extraer_seccion(texto, inicio, finales):
    """Extrae una sección del documento usando expresiones regulares de forma robusta."""
    patron_finales = "|".join(finales)
    
    # Permite espacios, puntos o dos puntos después del título
    patron = rf'(?:{inicio})[\s\.\:]*(.*?)(?:{patron_finales})'
    
    # Usamos finditer para buscar en todo el documento y esquivar el índice
    coincidencias = re.finditer(patron, texto, flags=re.IGNORECASE | re.DOTALL)
    
    mejor_coincidencia = "No encontrado"
    max_len = 0
    
    # Nos quedamos con la coincidencia más larga (el texto real de la sección)
    for match in coincidencias:
        contenido = match.group(1).strip()
        if len(contenido) > max_len:
            max_len = len(contenido)
            mejor_coincidencia = contenido
            
    # Retornamos solo si la sección capturada tiene un tamaño lógico
    if max_len > 50: 
        return mejor_coincidencia
        
    return "No encontrado"
# ──────────────────────────────────────────────────────────────────────


def extraer_anios(texto):
    anios = re.findall(r"\b(19\d{2}|20\d{2})\b", texto)
    return sorted(set(anios))


def extraer_software(texto):
    lista = ["R", "Python", "SPSS", "Stata", "SAS", "Excel", "Power BI",
             "Tableau", "NVivo", "Atlas.ti", "Jamovi", "JASP", "Minitab", "MATLAB"]
    return [s for s in lista if re.search(r"\b" + re.escape(s) + r"\b", texto, re.IGNORECASE)]


def extraer_metodos(texto):
    metodos = [
        "regresión lineal", "regresión logística", "anova", "chi cuadrado",
        "prueba t", "correlación", "clustering", "k-means", "pca",
        "análisis factorial", "árbol de decisión", "random forest", "svm",
        "redes neuronales", "machine learning", "minería de datos",
        "topic modeling", "lda", "procesamiento de lenguaje natural",
        "tf-idf", "bert", "análisis de sentimiento"
    ]
    return [m for m in metodos if m.lower() in texto.lower()]


def generar_wordcloud(texto_limpio):
    """Genera una nube de palabras a partir del texto preprocesado."""
    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap="Blues",
        max_words=80,
        stopwords=stop_words
    ).generate(texto_limpio if texto_limpio.strip() else "texto académico universitario")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Nube de Palabras del Documento", fontsize=14, fontweight="bold")
    return fig


def lista_a_texto(lista):
    if isinstance(lista, list):
        return ", ".join(str(x) for x in lista) if lista else "No encontrado"
    return str(lista)


# ══════════════════════════════════════════════
# INTERFAZ STREAMLIT
# ══════════════════════════════════════════════

st.title("📄 Sistema Inteligente de Extracción de Información Académica")
st.markdown(
    "Aplicación de **Procesamiento de Lenguaje Natural** para extraer y analizar información "
    "relevante de documentos académicos en PDF. Utiliza **NER, TF-IDF, Análisis de Sentimiento, "
    "WordCloud y Topic Modeling (LDA)**."
)
st.markdown("---")

archivo_pdf = st.file_uploader("📂 Sube un documento académico en PDF", type=["pdf"])

if archivo_pdf is not None:
    with st.spinner("⏳ Procesando documento... esto puede tomar unos segundos."):

        # ── Extracción y limpieza ──
        texto_documento = extraer_texto_pdf(archivo_pdf)

        if not texto_documento.strip():
            st.error("No se pudo extraer texto. El PDF puede ser una imagen escaneada.")
            st.stop()

        texto_limpio = limpiar_texto(texto_documento)

        # ── NER ──
        df_entidades = extraer_entidades(texto_documento)
        if not df_entidades.empty:
            df_entidades_unicas = (
                df_entidades
                .drop_duplicates(subset=["Entidad", "Tipo"])
                .sort_values(["Tipo", "Entidad"])
                .reset_index(drop=True)
            )
        else:
            df_entidades_unicas = pd.DataFrame(columns=["Entidad", "Tipo", "Inicio", "Fin"])

        # ── TF-IDF ──
        df_keywords = extraer_keywords_tfidf(texto_documento, n=25)

        # ── Sentimiento ──
        df_sentimiento = analisis_sentimiento(texto_documento)

        # ── LDA ──
        df_temas = topic_modeling_lda(texto_limpio, n_topics=4)

        # ── Secciones (Ahora usando la función mejorada) ──
        resumen      = extraer_seccion(texto_documento, r"\bResumen\b",
                                       [r"\bPalabras clave\b", r"\bAbstract\b", r"\bIntroducción\b"])
        abstract     = extraer_seccion(texto_documento, r"\bAbstract\b",
                                       [r"\bKeywords\b", r"\bIntroducción\b", r"\bIntroduction\b"])
        introduccion = extraer_seccion(texto_documento, r"\bIntroducción\b",
                                       [r"\bMetodología\b", r"\bMétodo\b", r"\bResultados\b"])
        metodologia  = extraer_seccion(texto_documento, r"\bMetodología\b|\bMétodo\b",
                                       [r"\bResultados\b", r"\bDiscusión\b", r"\bConclusiones\b"])
        resultados   = extraer_seccion(texto_documento, r"\bResultados\b",
                                       [r"\bDiscusión\b", r"\bConclusiones\b", r"\bReferencias\b"])
        conclusiones = extraer_seccion(texto_documento, r"\bConclusiones\b|\bConclusión\b",
                                       [r"\bReferencias\b", r"\bBibliografía\b"])

        # ── Metadatos adicionales ──
        anios      = extraer_anios(texto_documento)
        software   = extraer_software(texto_documento)
        metodos    = extraer_metodos(texto_documento)
        autores    = df_entidades[df_entidades["Tipo"] == "PER"]["Entidad"].drop_duplicates().tolist() if not df_entidades.empty else []
        orgs       = df_entidades[df_entidades["Tipo"] == "ORG"]["Entidad"].drop_duplicates().tolist() if not df_entidades.empty else []

    st.success(f"✅ Documento procesado: **{len(texto_documento):,} caracteres** extraídos.")

    # Métricas rápidas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Entidades detectadas", len(df_entidades_unicas))
    col2.metric("Keywords TF-IDF", len(df_keywords))
    col3.metric("Segmentos analizados", len(df_sentimiento))
    col4.metric("Temas LDA", len(df_temas))

    st.markdown("---")

    # ── Tabs principales ──
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Resumen general",
        "🔍 Entidades NER",
        "📊 TF-IDF & WordCloud",
        "💬 Sentimiento",
        "🧩 Topic Modeling",
        "📑 Secciones",
        "📄 Texto extraído"
    ])

    # ── Tab 1: Resumen ──
    with tab1:
        st.subheader("Resumen académico consolidado")
        resumen_data = {
            "Años detectados":         lista_a_texto(anios),
            "Posibles autores":        lista_a_texto(autores[:10]),
            "Posibles instituciones":  lista_a_texto(orgs[:10]),
            "Software detectado":      lista_a_texto(software),
            "Métodos detectados":      lista_a_texto(metodos),
            "Resumen encontrado":      "Sí" if resumen != "No encontrado" else "No",
            "Abstract encontrado":     "Sí" if abstract != "No encontrado" else "No",
            "Introducción encontrada": "Sí" if introduccion != "No encontrado" else "No",
            "Metodología encontrada":  "Sí" if metodologia != "No encontrado" else "No",
            "Resultados encontrados":  "Sí" if resultados != "No encontrado" else "No",
            "Conclusiones encontradas":"Sí" if conclusiones != "No encontrado" else "No",
        }
        df_resumen = pd.DataFrame({"Campo": resumen_data.keys(), "Resultado": resumen_data.values()})
        st.dataframe(df_resumen, use_container_width=True, hide_index=True)
        csv_res = df_resumen.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("⬇️ Descargar resumen CSV", csv_res, "resumen_academico.csv", "text/csv")

    # ── Tab 2: NER ──
    with tab2:
        st.subheader("Reconocimiento de Entidades Nombradas (NER)")
        st.caption("Detecta personas, organizaciones, lugares y entidades misceláneas usando spaCy.")
        if df_entidades_unicas.empty:
            st.warning("No se detectaron entidades.")
        else:
            tipo_sel = st.selectbox("Filtrar por tipo:", ["Todas"] + sorted(df_entidades_unicas["Tipo"].unique().tolist()))
            df_mostrar = df_entidades_unicas if tipo_sel == "Todas" else df_entidades_unicas[df_entidades_unicas["Tipo"] == tipo_sel]
            st.dataframe(df_mostrar[["Entidad", "Tipo"]], use_container_width=True, hide_index=True)

            # Gráfico de distribución
            conteo = df_entidades_unicas["Tipo"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(conteo.index, conteo.values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
            ax.set_title("Distribución de tipos de entidades")
            ax.set_xlabel("Tipo")
            ax.set_ylabel("Cantidad")
            st.pyplot(fig)

            csv_ner = df_entidades_unicas.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("⬇️ Descargar entidades CSV", csv_ner, "entidades_ner.csv", "text/csv")

    # ── Tab 3: TF-IDF y WordCloud ──
    with tab3:
        st.subheader("Términos relevantes – TF-IDF")
        st.caption("TF-IDF pondera la importancia de cada término según su frecuencia relativa en el documento.")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.dataframe(df_keywords, use_container_width=True, hide_index=True)
            csv_kw = df_keywords.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("⬇️ Descargar keywords CSV", csv_kw, "keywords_tfidf.csv", "text/csv")
        with col_b:
            fig_tfidf, ax2 = plt.subplots(figsize=(6, 7))
            ax2.barh(df_keywords["Término"][:15], df_keywords["Puntaje TF-IDF"][:15], color="#1f77b4")
            ax2.invert_yaxis()
            ax2.set_xlabel("Puntaje TF-IDF")
            ax2.set_title("Top 15 términos")
            st.pyplot(fig_tfidf)

        st.markdown("---")
        st.subheader("☁️ Nube de palabras")
        fig_wc = generar_wordcloud(texto_limpio)
        st.pyplot(fig_wc)

    # ── Tab 4: Sentimiento ──
    with tab4:
        st.subheader("Análisis de Sentimiento")
        st.caption("Evalúa la polaridad (positivo/negativo/neutro) y subjetividad de cada segmento del texto usando TextBlob.")

        if df_sentimiento.empty:
            st.warning("No hay segmentos suficientes para analizar.")
        else:
            # Resumen de distribución
            dist = df_sentimiento["Sentimiento"].value_counts()
            col_x, col_y, col_z = st.columns(3)
            col_x.metric("🟢 Positivos", dist.get("Positivo", 0))
            col_y.metric("⚪ Neutros",   dist.get("Neutro", 0))
            col_z.metric("🔴 Negativos", dist.get("Negativo", 0))

            # Gráfico de polaridad
            fig_sent, ax3 = plt.subplots(figsize=(10, 3))
            colores = ["green" if s == "Positivo" else ("red" if s == "Negativo" else "gray")
                       for s in df_sentimiento["Sentimiento"]]
            ax3.bar(range(len(df_sentimiento)), df_sentimiento["Polaridad"], color=colores)
            ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax3.set_xlabel("Segmento #")
            ax3.set_ylabel("Polaridad")
            ax3.set_title("Polaridad por segmento del documento")
            st.pyplot(fig_sent)

            st.dataframe(df_sentimiento[["Segmento", "Polaridad", "Subjetividad", "Sentimiento"]],
                         use_container_width=True, hide_index=True)
            csv_sent = df_sentimiento.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("⬇️ Descargar sentimiento CSV", csv_sent, "sentimiento.csv", "text/csv")

    # ── Tab 5: Topic Modeling ──
    with tab5:
        st.subheader("Topic Modeling – LDA (Latent Dirichlet Allocation)")
        st.caption("LDA descubre temas latentes en el documento agrupando palabras que tienden a aparecer juntas.")
        st.dataframe(df_temas, use_container_width=True, hide_index=True)
        st.info(
            "Cada tema representa un grupo de palabras estadísticamente relacionadas. "
            "Interpreta los temas a partir de las palabras clave que los definen."
        )

    # ── Tab 6: Secciones ──
    with tab6:
        st.subheader("Secciones académicas extraídas")
        secciones = {
            "Resumen": resumen,
            "Abstract": abstract,
            "Introducción": introduccion,
            "Metodología": metodologia,
            "Resultados": resultados,
            "Conclusiones": conclusiones,
        }
        for nombre, contenido in secciones.items():
            icono = "✅" if contenido != "No encontrado" else "❌"
            with st.expander(f"{icono} {nombre}"):
                st.write(contenido[:5000] if contenido != "No encontrado" else "Sección no detectada en el documento.")

    # ── Tab 7: Texto ──
    with tab7:
        st.subheader("Texto completo extraído del PDF")
        st.text_area("Contenido", texto_documento, height=500)

else:
    st.info("👆 Sube un archivo PDF académico para comenzar el análisis.")

st.markdown("---")
st.caption(
    "PC3 – Procesamiento de Lenguaje Natural | Técnicas: NER (spaCy) · TF-IDF (scikit-learn) · "
    "Análisis de Sentimiento (TextBlob) · WordCloud · Topic Modeling LDA · Extracción de secciones (Regex)"
)