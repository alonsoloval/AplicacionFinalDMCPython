import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Configuración de la página
st.set_page_config(page_title="EDA - Insurance Company", layout="wide")

#Clase DataAnalyzer
class DataAnalyzer:
    """Clase para encapsular funciones de análisis y visualización."""
    def __init__(self, df):
        self.df = df

    def classify_variables(self):
        """Identifica variables numéricas y categóricas según el contexto del dataset."""
        # Columnas conocidas que son categóricas por diccionario de datos
        known_cats = ['sourcing_channel', 'residence_area_type', 'renewal']
        
        num_cols = []
        cat_cols = []
        
        for col in self.df.columns:
            # Ignoramos el ID ya que es un identificador único, no una variable 
            if col.lower() == 'id':
                continue
            # Clasificamos categóricas si están en nuestra lista conocida o si son de tipo categoría
            elif col in known_cats or pd.api.types.is_object_dtype(self.df[col]) or pd.api.types.is_categorical_dtype(self.df[col]):
                cat_cols.append(col)
            # El resto que sea numérico entra a la lista de numéricas
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                num_cols.append(col)
                
        return num_cols, cat_cols
    
    def get_missing_summary(self):
        """Retorna un conteo de valores nulos."""
        return self.df.isnull().sum()
    
    def plot_histogram(self, column, bins):
        """Genera un histograma para una variable numérica."""
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(self.df[column], bins=bins, kde=True, ax=ax, color='skyblue')
        ax.set_title(f'Distribución de {column}')
        return fig

    def plot_bar(self, column):
        """Genera un gráfico de barras para una variable categórica."""
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=self.df, x=column, palette='viridis', ax=ax)
        ax.set_title(f'Conteo de {column}')
        plt.xticks(rotation=45)
        return fig
    
    def plot_bivariate_num_cat(self, num_col, cat_col):
        """Genera un boxplot para analizar numérico vs categórico."""
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=self.df, x=cat_col, y=num_col, palette='Set2', ax=ax)
        ax.set_title(f'{num_col} vs {cat_col}')
        return fig
# MENÚ LATERAL (Sidebar)
st.sidebar.title("Menú Principal")
menu = st.sidebar.radio("Navegación:", [
    "1. Home", 
    "2. Carga y Análisis (EDA)", 
    "3. Conclusiones finales"
])

# MÓDULO 1: HOME
if menu == "1. Home":
    st.title("Análisis Exploratorio: Insurance Company")
    st.write("### Objetivo del Análisis")
    st.write("El objetivo de esta aplicación es realizar un Análisis Exploratorio de Datos (EDA) sobre los datos de clientes de una compañía de seguros, identificando patrones de renovación, características demográficas y comportamientos de pago.")
    
    st.write("### Datos del Autor")
    st.markdown("""
    * **Nombre completo:** Alonso Lovon
    * **Curso / Especialización:** Especialización en Python for Analytics
    * **Año:** 2026
    """)
    
    st.write("### Sobre el Dataset")
    st.write("El dataset contiene información histórica de pólizas de clientes, incluyendo variables como ingresos, edad, canales de captación, retrasos en pagos y si el cliente renovó o no su póliza.")
    
    st.write("### Tecnologías Utilizadas")
    st.write("Python, Pandas, NumPy, Matplotlib, Seaborn, y Streamlit.")

# MÓDULO 2: CARGA DEL DATASET Y EDA
elif menu == "2. Carga y Análisis (EDA)":
    st.title("Carga de Datos y Análisis Exploratorio")
    
    # Carga del dataset
    uploaded_file = st.file_uploader("Sube el archivo Insurance_Company.csv", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        analyzer = DataAnalyzer(df)
        num_cols, cat_cols = analyzer.classify_variables() # Obtenemos variables para todo el módulo
        
        st.success("¡Archivo cargado correctamente!")
        
        if st.checkbox("Mostrar vista previa del dataset (head)"):
            st.dataframe(df.head())
        
        st.write(f"**Dimensiones del dataset:** {df.shape[0]} filas y {df.shape[1]} columnas.")
        
        st.divider()
        st.header("Análisis Exploratorio de Datos (EDA)")
        
        # 10 pestañas para los 10 ítems de análisis
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "Ítem 1", "Ítem 2", "Ítem 3", "Ítem 4", "Ítem 5", 
            "Ítem 6", "Ítem 7", "Ítem 8", "Ítem 9", "Ítem 10"
        ])
        
        with tab1:
            st.subheader("Ítem 1: Información general del dataset")
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            
        with tab2:
            st.subheader("Ítem 2: Clasificación de variables")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Numéricas ({len(num_cols)}):**")
                st.write(num_cols)
            with col2:
                st.write(f"**Categóricas ({len(cat_cols)}):**")
                st.write(cat_cols)
                
        with tab3:
            st.subheader("Ítem 3: Estadísticas descriptivas")
            st.dataframe(df[num_cols].describe())
            st.write("*Interpretación:* Se observan las medias, medianas y la dispersión (desviación estándar) de las variables numéricas clave.")
            
        with tab4:
            st.subheader("Ítem 4: Análisis de valores faltantes")
            missing_data = analyzer.get_missing_summary()
            nulos = missing_data[missing_data > 0]
            if len(nulos) > 0:
                st.dataframe(nulos)
                st.write("Visualización de variables con nulos:")
                st.bar_chart(nulos)
            else:
                st.info("No se encontraron valores nulos en el dataset.")
            
        with tab5:
            st.subheader("Ítem 5: Distribución de variables numéricas")
            if num_cols:
                num_sel = st.selectbox("Selecciona una variable numérica:", num_cols, key='hist_var')
                bins_slider = st.slider("Número de bins para el histograma:", min_value=10, max_value=100, value=30)
                st.pyplot(analyzer.plot_histogram(num_sel, bins_slider))
            
        with tab6:
            st.subheader("Ítem 6: Análisis de variables categóricas")
            if cat_cols:
                cat_sel = st.selectbox("Selecciona una variable categórica:", cat_cols, key='bar_var')
                st.pyplot(analyzer.plot_bar(cat_sel))
            
        with tab7:
            st.subheader("Ítem 7: Análisis bivariado (Numérico vs Categórico)")
            if num_cols and cat_cols:
                col_biv1, col_biv2 = st.columns(2)
                with col_biv1:
                    num_biv = st.selectbox("Variable Numérica:", num_cols, key='num_biv')
                with col_biv2:
                    cat_biv = st.selectbox("Variable Categórica:", cat_cols, key='cat_biv')
                st.pyplot(analyzer.plot_bivariate_num_cat(num_biv, cat_biv))
            
        with tab8:
            st.subheader("Ítem 8: Análisis bivariado (Categórico vs Categórico)")
            if len(cat_cols) >= 2:
                col_cat1, col_cat2 = st.columns(2)
                with col_cat1:
                    cat1 = st.selectbox("Variable Categórica 1:", cat_cols, index=0, key='cat_biv1')
                with col_cat2:
                    cat2 = st.selectbox("Variable Categórica 2 (Hue):", cat_cols, index=1, key='cat_biv2')
                
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.countplot(data=df, x=cat1, hue=cat2, palette='Set1', ax=ax)
                st.pyplot(fig)
            else:
                st.write("Se requieren al menos 2 variables categóricas para este análisis.")
                
        with tab9:
            st.subheader("Ítem 9: Análisis basado en parámetros seleccionados")
            st.write("Selecciona múltiples variables numéricas para ver su matriz de correlación:")
            multi_vars = st.multiselect("Variables para correlación:", num_cols, default=num_cols[:3])
            if len(multi_vars) > 1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(df[multi_vars].corr(), annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
                st.pyplot(fig)
            else:
                st.warning("Selecciona al menos 2 variables numéricas.")
                
        with tab10:
            st.subheader("Ítem 10: Hallazgos clave")
            st.info("""
            * **Insight 1:** La mayoría de los clientes residen en áreas urbanas y el canal de captación influye en el volumen de primas.
            * **Insight 2:** Existen clientes con retrasos de 3 a 6 meses que representan un riesgo claro de no renovación de póliza.
            * **Insight 3:** Los ingresos de los clientes (`Income`) muestran una gran variabilidad, lo que puede influir en la capacidad de pago a tiempo.
            """)
    else:
        st.warning("Ningún análisis se ejecutará hasta que cargues el archivo CSV.")


# MÓDULO 3: CONCLUSIONES FINALES
elif menu == "3. Conclusiones finales":
    st.title("Conclusiones Finales")
    st.write("Basado en el Análisis Exploratorio realizado, se presentan las siguientes conclusiones orientadas a la toma de decisiones:")
    
    st.markdown("""
    1. **Estrategia de Retención:** Los clientes con pagos demorados entre 3 y 6 meses (`Count_3-6_months_late`) muestran una menor tasa de renovación. Se debe implementar una campaña de recordatorios de pago tempranos.
    2. **Segmentación por Ingresos:** Existe una relación directa entre el ingreso (`Income`) y el valor de la prima (`premium`). Se pueden diseñar paquetes premium específicos para los percentiles de mayores ingresos.
    3. **Optimización de Canales:** El canal de captación con mayor volumen de clientes no necesariamente es el que tiene la mejor tasa de renovación. Se debe reasignar presupuesto de marketing hacia los canales con mayor fidelización.
    4. **Evaluación de Riesgo (`application_underwriting_score`):** Esta métrica es un buen termómetro de la confiabilidad del cliente. Podría usarse para ofrecer descuentos por "buen comportamiento" a los clientes con puntajes altos.
    5. **Enfoque Geográfico:** Si bien el área urbana concentra más clientes, el área rural podría representar un nicho desatendido. Se requiere una investigación de mercado para crear productos adaptados a las zonas rurales.
    """)