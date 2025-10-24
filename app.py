# 🌟 App de Predicción de Enfermedad Cardíaca - Versión 3.0
# Incluye: Predicción, Exploración de Datos, Comparación de Modelos 💓📊🤖

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt

# 🎨 Configuración de la página
st.set_page_config(
    page_title="💓 Predicción de Enfermedad Cardíaca",
    layout="centered",
    page_icon="❤️"
)

# 💖 Título principal
st.title("💓 Predicción de Enfermedad Cardíaca")
st.markdown("Esta aplicación utiliza **Machine Learning** para predecir la presencia de enfermedad cardíaca 🧠⚙️.")

# 🚀 Cargar modelo y artefactos
@st.cache_resource
def cargar_modelo_y_artefactos():
    model = joblib.load("modelo_RF.pkl")
    scaler = joblib.load("scaler.pkl")
    try:
        label_encoders = joblib.load("label_encoders.pkl")
    except Exception:
        label_encoders = None
    with open("dummy_columns.json", "r", encoding="utf-8") as f:
        dummy_cols = json.load(f)
    return model, scaler, label_encoders, dummy_cols

model, scaler, label_encoders, dummy_cols = cargar_modelo_y_artefactos()

# 📊 Panel lateral
st.sidebar.header("🔍 Opciones")
opcion = st.sidebar.radio(
    "Selecciona una opción:",
    ["🏠 Inicio", "📈 Exploración de Datos", "🤖 Predicción"],
    index=0
)

# --- FUNCIONES AUXILIARES ---
def preprocess_input(row):
    df = pd.DataFrame([row])

    # 🔤 Mapear valores categóricos
    map_sex = {'M':1, 'F':0, 'Male':1, 'Female':0}
    if df['Sex'].iloc[0] in map_sex:
        df['Sex'] = df['Sex'].map(map_sex)

    map_fast = {'0':0, '1':1, 0:0, 1:1}
    df['FastingBS'] = df['FastingBS'].map(map_fast)

    map_ex = {'Y':1, 'N':0, 'S':1, 's':1, 'n':0, 'N':0}
    df['ExerciseAngina'] = df['ExerciseAngina'].map(map_ex)

    # 🧩 One-hot encoding
    df = pd.get_dummies(df, columns=['ChestPainType','RestingECG','ST_Slope'], drop_first=False)
    df = df.reindex(columns=dummy_cols, fill_value=0)

    # ⚖️ Escalar variables numéricas
    posibles_numericas = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
    cols_to_scale = [c for c in posibles_numericas if c in df.columns]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df

# --- SECCIÓN INICIO ---
if opcion == "🏠 Inicio":
    st.markdown("""
    ### 💬 ¿Qué hace esta app?
    Esta herramienta permite predecir si un paciente **tiene o no** enfermedad cardíaca ❤️.

    - **Modelo usado:** Random Forest 🌳  
    - **Datos de entrenamiento:** *Heart Disease Dataset (UCI)*  
    - **Autores:**
    - Paniagua, Luis    
    - Vivanco, Jimena    
    - Gòmez, Gustavo     
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774278.png", width=150)
    st.success("Usa el menú de la izquierda para explorar los datos o realizar una predicción 🤗")

# --- SECCIÓN EXPLORACIÓN DE DATOS ---
elif opcion == "📈 Exploración de Datos":
    st.header("📊 Exploración de Datos")

    try:
        data = pd.read_csv("heart.csv")

        # 📝 Renombrar columnas
        data = data.rename(columns={
            "Age": "Edad (años)",
            "Sex": "Sexo [M: Masculino, F: Femenino]",
            "ChestPainType": "Tipo de Dolor Torácico [TA, ATA, NAP, ASY]",
            "RestingBP": "Presión Arterial en Reposo (mm Hg)",
            "Cholesterol": "Colesterol Sérico (mg/dl)",
            "FastingBS": "Glucemia en Ayunas [1: >120 mg/dl, 0: ≤120]",
            "RestingECG": "ECG en Reposo [Normal, ST, HVI]",
            "MaxHR": "Frecuencia Cardíaca Máxima",
            "ExerciseAngina": "Angina de Ejercicio [S/N]",
            "Oldpeak": "Oldpeak (Depresión ST)",
            "ST_Slope": "Pendiente del ST [Up/Flat/Down]",
            "HeartDisease": "Enfermedad Cardíaca [1: Sí, 0: No]"
        })

        st.subheader("👀 Vista previa del dataset")
        st.dataframe(data.head())

        # ❤️ Distribución del target
        st.subheader("❤️ Distribución de la variable objetivo (Enfermedad Cardíaca)")
        fig, ax = plt.subplots()
        sns.countplot(data=data, x="Enfermedad Cardíaca [1: Sí, 0: No]", palette="Reds", ax=ax)
        ax.set_xticklabels(["No (0)", "Sí (1)"])
        st.pyplot(fig)

        # 🔢 Correlación solo entre numéricas
        st.subheader("📉 Mapa de correlación entre variables numéricas")
        data_numerica = data.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(data_numerica.corr(), cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
        st.pyplot(fig)

        # 🌟 Importancia de variables
        if hasattr(model, "feature_importances_"):
            st.subheader("🌟 Importancia de las variables según el modelo")
            importances = model.feature_importances_
            feat_importances = pd.Series(importances, index=dummy_cols).sort_values(ascending=False)[:10]

            mapa_vars = {
                "Age": "Edad",
                "Sex": "Sexo",
                "ChestPainType": "Tipo de Dolor",
                "RestingBP": "Presión en Reposo",
                "Cholesterol": "Colesterol",
                "FastingBS": "Glucemia en Ayunas",
                "RestingECG": "ECG en Reposo",
                "MaxHR": "Frecuencia Máxima",
                "ExerciseAngina": "Angina Ejercicio",
                "Oldpeak": "Oldpeak",
                "ST_Slope": "Pendiente ST"
            }

            feat_importances.index = [mapa_vars.get(col.split('_')[0], col) for col in feat_importances.index]
            fig, ax = plt.subplots()
            sns.barplot(x=feat_importances, y=feat_importances.index, palette="magma", ax=ax)
            ax.set_xlabel("Importancia")
            ax.set_ylabel("Variable")
            st.pyplot(fig)
        else:
            st.info("El modelo actual no proporciona importancias de variables.")
    except Exception as e:
        st.error(f"No se pudo cargar el dataset: {e}")

# --- SECCIÓN PREDICCIÓN ---
elif opcion == "🤖 Predicción":
    st.header("🤖 Hacer una predicción")
    st.write("Ingresa los datos del paciente para estimar la probabilidad de enfermedad cardíaca 💉")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Edad (años)", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sexo", options=["M", "F"])
        chest_pain = st.selectbox("Tipo de Dolor Torácico", options=["TA","ATA","NAP","ASY"])
        resting_bp = st.number_input("Presión arterial en reposo (mm Hg)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Colesterol sérico (mg/dl)", min_value=50, max_value=600, value=200)

    with col2:
        fasting_bs = st.selectbox("Glucemia en ayunas (>120 mg/dl)", options=["0","1"])
        resting_ecg = st.selectbox("ECG en reposo", options=["Normal","ST","LVH"])
        max_hr = st.number_input("Frecuencia cardíaca máxima", min_value=60, max_value=220, value=150)
        exercise_angina = st.selectbox("Angina de ejercicio", options=["S","N"])
        oldpeak = st.number_input("Oldpeak (depresión ST)", min_value=-5.0, max_value=10.0, value=1.0, step=0.1)
        st_slope = st.selectbox("Pendiente del ST", options=["Up","Flat","Down"])

    if st.button("💡 Predecir"):
        row = {
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": resting_bp,
            "Cholesterol": chol,
            "FastingBS": fasting_bs,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope
        }

        try:
            X_input = preprocess_input(row)
            pred = model.predict(X_input)
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_input)[:,1][0]

            st.markdown("### 🩺 Resultado de la Predicción")
            if int(pred[0]) == 1:
                st.error("⚠️ El modelo predice que **EL PACIENTE TIENE RIESGO DE ENFERMEDAD CARDÍACA.**")
                st.markdown('Para más información, ingresar a la siguiente dirección: '
                            '<a href="https://medlineplus.gov/spanish/heartdiseases.html" target="_blank">clic aquí</a>',
                            unsafe_allow_html=True)
            else:
                st.success("💚 El modelo predice que **EL PACIENTE NO TIENE ENFERMEDAD CARDÍACA.**")

            if proba is not None:
                st.metric(label="Probabilidad estimada de enfermedad", value=f"{proba*100:.2f}%")

        except Exception as e:
            st.error(f"Error durante la predicción: {e}")