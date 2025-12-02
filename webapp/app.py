import streamlit as st
import pandas as pd
import requests
import os


# API URL (docker-compose sets API_URL in environment)
API_URL = os.getenv("API_URL", "http://localhost:5000/predict")

# Load TEST dataset (raw CSV)
TEST_PATH = "data/jigsaw/test.csv"
df_test = pd.read_csv(TEST_PATH)

# The column with the text
texts = df_test["comment_text"].tolist()

# Streamlit UI---
st.title("Toxic Comment Classifier – Test Set Explorer")

st.write("Selecciona un comentario del dataset de test:")

selected_text = st.selectbox("Comentario:", texts)

st.write("### Texto seleccionado:")
st.write(selected_text)

if st.button("Clasificar"):
    payload = {"text": selected_text}

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        st.write("### Resultados del modelo:")
        st.json(result)

    except Exception as e:
        st.error(f"Error llamando a la API: {e}")
