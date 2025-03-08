import streamlit as st
import requests

# URL de l'API FastAPI
API_URL = "http://localhost:8000/predict"

st.title("Détection de Fake News")

# Zone de saisie du texte
text = st.text_area("Entrez un texte à analyser", "")

# Sélection du modèle à utiliser
subject = st.radio("Choisissez le sujet :", ["general", "covid"])

if st.button("Prédire"):
    if text.strip():
        # Envoi de la requête à l'API FastAPI
        response = requests.post(API_URL, json={"text": text, "subject": subject})

        if response.status_code == 200:
            prediction = response.json().get("prediction")
            st.subheader(f"Résultat : {prediction}")
        else:
            st.error("Erreur lors de la prédiction.")
    else:
        st.warning("Veuillez entrer un texte avant de prédire.")
