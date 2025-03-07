import joblib
import os
from dotenv import load_dotenv
# import requests
from fastapi import FastAPI
from pydantic import BaseModel
import deepl

# Charger les modèles depuis le dossier 'models'
# model_dir = os.path.join("api", "models")
# model = joblib.load(os.path.join(model_dir, "model_xgb.joblib"))
# vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.joblib"))

model = joblib.load('../models/xgb_model.pkl')
vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')

app = FastAPI()

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Récupérer la clé API depuis l'environnement
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

# Classe pour définir le format de l'entrée
class InputData(BaseModel):
    text: str


# Fonction pour traduire le texte en anglais
def translate_to_english(text: str) -> str:
    translator = deepl.Translator(DEEPL_API_KEY)
    try:
        translated_text = translator.translate_text(text, target_lang="EN")
        return translated_text.text
    except Exception as e:
        raise Exception(f"Erreur de traduction : {str(e)}")

# Endpoint de prédiction
@app.post("/predict")
def predict(data: InputData):
    text = data.text
    
    # Traduire systématiquement le texte en anglais
    translated_text = translate_to_english(text)
    
    # Transforme le texte traduit avec le vectoriseur TF-IDF
    text_tfidf = vectorizer.transform([translated_text])
    
    # Effectue la prédiction avec le modèle XGBoost
    prediction = model.predict(text_tfidf)
    
    # Générer la réponse en fonction de la prédiction
    result = "Fake News" if prediction[0] == 1 else "Not Fake News"
    
    # Retourner la prédiction avec le texte d'entrée et la prédiction
    return {
        "text": text,
        "prediction": result,
        "translated_text": translated_text  # Optionnel, montre le texte traduit en anglais
    }
