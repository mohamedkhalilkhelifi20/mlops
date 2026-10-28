"""API FastAPI de prédiction de churn : sert le modèle 'champion' du Model Registry MLflow."""

from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ===== Configuration =====
TRACKING_URI = "sqlite:///mlflow.db"
MODEL_URI = "models:/churn-model@champion"

# Le modèle est chargé UNE SEULE FOIS au démarrage et partagé entre les requêtes
ml: dict = {}


# ===== Schémas =====
class Client(BaseModel):
    age: int = Field(..., ge=18, le=120)
    salaire: float = Field(..., ge=0)
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float = Field(..., ge=0)
    consommation: float = Field(..., ge=0)


class Prediction(BaseModel):
    churn: Literal[0, 1]
    message: str


# ===== Cycle de vie =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    ml["model"] = mlflow.sklearn.load_model(MODEL_URI)
    yield
    ml.clear()


app = FastAPI(title="API Churn", lifespan=lifespan)


# ===== Logique métier =====
def preparer_features(client: Client, colonnes_modele) -> pd.DataFrame:
    """Reproduit le prétraitement de l'entraînement (get_dummies) et aligne les colonnes."""
    ligne = pd.DataFrame([client.model_dump()])
    ligne = pd.get_dummies(ligne)
    # Colonnes du modèle, dans le bon ordre ; les absentes valent 0
    return ligne.reindex(columns=colonnes_modele, fill_value=0)


# ===== Routes =====
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in ml}


@app.post("/predict", response_model=Prediction)
def predict(client: Client) -> Prediction:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    features = preparer_features(client, model.feature_names_in_)
    resultat = int(model.predict(features)[0])
    return Prediction(
        churn=resultat,
        message="Le client va partir" if resultat == 1 else "Le client reste",
    )