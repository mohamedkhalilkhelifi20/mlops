"""Load the 'champion' model from the MLflow Model Registry and predict churn for a new client."""

import mlflow
import mlflow.sklearn
import pandas as pd

# ===== Configuration =====
TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "churn-model"
ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{ALIAS}"

# Subscription columns are already one-hot encoded (same as at training time)
NEW_CLIENT = {
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True,  # Standard subscription
}


def load_champion(model_uri: str):
    """Load a scikit-learn model from the Registry (by alias, no need to know version or run)."""
    return mlflow.sklearn.load_model(model_uri)


def build_input(client_data: dict, model) -> pd.DataFrame:
    """Build a one-row DataFrame aligned with the columns the model was trained on."""
    client = pd.DataFrame([client_data])
    expected = list(model.feature_names_in_)

    missing = set(expected) - set(client.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour le modèle : {sorted(missing)}")

    # Keep only known columns, in the right order
    # (e.g. 'salaire' is dropped automatically if the champion was trained without it)
    return client[expected]


def predict_churn(model, client: pd.DataFrame) -> tuple[int, float]:
    """Return the predicted class (1 = will leave, 0 = stays) and the churn probability."""
    prediction = int(model.predict(client)[0])
    probability = float(model.predict_proba(client)[0][1])
    return prediction, probability


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)

    model = load_champion(MODEL_URI)
    print(f"Modèle chargé : {type(model).__name__}")
    print(f"Variables attendues : {list(model.feature_names_in_)}")

    client = build_input(NEW_CLIENT, model)
    prediction, probability = predict_churn(model, client)

    print(f"Le client va-t-il partir ? -> {'OUI' if prediction == 1 else 'NON'} "
          f"(probabilité de churn : {probability:.1%})")


if __name__ == "__main__":
    main()
