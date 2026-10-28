"""Client de test pour l'API Churn : vérifie /health, /predict et la validation des entrées."""

import requests

# ===== Configuration =====
URL = "http://localhost:8000"
TIMEOUT = 5  # secondes : évite qu'un script reste bloqué si l'API ne répond pas

# ===== Cas de test =====
CLIENT_A_RISQUE = {
    "age": 35, "salaire": 2500, "abonnement": "Standard",
    "anciennete": 1, "consommation": 10,
}
CLIENT_FIDELE = {
    "age": 50, "salaire": 4000, "abonnement": "Premium",
    "anciennete": 10, "consommation": 80,
}
CLIENT_INVALIDE = {**CLIENT_A_RISQUE, "age": "abc"}  # âge en texte : doit être rejeté


def check_health() -> dict:
    """Vérifie que l'API est en vie et que le modèle est chargé."""
    r = requests.get(f"{URL}/health", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def predict(client: dict) -> requests.Response:
    """Envoie un client à /predict et retourne la réponse brute."""
    return requests.post(f"{URL}/predict", json=client, timeout=TIMEOUT)


def main() -> None:
    # 1. L'API est-elle en vie ?
    print("GET /health ->", check_health())

    # 2. et 3. Prédictions valides
    for nom, client in [("client à risque", CLIENT_A_RISQUE),
                        ("client fidèle", CLIENT_FIDELE)]:
        r = predict(client)
        print(f"POST /predict ({nom}) ->", r.status_code, r.json())

    # 4. Erreur volontaire : la validation Pydantic doit renvoyer 422
    r = predict(CLIENT_INVALIDE)
    print("POST /predict (age='abc') ->", r.status_code)
    if r.status_code == 422:
        print("Message :", r.json()["detail"][0]["msg"])
    else:
        print("Attention : une erreur 422 était attendue.")


if __name__ == "__main__":
    main()