"""
Génération d'un jeu de données fictif de clients pour un cas de churn.

Features : age, salaire, abonnement, anciennete, consommation
Target   : churn (1 = client à risque de départ, 0 = client stable)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
SEED = 42                      # point de départ du générateur aléatoire
N_CLIENTS = 500                # nombre de clients fictifs à générer
OUTPUT_FILE = "churn_data.csv"

SEUIL_CONSOMMATION = 30        # en dessous : client peu actif
SEUIL_ANCIENNETE = 2           # en dessous : client récent (en années)


# ---------------------------------------------------------------------------
# Fonctions
# ---------------------------------------------------------------------------
def generer_features(n: int) -> pd.DataFrame:
    """Construit les colonnes d'entrée (features) du futur modèle."""
    return pd.DataFrame({
        "age": np.random.randint(18, 70, n),                  # entre 18 et 69 ans
        "salaire": np.random.randint(1200, 8000, n),          # salaire mensuel simulé
        "abonnement": np.random.choice(
            ["Basique", "Standard", "Premium"], n
        ),                                                    # type d'offre
        "anciennete": np.random.randint(0, 15, n),            # en années
        "consommation": np.random.randint(0, 100, n),         # score d'usage sur 100
    })


def ajouter_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute la colonne cible 'churn'.

    Règle volontairement simplifiée : un client peu actif ET récent
    est considéré comme « à risque de départ ».
    """
    est_peu_actif = data["consommation"] < SEUIL_CONSOMMATION
    est_recent = data["anciennete"] < SEUIL_ANCIENNETE

    data["churn"] = (est_peu_actif & est_recent).astype(int)  # True/False -> 1/0
    return data


def sauvegarder(data: pd.DataFrame, chemin: str) -> None:
    """
    Enregistre le tableau au format CSV.

    Attention : à ce stade, le fichier n'est protégé par aucun
    système de gestion de versions.
    """
    data.to_csv(chemin, index=False)


def afficher_controle(data: pd.DataFrame) -> None:
    """Affiche deux contrôles visuels rapides."""
    print(data.head())
    print(f"Taux de churn : {data['churn'].mean():.1%}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
def main() -> None:
    # La graine fixe le point de départ du générateur : les mêmes données
    # sont recréées à chaque exécution (reproductibilité des expériences).
    np.random.seed(SEED)

    data = generer_features(N_CLIENTS)
    data = ajouter_target(data)
    sauvegarder(data, OUTPUT_FILE)
    afficher_controle(data)


if __name__ == "__main__":
    main()