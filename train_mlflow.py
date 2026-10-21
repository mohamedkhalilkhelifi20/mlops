# =============================================================================
# 1. IMPORTS
# =============================================================================
import pandas as pd                                   # lire et manipuler le CSV
import mlflow                                         # module principal de MLflow
import mlflow.sklearn                                 # partie de MLflow dédiée à scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 4 façons de mesurer la qualité du modèle :
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# =============================================================================
# 2. CHARGEMENT DES DONNÉES (une seule fois)
# =============================================================================
df = pd.read_csv("churn_data.csv")  # le tableau de 500 clients
y = df["churn"]                     # la réponse à prédire : 1 = part, 0 = reste


# =============================================================================
# 3. FONCTION : PRÉPARER LES DONNÉES POUR UNE LISTE DE COLONNES
# =============================================================================
def preparer(colonnes):
    # Transforme le texte 'abonnement' en colonnes 0/1 (si présente)
    X = pd.get_dummies(df[colonnes], drop_first=True)

    # 80 % apprentissage / 20 % test
    # Même découpage à chaque fois (random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# =============================================================================
# 4. FONCTION : LANCER UN ESSAI ET L'ENREGISTRER DANS MLFLOW
# =============================================================================
def lancer_essai(nom, n_estimators, max_depth, colonnes):
    X_train, X_test, y_train, y_test = preparer(colonnes)

    with mlflow.start_run(run_name=nom):  # ouvre un run au nom de l'essai

        # --- 4.1 Créer et entraîner le modèle ---
        model = RandomForestClassifier(
            n_estimators=n_estimators,  # nombre d'arbres
            max_depth=max_depth,        # profondeur maximale des arbres
            random_state=42,            # graine : résultat reproductible
        )
        model.fit(X_train, y_train)     # entraînement

        # --- 4.2 Prédire ---
        pred = model.predict(X_test)    # prédictions sur les 100 clients de test

        # --- 4.3 Paramètres : les réglages de cet essai ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(colonnes))  # colonnes utilisées

        # --- 4.4 Métriques : les résultats de cet essai ---
        mlflow.log_metric("accuracy", accuracy_score(y_test, pred))
        # zero_division=0 : évite une erreur si le modèle ne prédit aucun départ
        mlflow.log_metric("precision", precision_score(y_test, pred, zero_division=0))
        mlflow.log_metric("recall", recall_score(y_test, pred, zero_division=0))
        mlflow.log_metric("f1", f1_score(y_test, pred, zero_division=0))

        # --- 4.5 Artifact : le modèle sauvegardé dans ce run ---
        mlflow.sklearn.log_model(
            model, name="model", serialization_format="cloudpickle"
        )

        # --- 4.6 Affichage dans le terminal ---
        print(
            nom,
            "-> accuracy :", round(accuracy_score(y_test, pred), 3),
            "| f1 :", round(f1_score(y_test, pred, zero_division=0), 3),
        )


# =============================================================================
# 5. PROGRAMME PRINCIPAL
# =============================================================================
mlflow.set_experiment("churn-prediction")  # même expérience que l'Étape 1

TOUT = ["age", "salaire", "abonnement", "anciennete", "consommation"]

# Essai 1 : 100 arbres, profondeur 5, toutes les colonnes
lancer_essai("essai-1", 100, 5, TOUT)

# Essai 2 : modèle plus simple (50 arbres, profondeur 3)
lancer_essai("essai-2", 50, 3, TOUT)

# Essai 3 : sans la colonne salaire
lancer_essai("essai-3", 100, 5, ["age", "abonnement", "anciennete", "consommation"])
