# =============================================================================
# 1. IMPORTS
# =============================================================================
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# =============================================================================
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================
df = pd.read_csv("churn_data.csv")

X = pd.get_dummies(
    df[["age", "salaire", "abonnement", "anciennete", "consommation"]],
    drop_first=True,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, df["churn"], test_size=0.2, random_state=42
)


# =============================================================================
# 3. CONFIGURATION DE MLFLOW
# =============================================================================
mlflow.set_experiment("churn-prediction")

# Active l'enregistrement automatique pour scikit-learn : MLflow note tout seul
# les paramètres du modèle, des métriques et le modèle lui-même.
mlflow.sklearn.autolog(serialization_format="cloudpickle")


# =============================================================================
# 4. ENTRAÎNEMENT ET ÉVALUATION
# =============================================================================
with mlflow.start_run(run_name="essai-auto"):

    # --- 4.1 Créer et entraîner le modèle ---
    model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)  # l'enregistrement se déclenche au fit()

    # --- 4.2 Évaluer le modèle ---
    # score() calcule l'accuracy sur les données de test
    print("score test :", model.score(X_test, y_test))
