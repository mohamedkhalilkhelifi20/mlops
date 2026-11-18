
# ===== 1. Importer les bibliothèques =====
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier


# ===== 2. Se connecter au serveur MLflow lancé par Docker Compose =====
# Depuis notre poste, il est joignable sur localhost grâce au port publié (5000)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-docker")


# ===== 3. Lire et préparer les données =====
df = pd.read_csv("churn_data.csv")

# get_dummies : transformer le texte « abonnement » en colonnes 0/1
X = pd.get_dummies(df.drop(columns=["churn"]), columns=["abonnement"])
y = df["churn"]


# ===== 4. Entraîner le modèle =====
modele = RandomForestClassifier(n_estimators=100, random_state=42)
modele.fit(X, y)


# ===== 5. Enregistrer le modèle dans le Registry du serveur =====
# registered_model_name : nom dans le catalogue (créé s'il n'existe pas)
# serialization_format : même format qu'aux TP précédents
with mlflow.start_run():
    mlflow.sklearn.log_model(
        modele,
        name="model",
        registered_model_name="churn-model",
        serialization_format="cloudpickle",
    )


# ===== 6. Poser l'alias « champion » sur la version créée =====
client = MlflowClient()

version = client.search_model_versions("name='churn-model'")[0].version

client.set_registered_model_alias("churn-model", "champion", version)

print(f"Version {version} de churn-model enregistrée avec l'alias champion")
