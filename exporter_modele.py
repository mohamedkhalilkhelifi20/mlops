# ===== 1. Importer les bibliothèques =====
import mlflow # le module principal de MLflow
import mlflow.sklearn # les fonctions dédiées à scikit-learn
# ===== 2. Se connecter à la base du Registry (celle du TP4) =====
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# ===== 3. Charger le champion depuis le Registry =====
model = mlflow.sklearn.load_model("models:/churn-model@champion")
# ===== 4. L'écrire dans un dossier ordinaire, « model » =====
# serialization_format="cloudpickle" : même format qu'aux TP3 et TP4
# (le format par défaut de MLflow 3 est refusé au chargement).
# Ce dossier sera copié dans l'image Docker : l'image n'aura plus besoin du Registry.
mlflow.sklearn.save_model(model, "model", serialization_format="cloudpickle")
print("Champion exporté dans le dossier model/")