"""Démo du Model Registry : crée une nouvelle version, déplace l'alias 'champion', puis le restaure."""

import mlflow
from mlflow import MlflowClient

# ===== Configuration =====
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "churn-prediction"
RUN_NAME = "essai-2"          # le modèle le plus faible du TP3
MODEL_NAME = "churn-model"
MODEL_ARTIFACT_PATH = "model"
ALIAS = "champion"


def get_run_id_by_name(experiment_name: str, run_name: str) -> str:
    """Retourne l'id du run le plus récent portant ce nom."""
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        raise ValueError(f"Aucun run nommé '{run_name}' dans '{experiment_name}'.")
    return runs.iloc[0]["run_id"]


def register_model(run_id: str, model_name: str, artifact_path: str):
    """Enregistre le modèle du run : crée une NOUVELLE VERSION dans le Registry."""
    return mlflow.register_model(f"runs:/{run_id}/{artifact_path}", model_name)


def get_alias_version(client: MlflowClient, model_name: str, alias: str) -> str:
    """Retourne la version actuellement pointée par l'alias."""
    return client.get_model_version_by_alias(model_name, alias).version


def move_alias(client: MlflowClient, model_name: str, alias: str, version: str) -> None:
    """Déplace l'alias vers la version donnée."""
    client.set_registered_model_alias(model_name, alias, version)


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # 1. Retrouver le run « essai-2 »
    run_id = get_run_id_by_name(EXPERIMENT_NAME, RUN_NAME)

    # 2. Créer une nouvelle version
    new_version = register_model(run_id, MODEL_NAME, MODEL_ARTIFACT_PATH).version
    print(f"Nouvelle version créée : {new_version}")

    # 3. Où pointe « champion » aujourd'hui ?
    original_version = get_alias_version(client, MODEL_NAME, ALIAS)
    print(f"Champion actuel : version {original_version}")

    # 4. Déplacer l'alias vers la nouvelle version, puis le restaurer
    #    try/finally : l'alias est remis en place même si une erreur survient
    try:
        move_alias(client, MODEL_NAME, ALIAS, new_version)
        print(f"Après déplacement : champion -> version "
              f"{get_alias_version(client, MODEL_NAME, ALIAS)}")
    finally:
        move_alias(client, MODEL_NAME, ALIAS, original_version)
        print(f"Après retour : champion -> version "
              f"{get_alias_version(client, MODEL_NAME, ALIAS)}")


if __name__ == "__main__":
    main()