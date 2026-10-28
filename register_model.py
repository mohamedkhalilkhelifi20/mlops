"""Enregistre le meilleur run MLflow dans le Model Registry et lui pose l'alias 'champion'."""

import mlflow
from mlflow import MlflowClient

# ===== Configuration =====
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "churn-prediction"
METRIC_NAME = "f1"
MODEL_NAME = "churn-model"
MODEL_ARTIFACT_PATH = "model"
ALIAS = "champion"


def get_best_run(experiment_name: str, metric: str):
    """Retourne le meilleur run d'une expérience selon une métrique (ordre décroissant)."""
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{metric} DESC"],
    )
    if runs.empty:
        raise ValueError(f"Aucun run trouvé pour l'expérience '{experiment_name}'.")
    return runs.iloc[0]


def register_model(run_id: str, model_name: str, artifact_path: str):
    """Enregistre le modèle d'un run dans le Registry (le crée s'il n'existe pas)."""
    return mlflow.register_model(f"runs:/{run_id}/{artifact_path}", model_name)


def set_alias(model_name: str, alias: str, version: str) -> None:
    """Associe un alias à une version du modèle."""
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, version)


def main() -> None:
    # 1. Connexion à la base de tracking (la même que le TP3)
    mlflow.set_tracking_uri(TRACKING_URI)

    # 2. Meilleur run
    best = get_best_run(EXPERIMENT_NAME, METRIC_NAME)
    run_id = best["run_id"]
    print(f"Meilleur run : {best['tags.mlflow.runName']} | id : {run_id[:8]}... "
          f"| {METRIC_NAME} = {best[f'metrics.{METRIC_NAME}']:.4f}")

    # 3. Enregistrement dans le Registry
    result = register_model(run_id, MODEL_NAME, MODEL_ARTIFACT_PATH)
    print(f"Modèle enregistré : {MODEL_NAME} | version {result.version}")

    # 4. Alias « champion »
    set_alias(MODEL_NAME, ALIAS, result.version)
    print(f"Alias '{ALIAS}' -> version {result.version}")


if __name__ == "__main__":
    main()
