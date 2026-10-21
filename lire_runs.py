# =============================================================================
# 1. IMPORT
# =============================================================================
import mlflow  # on réutilise le module MLflow pour LIRE les résultats


# =============================================================================
# 2. RÉCUPÉRER LES RUNS DE L'EXPÉRIENCE
# =============================================================================
# search_runs renvoie tous les runs de l'expérience sous forme de tableau pandas.
# order_by trie les lignes : "metrics.f1 DESC" = f1 décroissant (meilleur en premier)
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"],
)


# =============================================================================
# 3. AFFICHER LE TABLEAU COMPARATIF
# =============================================================================
# Colonnes à afficher : nom du run, deux paramètres, deux métriques
colonnes = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1",
]

print(runs[colonnes].to_string(index=False))  # index=False : sans numéros de ligne



# =============================================================================
# 4. AFFICHER LE MEILLEUR RUN
# =============================================================================
# La première ligne du tableau trié est le meilleur run
meilleur = runs.iloc[0]

print()
# [:8] : on n'affiche que les 8 premiers caractères de l'identifiant du run
print(
    "Meilleur run :", meilleur["tags.mlflow.runName"],
    "| id :", meilleur["run_id"][:8],
    "...",
)
