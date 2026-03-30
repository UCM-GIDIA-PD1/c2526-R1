import optuna
import wandb
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from preprocess_utils import build_preprocess

def objective(trial, X_train, y_train, preprocess_type, columns, score_name):

    #Rango de búsqueda para alpha
    alpha_suggested = trial.suggest_float("alpha", 1e-4, 1.0, log=True)

    preprocess = build_preprocess(
        preprocess_type, 
        columns, 
        X_train, 
        5000,
        (1, 2),
        None     # svd
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", MultinomialNB(alpha=alpha_suggested))
    ])

    metric = "f1_weighted" if score_name == "F1" else "precision_weighted"
    
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring=metric, n_jobs=1) #Usa un solo núcleo para evitar problemas de paralelización con Optuna
    current_score = scores.mean()

    return current_score

def run_optuna_nb(project_, name, X_train, y_train, preprocess_type, columns, score_name, n_trials=20):

    wandb.init(project=project_, name=f"{name}_optuna")

    study = optuna.create_study(direction="maximize")
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, preprocess_type, columns, score_name),
        n_trials=n_trials
    )

    wandb.log({
        "best_alpha_optuna": study.best_params["alpha"],
        "best_cv_score_optuna": study.best_value
    })

    print(f"\n[OPTUNA] Mejor Alpha encontrado: {study.best_params['alpha']}")
    print(f"[OPTUNA] Mejor Score de CV: {study.best_value:.4f}")

    wandb.finish()
    return study.best_params["alpha"]