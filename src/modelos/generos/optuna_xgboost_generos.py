import optuna
from xgboost import XGBClassifier
from optuna_utils import entrenamiento

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 30, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.7),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "n_jobs": -1,
        "random_state": 42
    }
    
    project_name = "Modelo XGBoost Generos Optuna"
    trial_name = f"xgb_trial_{trial.number}"
    
    score = entrenamiento(
        project_=project_name,
        trial_name=trial_name,
        modelo=XGBClassifier,
        to_predict="Generos",
        max_features=3000,
        ngram=(1, 3),
        svd=150,
        preprocess_type="Word2Vec",
        columns=["Titulo", "Descripcion", "Tags", "Made for kids","Duracion","Subtitulos","Titulo_canal"],
        params=params,
        score_metric="F1",
        average="weighted",
        n_splits=2, 
        filtrado=2
    )
    
    return score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    
    study.optimize(objective, n_trials=10)

    print("\n" + "="*50)
    print("RESUMEN DE OPTIMIZACIÓN - XGBOOST")
    print("="*50)
    print(f"Mejor F1-Score:  {study.best_value:.4f}")
    print("-"*50)
    print("MEJORES HIPERPARÁMETROS:")
    for key, value in study.best_params.items():
        print(f" > {key:20}: {value}")
    print("="*50)