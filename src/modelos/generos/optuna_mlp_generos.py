import optuna
from sklearn.neural_network import MLPClassifier
from optuna_utils import entrenamiento

def objective(trial):
    params = {
        "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [
            (10,), (30,), (50,), (30, 15), (50, 25)
        ]),
        "activation": trial.suggest_categorical("activation", ["tanh", "relu"]),
        "solver": "adam",
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
        "learning_rate": "adaptive",
        "max_iter": 2000,
        "random_state": 42
    }
    
    project_name = "Modelo MLP Generos Optuna"
    trial_name = f"mlp_trial_{trial.number}"
    
    score = entrenamiento(
        project_=project_name,
        trial_name=trial_name,
        modelo=MLPClassifier,
        to_predict="Generos",
        max_features=5000,
        ngram=(1, 2),
        svd=150,
        preprocess_type="Word2Vec",
        columns=["Titulo", "Descripcion", "Tags", "Made for kids", "Duracion", "Subtitulos","Titulo_canal"],
        
        params=params,
        score_metric="F1",
        average="weighted",
        n_splits=2, 
        filtrado=0
    )
    
    return score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("\n" + "="*50)
    print("RESUMEN DE OPTIMIZACIÓN - MULTILAYER PERCEPTRON")
    print("="*50)
    print(f"Mejor F1-Score:  {study.best_value:.4f}")
    print("-"*50)
    print("MEJORES HIPERPARÁMETROS:")
    for key, value in study.best_params.items():
        print(f" > {key:20}: {value}")
    print("="*50)