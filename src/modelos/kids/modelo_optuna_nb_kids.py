from filter_and_divide_data import get_data_models_train_test
from training_utils_optuna import run_optuna_nb

if __name__ == "__main__":
    to_predict = "Made for kids"
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos"]
    
    print("Starting data acquisition")
    X_train, X_test, y_train, y_test = get_data_models_train_test(
        filtrado=True, 
        to_predict=to_predict
    )

    best_alpha = run_optuna_nb(
        project_="Modelo Optuna Naive Bayes Kids",
        name="V0",
        X_train=X_train,
        y_train=y_train,
        preprocess_type="TF-IDF",
        columns=columns,
        score_name="Precision",
        n_trials=20
    )