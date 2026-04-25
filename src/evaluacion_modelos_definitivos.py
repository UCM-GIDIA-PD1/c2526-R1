from joblib import load
import json
import wandb
import pandas as pd
from sklearn.inspection import permutation_importance
from comun.Server_PD import download_model_minio
from sklearn.metrics import classification_report, accuracy_score
from comun.filter_and_divide_data import extract_definitive_test, get_data_models_train_test_latest

def evaluar_modelo_final(proyecto, nombre, model, X_test_trans, y_test, encoder = None):
    wandb.init(project=proyecto, name=nombre)

    raw_preds = model.predict(X_test_trans)
    
    if encoder:
        class_names = encoder.classes_.tolist()
        y_test_text = encoder.inverse_transform(y_test)
        pred_test_text = encoder.inverse_transform(raw_preds)
    else:
        class_names = ["Adult", "Kids"]
        y_test_text = y_test
        pred_test_text = raw_preds

    # Matriz de Confusión
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, 
            y_true=y_test, 
            preds=raw_preds, 
            class_names=class_names)
    })

    # Métricas
    report = classification_report(y_test_text, pred_test_text, output_dict=True)
    
    wandb.summary["accuracy"] = report["accuracy"]
    wandb.summary["f1_weighted"] = report["weighted avg"]["f1-score"]
    wandb.summary["precision_weighted"] = report["weighted avg"]["precision"]
    wandb.summary["recall_weighted"] = report["weighted avg"]["recall"]
    
    print(f"Evaluación de {nombre} finalizada. F1-Score: {report['weighted avg']['f1-score']:.4f}")
    wandb.finish()
def calcular_importancia_generos(model, pipe, X_test, y_test):
    
    class FullPipelineWrapper:
        def __init__(self, pipe, model):
            self.pipe = pipe
            self.model = model

        def predict(self, X):
            X_trans = self.pipe.transform(X)
            return self.model.predict(X_trans)
            
        def score(self, X, y):
            # Este es el método que faltaba para que permutation_importance funcione
            preds = self.predict(X)
            return accuracy_score(y, preds)

        def fit(self, X, y=None):
            return self

    full_model = FullPipelineWrapper(pipe, model)

    # 2. Ejecutar la permutación (n_jobs=1 suele ser más estable en Windows para evitar errores de pickling)
    result = permutation_importance(
        full_model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
    )

    # 3. Organizar y mostrar resultados
    importancia_df = pd.DataFrame({
        'Variable Real': X_test.columns,
        'Importancia Media': result.importances_mean
    }).sort_values(by='Importancia Media', ascending=False)

    print("\n--- IMPORTANCIA DE VARIABLES REALES (GÉNEROS) ---")
    print(importancia_df)
    
    # Opcional: Registrar en WandB para que no se pierda el dato
    wandb.init(project="modelo_generos_definitivo", name="Importancia_Variables")
    tabla = wandb.Table(dataframe=importancia_df)
    wandb.log({"importancia_variables_reales": wandb.plot.bar(tabla, "Variable Real", "Importancia Media", title="Importancia por Columna Original")})
    wandb.finish()

if __name__ == '__main__':

    # Descarga de modelos
    with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)
    model_genre = (download_model_minio("pd1", "grupo1/models/genres/genres_definitive", claves))
    encoder_genre = (download_model_minio("pd1", "grupo1/models/genres/encoder", claves)) # encoder
    model_kids = (download_model_minio("pd1", "grupo1/models/kids/kids_definitive", claves))
    pipe_kids = (download_model_minio("pd1", "grupo1/models/kids/pipe_kids", claves))
    pipe_genres = (download_model_minio("pd1", "grupo1/models/genres/pipe_genres", claves))
    
    # Descarga datos

    # Kids
    X_test, y_test = extract_definitive_test()
    X_test_trans_kids = pipe_kids.transform(X_test)

    evaluar_modelo_final("modelo_kids_definitivo", "V0", model_kids, X_test_trans_kids, y_test)

    # Generos
    X_test, y_test = extract_definitive_test(columna = "Generos")
    X_test_trans_genre = pipe_genres.transform(X_test)
    y_test_encoded = encoder_genre.transform(y_test)
    evaluar_modelo_final("modelo_generos_definitivo", "V0", model_genre, X_test_trans_genre, y_test_encoded, encoder_genre)
    print(f'Evaluamos importancia')
    calcular_importancia_generos(model_genre, pipe_genres, X_test, y_test)

