from joblib import load
import json
from comun.Server_PD import download_model_minio
from filter_and_divide_data import get_data_models_train_test_latest

# Descarga de datos nuevos kids



# Descarga de modelos
with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
    claves = json.load(archivo)
model_genre = (download_model_minio("pd1", "grupo1/models/genres/genres_definitive", claves))
encoder_genre = (download_model_minio("pd1", "grupo1/models/genres/encoder", claves)) # encoder
model_kids = (download_model_minio("pd1", "grupo1/models/kids/kids_definitive", claves))
pipe_kids = (download_model_minio("pd1", "grupo1/models/kids/pipe_kids", claves))
pipe_genres = (download_model_minio("pd1", "grupo1/models/genres/pipe_genres", claves))

# Kids
X_test, y_test = extract_definitive_test(2)
X_test_trans_kids = pipe_kids.transform(X_test)

predict_kids = model_kids.predict(X_test_trans_kids)

# Generos
X_test, y_test = extract_definitive_test(2, to_predict = "Generos")
X_test_trans_genre = pipe_genres.transform(X_test)

predict_genre = model_genre.predict(X_test_trans_genre)

# Matriz Confusion
wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
                                probs=None, 
                                y_true=y_test, 
                                preds=raw_preds, 
                                class_names=class_names)})
