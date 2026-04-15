import comun.filter_and_divide_data as f
import comun.Server_PD as spd
import comun.training_utils as train
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD 
from sklearn.preprocessing import LabelEncoder
import json
if __name__ == '__main__':
    with open("src/Private/claves.json", "r", encoding="utf-8") as archivo:
        claves = json.load(archivo)
    
    #Kids/XGBoost
    X_train, y_train = f.extract_definitive_model_data("Made for kids")
    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Generos", "Subgeneros", "Duracion", "Titulo_canal"]
    preprocess = train.build_preprocess("Word2Vec", columns, X_train, 5000, (1,2), 150)
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("svd", TruncatedSVD(n_components=100, random_state=42))
    ])

    X_train_trans = pipe.fit_transform(X_train, y_train)
    modelo_kids = XGBClassifier()
    modelo_kids.fit(X_train_trans, y_train)

    spd.upload_model_minio(modelo_kids, "pd1", "grupo1/models/kids/kids_definitive", claves)

    #Generos/Knn
    X_train, y_train = f.extract_definitive_model_data("Generos")

    columns = ["Titulo", "Descripcion", "Tags", "Subtitulos", "Made for Kids", "Duracion", "Titulo_canal"]
    preprocess = train.build_preprocess("Word2Vec", columns, X_train, 3000, (1,3), 100)
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("svd", TruncatedSVD(n_components=150, random_state=42))
    ])

    X_train_trans = pipe.fit_transform(X_train, y_train)
    le = LabelEncoder()
    y_train_trans = le.fit_transform(y_train)
    modelo_generos = KNeighborsClassifier()
    modelo_generos.fit(X_train_trans, y_train_trans)
    spd.upload_model_minio(modelo_generos, "pd1", "grupo1/models/genres/genres_definitive", claves)



