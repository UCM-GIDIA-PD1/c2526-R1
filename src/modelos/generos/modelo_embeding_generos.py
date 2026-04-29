from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from training_utils import entrenamiento

def build_ensemble(voting='soft'):
    clf1 = KNeighborsClassifier(n_neighbors=6, metric="cosine", weights="distance")
    clf2 = LogisticRegression(C=0.1, solver="saga", max_iter=3000)
    clf3 = XGBClassifier(n_estimators=50, max_depth=10, learning_rate=0.5)

    return VotingClassifier(
        estimators=[('knn', clf1), ('lr', clf2), ('xgb', clf3)],
        voting=voting
    )

if __name__ == '__main__':
    to_predict = "Generos"
    columns = ["Titulo", "Descripcion", "Tags", "Duracion", "Titulo_canal"]
    
    project = "Modelo Ensemble Mixto"
    name = "KNN_LR_XGB_Ensemble con deepLearning"
    preprocess_type = "DeepLearning" 

    max_features = 3000
    ngram = (1,2)
    svd = 150

    # Al menos un valor para que el bucle funcione
    params = {
        "voting": ["soft"] 
    }
    
    score = "F1"
    average = "weighted"
    n_fold = 5
    filtrado = 2

    entrenamiento(project, name, build_ensemble, to_predict, max_features, ngram, svd, 
                  preprocess_type, columns, params, score, average, n_fold, filtrado)